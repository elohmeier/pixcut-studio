import { encoder, decoder, concat, sleep, usbPacket, bleFrame, extractFrames, jobRequest } from './protocol.js';
const PROPS = ['printer-state', 'printer-sub-state', 'printer-state-alerts'];
const ALLOWED = new Set(['get-prop', 'get-job-info', 'combo-job', 'print-job', 'cancel-job']);
export function resultOf(response, id) {
  if (!response || typeof response !== 'object' || (response.id !== undefined && response.id !== id)) throw Error('Mismatched printer response.');
  if (response.error) throw Error(`Printer error: ${JSON.stringify(response.error)}`);
  if (response.result?.['error-code'] !== undefined && Number(response.result['error-code']) !== 0) throw Error(`Printer error: ${JSON.stringify(response.result)}`);
  if (!('result' in response)) throw Error('Printer response has no result.');
  return response.result;
}
export class Printer {
  constructor() { this.id = 0; this.tail = Promise.resolve(); this.channel = 0; this.kind = 'usb'; }
  exclusive(fn) { const task = this.tail.then(fn); this.tail = task.catch(() => {}); return task; }
  command(method, params) {
    if (!ALLOWED.has(method)) return Promise.reject(Error('Command not allowed.'));
    return this.exclusive(async () => { const id = ++this.id; return resultOf(await this.exchange({ method, params, id }), id); });
  }
  async status() {
    const r = await this.command('get-prop', PROPS);
    if (!Array.isArray(r) || r.length < 3) throw Error('Incomplete printer status.');
    return { state: String(r[0]), substate: String(r[1]), alerts: String(r[2]) };
  }
  async jobInfo(id) { const r = await this.command('get-job-info', { 'job-id': id }); return Array.isArray(r) ? r[0] ?? {} : r; }
  async start(job, onCreated, onProgress) {
    const status = await this.status();
    if (status.state !== '20' || status.alerts !== '::0') throw Error(`Printer is not ready (${status.state}/${status.substate}/${status.alerts}).`);
    onCreated(null); // A lost creation response must still block duplicate jobs.
    const request=jobRequest(job,this.channel);
    const r = await this.command(request.method,request.params);
    const entry = Array.isArray(r) ? r[0] : r, id = entry?.job_id ?? entry?.['job-id'];
    if (!Number.isInteger(id) || id < 0) throw Error('Missing job ID. Do not retry until printer status is checked.');
    onCreated(id); // Persist before sending: a failed upload must never auto-resubmit.
    await this.exclusive(() => this.upload(id, job.bytes, onProgress));
    return id;
  }
}
export class USBPrinter extends Printer {
  constructor(usb = globalThis.navigator?.usb) { super(); this.usb = usb; this.claimed = []; }
  async connect() {
    if (!this.usb) throw Error('WebUSB is unavailable. Use a supported Chrome browser over HTTPS or localhost.');
    this.device = await this.usb.requestDevice({ filters: [{ vendorId: 0x302c, productId: 0x3101 }] });
    try {
      await this.device.open();
      if (!this.device.configuration) await this.device.selectConfiguration(1);
      await this.device.claimInterface(2); this.claimed.push(2);
      try { await this.device.claimInterface(3); this.claimed.push(3); } catch { /* Existing Python driver permits this. */ }
      const iface = this.device.configuration.interfaces.find(i => i.interfaceNumber === 2);
      if (!iface?.alternate.endpoints.some(e => e.endpointNumber === 6 && e.direction === 'out') || !iface.alternate.endpoints.some(e => e.endpointNumber === 6 && e.direction === 'in')) throw Error('Unexpected USB endpoints.');
    } catch (e) { await this.close(); throw e; }
  }
  async timed(promise) {
    let timer;
    try { return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => { void this.close(); reject(Error('USB timed out. Reconnect and check the existing job before retrying.')); }, 10000); })]); }
    finally { clearTimeout(timer); }
  }
  async write(bytes) {
    const r = await this.timed(this.device.transferOut(6, bytes));
    if (r.status !== 'ok' || r.bytesWritten !== bytes.length) throw Error('Incomplete USB write.');
  }
  async read(length) {
    const r = await this.timed(this.device.transferIn(6, length));
    if (r.status !== 'ok' || !r.data) throw Error('USB read failed.');
    return new Uint8Array(r.data.buffer, r.data.byteOffset, r.data.byteLength);
  }
  async exchange(request) { await this.write(encoder.encode('cmd json\n' + JSON.stringify(request))); return JSON.parse(decoder.decode(await this.read(8192))); }
  async upload(id, bytes, progress) {
    for (let offset = 0; offset < bytes.length; offset += 10211) {
      const started = performance.now(), part = bytes.slice(offset, offset + 10211);
      await this.write(usbPacket(id, part));
      const ack = decoder.decode(await this.read(256));
      if (ack !== `cmd data EXTLEN=${part.length + 4} OK\r\n`) throw Error('Invalid USB acknowledgement. Upload stopped; check the existing job.');
      progress(Math.min(offset + part.length, bytes.length) / bytes.length);
      await sleep(Math.max(0, 118 - (performance.now() - started)));
    }
  }
  async close() { const d = this.device; this.device = null; if (d?.opened) { for (const i of this.claimed.splice(0)) { try { await d.releaseInterface(i); } catch {} } try { await d.close(); } catch {} } }
}
const uuid = n => `0000${n}-0000-1000-8000-00805f9b34fb`;
export class BLEPrinter extends Printer {
  constructor(bluetooth = globalThis.navigator?.bluetooth) {
    super(); this.bluetooth = bluetooth; this.kind = 'ble'; this.channel = 2; this.buffer = new Uint8Array(); this.parts = []; this.pending = null; this.ack = null; this.fault = null;
  }
  async connect() {
    if (!this.bluetooth) throw Error('Web Bluetooth is unavailable in this browser. Try USB on desktop Chrome.');
    this.device = await this.bluetooth.requestDevice({ filters: [{ namePrefix: 'Liene PixCut S1' }], optionalServices: [uuid('ff00')] });
    this.device.addEventListener('gattserverdisconnected', () => this.fail(Error('Bluetooth disconnected. Reconnect to check the existing job.')));
    try {
      const server = await this.device.gatt.connect(), service = await server.getPrimaryService(uuid('ff00'));
      this.tx = await service.getCharacteristic(uuid('ff02'));
      this.rx = await service.getCharacteristic(uuid('ff01')); this.flow = await service.getCharacteristic(uuid('ff03'));
      this.rx.addEventListener('characteristicvaluechanged', e => this.receive(new Uint8Array(e.target.value.buffer, e.target.value.byteOffset, e.target.value.byteLength)));
      this.flow.addEventListener('characteristicvaluechanged', e => { const d = e.target.value; if (d.byteLength === 2 && d.getUint8(0) === 1 && d.getUint8(1) === 1) this.ack?.resolve(); });
      await this.rx.startNotifications(); await this.flow.startNotifications();
    } catch (e) { await this.close(); throw e; }
  }
  fail(error) { this.fault = error; this.pending?.reject(error); this.ack?.reject(error); }
  receive(bytes) {
    try {
      this.buffer = concat(this.buffer, bytes);
      if (this.buffer.length > 65536) throw Error('BLE response exceeded its limit.');
      const parsed = extractFrames(this.buffer); this.buffer = parsed.remaining;
      for (const frame of parsed.frames) {
        let body = frame.body;
        if (frame.total > 1) {
          if (frame.number === 1) this.parts = [];
          if (frame.number !== this.parts.length + 1) throw Error('Out-of-order BLE response.');
          this.parts.push(body); if (frame.number !== frame.total) continue;
          body = concat(...this.parts); this.parts = [];
        }
        const r = JSON.parse(decoder.decode(body));
        if (r.method === 'event.rpt_err') { this.fail(Error(`Printer reported: ${JSON.stringify(r.params)}`)); continue; }
        if (r.id === this.pending?.id) this.pending.resolve(r);
      }
    } catch (e) { this.fail(e); }
  }
  waitSlot(name, ms, id) {
    let timer;
    const promise = new Promise((resolve, reject) => {
      this[name] = { id, resolve, reject };
      timer = setTimeout(() => reject(Error(`Bluetooth ${name === 'ack' ? 'upload acknowledgement' : 'response'} timed out. Reconnect to check the printer before retrying any print submission.`)), ms);
    });
    // Install rejection handling before a write can fail or trigger notification.
    promise.catch(() => {});
    return { promise, dispose: () => { clearTimeout(timer); this[name] = null; } };
  }
  async write(frame) {
    if (this.fault) throw this.fault;
    // Preserve the reference client's characteristic-write boundaries. Data
    // frames are 182 bytes; the platform handles the longer JSON command.
    await this.tx.writeValueWithoutResponse(frame);
  }
  async exchange(request) {
    if (this.fault) throw this.fault;
    const wait = this.waitSlot('pending', 10000, request.id);
    try { await this.write(bleFrame(encoder.encode(JSON.stringify(request)))); return await wait.promise; }
    catch (e) { this.fail(e); throw e; }
    finally { wait.dispose(); }
  }
  async upload(id, bytes, progress) {
    const total = Math.ceil(bytes.length / 156);
    if (total > 65535) throw Error('Job is too large for BLE. Use USB.');
    for (let i = 0; i < total; i++) {
      const prefix = new Uint8Array(4); new DataView(prefix.buffer).setUint32(0, id, true);
      const body = concat(prefix, bytes.slice(i * 156, (i + 1) * 156));
      const wait = this.waitSlot('ack', 5000);
      try { await this.write(bleFrame(body, { data: true, sequence: i + 1, total })); await wait.promise; }
      catch (e) { this.fail(e); throw e; } finally { wait.dispose(); }
      progress((i + 1) / total);
    }
  }
  async close() { this.device?.gatt?.disconnect(); this.device = null; }
}
