export const W = 1200, H = 2100;
export const encoder = new TextEncoder(), decoder = new TextDecoder();
export const concat = (...parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.length; }
  return out;
};
export const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
// Python round uses ties-to-even, unlike Math.round.
export function roundEven(n) { const f = Math.floor(n), r = n - f; return r === .5 ? f + (f % 2 !== 0 ? 1 : 0) : Math.round(n); }
export function toDevice([x, y]) {
  return [roundEven((H - (1.0559 * y - 102.6)) * 3.25), roundEven((1.0576 * x - 6.495) * 3.25)];
}
export function fromDevice([x, y]) { return [(y / 3.25 + 6.495) / 1.0576, (H - x / 3.25 + 102.6) / 1.0559]; }
export function validateContours(contours, tolerance = 0) {
  if (!Array.isArray(contours) || !contours.length || contours.length > 500) throw Error('Provide 1–500 cut outlines.');
  let points = 0;
  for (const path of contours) {
    if (!Array.isArray(path) || path.length < 3) throw Error('Each cut outline needs at least three points.');
    for (const p of path) {
      if (!Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite) || p[0] < -tolerance || p[0] > W - 1 + tolerance || p[1] < -tolerance || p[1] > H - 1 + tolerance) throw Error('A cut outline is outside the sheet. Move or resize the sticker.');
    }
    points += path.length;
  }
  if (points > 50000) throw Error('Too many cut points. Simplify the outlines.');
}
export function makePLT(contours) {
  validateContours(contours);
  let text = 'IN VER0.1.0 KP42';
  for (const path of contours) {
    const pts = path.map(toDevice);
    if (String(pts[0]) !== String(pts.at(-1))) pts.push(pts[0]);
    text += ` U${pts[0].join(',')}` + pts.slice(1).map(p => ` D${p.join(',')}`).join('');
  }
  return encoder.encode(text + ' U6476,0  @ ');
}
export function parsePLT(bytes) {
  const text = decoder.decode(bytes);
  if (!/^IN VER0\.1\.0 KP42(?:\s+[UD]-?\d+,-?\d+)+\s+@\s*$/.test(text)) throw Error('Unsupported cut file. Expected native PixCut PLT with KP42.');
  const paths = []; let path;
  for (const match of text.matchAll(/([UD])(-?\d+),(-?\d+)/g)) {
    if (match[1] === 'U') { if (path?.length > 1) paths.push(path); path = []; }
    if (!path) throw Error('Cut file starts with a pen-down command.');
    path.push(fromDevice([Number(match[2]), Number(match[3])]));
  }
  if (path?.length > 1) paths.push(path);
  for (const p of paths) if (Math.hypot(p[0][0] - p.at(-1)[0], p[0][1] - p.at(-1)[1]) > .01) throw Error('Cut outlines must be closed.');
  validateContours(paths, .2);
  return paths;
}
export function jpegSize(bytes) {
  if (bytes[0] !== 255 || bytes[1] !== 216 || bytes.at(-2) !== 255 || bytes.at(-1) !== 217) throw Error('Invalid JPEG markers.');
  let i = 2;
  while (i + 3 < bytes.length) {
    if (bytes[i++] !== 255) throw Error('Invalid JPEG segment.');
    while (bytes[i] === 255) i++;
    const marker = bytes[i++];
    if (marker === 218 || marker === 217) break;
    const len = bytes[i] * 256 + bytes[i + 1];
    if (len < 2 || i + len > bytes.length) throw Error('Truncated JPEG.');
    if ([192, 193, 194].includes(marker)) return [bytes[i + 5] * 256 + bytes[i + 6], bytes[i + 3] * 256 + bytes[i + 4]];
    i += len;
  }
  throw Error('JPEG dimensions not found.');
}
export function inspectJob(bytes) {
  if (bytes.length > 20 * 1024 * 1024) throw Error('Job exceeds 20 MB.');
  if(bytes[0]===255&&bytes[1]===216){
    if(String(jpegSize(bytes))!=='1200,1800')throw Error('A photo-only job must be a 1200 × 1800 JPEG.');
    return {mode:'photo',bytes,jpeg:bytes,plt:new Uint8Array(),contours:[]};
  }
  let split = -1;
  for (let i = 0; i < bytes.length - 1; i++) if (bytes[i] === 255 && bytes[i + 1] === 216) { split = i; break; }
  if (split < 1) throw Error('Expected PLT followed by JPEG.');
  const plt = bytes.slice(0, split), jpeg = bytes.slice(split);
  if (String(jpegSize(jpeg)) !== `${W},${H}`) throw Error('Print image must be 1200 × 2100 pixels.');
  return { mode:'sticker', bytes, plt, jpeg, contours: parsePLT(plt) };
}
export function jobRequest(job, channel=0){
  if(job.mode==='photo')return {method:'print-job',params:{channel,copies:1,'file-size':job.jpeg.length,'media-size':5012,'media-type':2010,'job-type':0}};
  return {method:'combo-job',params:combo(job,channel)};
}
export function combo(job, channel = 0) {
  return [
    { method: 'print-job', params: { channel, copies: 1, 'file-size': job.jpeg.length, 'media-size': 5013, 'media-type': 2030, 'job-type': 600 } },
    { method: 'cut-job', params: { 'file-size': job.plt.length, 'job-type': 600 } },
  ];
}
export function usbPacket(id, data) {
  if (!Number.isInteger(id) || id < 0 || id > 0xffffffff || data.length < 1 || data.length > 10211) throw Error('Invalid USB packet.');
  const prefix = new Uint8Array(4); new DataView(prefix.buffer).setUint32(0, id, true);
  return concat(encoder.encode(`cmd data EXTLEN=${data.length + 4}\n`), prefix, data);
}
export function bleFrame(body, { sequence = 1, total = 1, data = false } = {}) {
  if (body.length > 896 || total > 65535) throw Error('BLE frame too large.');
  const frame = new Uint8Array(body.length + 22), view = new DataView(frame.buffer);
  frame.set([0x7e, 100, 0, data ? 2 : 1, 6, data ? 2 : 3]);
  view.setUint32(10, sequence, true); view.setUint16(14, total, true); view.setUint16(16, sequence, true);
  view.setUint16(18, body.length | (data ? 0x2000 : 0), true); frame.set(body, 20);
  frame[frame.length - 2] = frame.slice(1, -2).reduce((a, b) => (a + b) & 255, 0);
  frame[frame.length - 1] = 0x7e; return frame;
}
export function extractFrames(buffer) {
  const frames = []; let offset = 0;
  while (offset < buffer.length) {
    if (buffer[offset] !== 0x7e) throw Error('Invalid BLE frame boundary.');
    if (buffer.length - offset < 20) break;
    const view = new DataView(buffer.buffer, buffer.byteOffset + offset);
    const attr = view.getUint16(18, true), size = (attr & 1023) + 22;
    if (buffer.length - offset < size) break;
    const f = buffer.slice(offset, offset + size);
    if (f.at(-1) !== 0x7e || f.slice(1, -2).reduce((a, b) => (a + b) & 255, 0) !== f.at(-2)) throw Error('BLE checksum mismatch.');
    if ((attr >> 10) & 7) throw Error('Encrypted BLE responses are not supported.');
    frames.push({ body: f.slice(20, -2), total: view.getUint16(14, true), number: view.getUint16(16, true) }); offset += size;
  }
  return { frames, remaining: buffer.slice(offset) };
}
