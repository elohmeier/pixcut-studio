import test from 'node:test';
import assert from 'node:assert/strict';
import { makePLT, parsePLT, usbPacket, bleFrame, extractFrames, encoder, concat, toDevice, fromDevice, inspectJob, combo } from '../src/protocol.js';
import { validateLayout, transformPoint, expand, traceAlpha } from '../src/geometry.js';
import { USBPrinter, BLEPrinter, resultOf } from '../src/transports.js';
import { jobRequest } from '../src/protocol.js';
import { fitScale } from '../src/geometry.js';
test('4×6 photo uses a standalone print job with no cutting payload',()=>{
  // Minimal SOF fixture exercises header inspection, not JPEG decoding.
  const bytes=new Uint8Array([255,216,255,192,0,11,8,7,8,4,176,1,1,17,0,255,217]);
  const job=inspectJob(bytes),request=jobRequest(job,2);
  assert.equal(job.mode,'photo');assert.equal(job.bytes,job.jpeg);
  assert.equal(job.plt.length,0);assert.deepEqual(job.contours,[]);
  assert.deepEqual(request,{method:'print-job',params:{channel:2,copies:1,'file-size':bytes.length,'media-size':5012,'media-type':2010,'job-type':0}});
  const landscape=bytes.slice();landscape.set([4,176,7,8],7);
  assert.throws(()=>inspectJob(landscape),/1200 × 1800/);
});
test('quarter-turn fits landscape photos to portrait paper without cropping',()=>{
  assert.equal(fitScale(1800,1200,0,1200,1800),2/3);
  for(const angle of [90,270])assert.ok(Math.abs(fitScale(1800,1200,angle,1200,1800)-1)<1e-12);
  const angle=35,s=fitScale(1800,1200,angle,1200,1800),r=angle*Math.PI/180;
  assert.ok(s*(1800*Math.cos(r)+1200*Math.sin(r))<=1200.000001);
});
const rectangle = [[150,250],[400,250],[400,550],[150,550]];
test('calibration matches the Python profile and round trips',()=>{
  assert.deepEqual(toDevice([150,250]),[6301,494]);
  for(const p of rectangle) assert.ok(Math.hypot(...fromDevice(toDevice(p)).map((v,i)=>v-p[i]))<.3);
  const paths=parsePLT(makePLT([rectangle]));assert.equal(paths.length,1);assert.equal(paths[0].length,5);
});
test('rejects unsupported plotter commands, open paths and out-of-sheet geometry',()=>{
  assert.throws(()=>makePLT([[[0,0],[1200,0],[5,5]]]));
  assert.throws(()=>parsePLT(encoder.encode('IN VER0.1.0 KP100 U100,100 D0,0 @ ')));
  assert.throws(()=>parsePLT(encoder.encode('IN VER0.1.0 KP42 U100,100 D200,200 D300,300 @ ')));
});
test('USB packet carries little-endian job ID in every frame',()=>{
  const packet=usbPacket(0x12345678,new Uint8Array([7,8]));
  const header=encoder.encode('cmd data EXTLEN=6\n');
  assert.deepEqual(packet.slice(header.length),new Uint8Array([0x78,0x56,0x34,0x12,7,8]));
  assert.throws(()=>usbPacket(1,new Uint8Array(10212)));
});
test('BLE handles split notifications and back-to-back frames, checks corruption',()=>{
  const a=bleFrame(encoder.encode('{"id":1}')),b=bleFrame(encoder.encode('{"id":2}'));
  const first=extractFrames(a.slice(0,12));assert.equal(first.frames.length,0);
  const complete=extractFrames(concat(first.remaining,a.slice(12),b));assert.equal(complete.frames.length,2);assert.equal(complete.remaining.length,0);
  const corrupt=a.slice();corrupt[21]^=1;assert.throws(()=>extractFrames(corrupt));
  assert.equal(bleFrame(new Uint8Array(160),{data:true}).length,182);
});
test('combo declares separate byte sizes and transport channel',()=>{
  const c=combo({jpeg:new Uint8Array(20),plt:new Uint8Array(7)},2);
  assert.equal(c[0].params.channel,2);assert.equal(c[0].params['media-size'],5013);assert.equal(c[1].params['file-size'],7);
});
test('response correlation and errors are enforced',()=>{
  assert.throws(()=>resultOf({id:2,result:{}},1));assert.throws(()=>resultOf({id:1,error:{code:3}},1));
  assert.deepEqual(resultOf({id:1,result:[20]},1),[20]);
});
test('rotated cut geometry is transformed in scene space; border expands outward',()=>{
  assert.deepEqual(transformPoint([20,10],[0,2,-2,0,100,100]),[80,140]);
  const enlarged=expand([rectangle],10);assert.ok(enlarged[0].some(p=>p[0]<150));validateLayout(enlarged);
});
test('overlapping and nested cuts are rejected; separated cuts accepted',()=>{
  assert.throws(()=>validateLayout([rectangle,rectangle.map(([x,y])=>[x+20,y+20])]),/overlap/);
  assert.throws(()=>validateLayout([rectangle,[[200,300],[210,300],[210,310],[200,310]]]),/overlap/);
  validateLayout([rectangle,rectangle.map(([x,y])=>[x+500,y+500])]);
});
test('alpha tracing retains separate islands without cutting inner holes',()=>{
  const rgba=new Uint8Array(20*20*4);
  for(let y=2;y<18;y++)for(let x=2;x<18;x++)if(x<6||x>13)rgba[(y*20+x)*4+3]=255;
  const p=traceAlpha(rgba,20,20,200,200);assert.equal(p.length,2);
});
test('USB stops on bad acknowledgement and does not retry or create another job',async()=>{
  const p=new USBPrinter();let writes=0;
  p.write=async()=>{writes++;};p.read=async()=>encoder.encode('cmd data EXTLEN=99 OK\r\n');
  await assert.rejects(p.upload(1,new Uint8Array(15000),()=>{}),/acknowledgement/);assert.equal(writes,1);
});
test('USB refuses non-ready printer before job creation',async()=>{
  const p=new USBPrinter();p.status=async()=>({state:'60',alerts:'::5306',substate:'3015'});let created=false;p.command=async()=>{created=true;};
  await assert.rejects(p.start({},()=>{},()=>{}),/not ready/);assert.equal(created,false);
});
test('BLE upload registers ACK before write and uses 156-byte job fragments',async()=>{
  const p=new BLEPrinter();const frames=[];let progress=0;
  p.tx={writeValueWithoutResponse:async frame=>{frames.push(frame);p.ack.resolve();}};
  await p.upload(42,new Uint8Array(400),v=>{progress=v;});
  assert.deepEqual(frames.map(f=>f.length),[182,182,114]);assert.equal(progress,1);
  for(const f of frames){const {frames:[parsed]}=extractFrames(f);assert.equal(new DataView(parsed.body.buffer).getUint32(0,true),42);}
});
test('BLE event errors stop subsequent writes',async()=>{
  const p=new BLEPrinter();p.receive(bleFrame(encoder.encode(JSON.stringify({method:'event.rpt_err',params:{code:-8013}}))));
  await assert.rejects(p.write(new Uint8Array()),/Printer reported/);
});
test('invalid job never reaches device',()=>{assert.throws(()=>inspectJob(encoder.encode('bad')));});
