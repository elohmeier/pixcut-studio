import { Canvas, StaticCanvas, FabricImage, Point, util } from 'fabric';
import { W, H, encoder, concat, makePLT, inspectJob, validateContours, sleep } from './protocol.js';
import { traceAlpha, rectPath, transformPoint, expand, validateLayout, PX_PER_MM, fitScale } from './geometry.js';
import { USBPrinter, BLEPrinter } from './transports.js';
import { installCutEditor, cutPaths } from './cut-editor.js';
import { validateAnalysisData } from './project.js';
import './style.css';
import './cut-editor.css';

const $ = id => document.getElementById(id);
const canvas = new Canvas('editor', { width: W, height: H, backgroundColor: '#fff', preserveObjectStacking: true, selection: false, enableRetinaScaling: false });
const overlay = $('cuts').getContext('2d');
let exactJob = null, printer = null, busy = false, monitoring = false, drawing = null, drawingObject = null, previewURL, media='sticker';
const sheetHeight=()=>media==='photo'?1800:H;
function setMedia(value){
  media=value;$('media').value=value;canvas.setDimensions({width:W,height:sheetHeight()});$('cuts').height=sheetHeight();
  document.documentElement.style.setProperty('--paper-ratio',media==='photo'?'4/6':'4/7');
  $('paper-tag').innerHTML=media==='photo'?'4 × 6 in <span>300 dpi · photo paper · no cutting</span>':'4 × 7 in <span>300 dpi · sticker paper</span>';
  $('approval-label').textContent=media==='photo'?'I reviewed the photo and loaded 4×6 photo paper.':'I checked the outlines and loaded 4×7 sticker paper.';
  $('print').textContent=media==='photo'?'Print one photo →':'Print & cut one sheet →';
  $('export').textContent=media==='photo'?'Download photo JPEG':'Download print/cut job';
  $('preview-help').textContent=media==='photo'?'Review orientation and framing. This photo will print without cutting.':'Check every green outline before printing.';
  $('show-cuts').disabled=media==='photo';changed();
}
let jobRecord;
try { jobRecord = JSON.parse(localStorage.getItem('pixcut-job') || 'null'); } catch { jobRecord = null; }
if (jobRecord && (!(jobRecord.id === null || Number.isInteger(jobRecord.id)) || !['usb', 'ble'].includes(jobRecord.kind))) jobRecord = null;
function log(message) { $('log').textContent = `${new Date().toLocaleTimeString()}  ${message}\n${$('log').textContent}`.slice(0, 10000); }
function notice(message) { $('job-status').textContent = message; log(message); }
function remember(record) { jobRecord = record; if (record) localStorage.setItem('pixcut-job', JSON.stringify(record)); else localStorage.removeItem('pixcut-job'); controls(); }
function controls() {
  $('print').disabled = busy || !printer || !canvas.getObjects().length || ! $('approved').checked || !!jobRecord || !!drawing;
  $('recovery').hidden = !jobRecord;
  for (const id of ['usb','ble','disconnect','status','resume','cancel','forget','confirm-finished']) $(id).disabled = busy || (['status','resume','cancel'].includes(id) && !printer);
  for (const id of ['images','job-file','contours-file','demo','clear','duplicate','delete','draw','cut-mode','border','preview','export','media','fit','rotate','auto-cut','save-project','open-project']) $(id).disabled = busy;
  for(const id of ['draw','cut-mode','border','contours-file','auto-cut'])$(id).disabled ||= media==='photo';
  $('border').disabled ||= canvas.getActiveObject()?.cutMode==='auto';
  $('usb').disabled ||= !navigator.usb; $('ble').disabled ||= !navigator.bluetooth;
  canvas.skipTargetFind = busy || !!exactJob || !!drawing;
}
async function run(fn) {
  if (busy) return;
  busy = true; controls();
  try { await fn(); } catch (e) { notice(e.message || String(e)); console.error(e); }
  finally { busy = false; controls(); }
}
function changed() { $('approved').checked = false; controls(); paintCuts(); }
function selected() { const o = canvas.getActiveObject(); if (!o || exactJob) throw Error('Select an editable image first. Prepared jobs are locked.'); return o; }
function pathsFor(object) {
  if(object.cutMode==='auto') {
    if(!object.autoCut)throw Error('Generate automatic cuts first.');
    if(Math.abs(object.scaleX-object.autoCut.scaleAtAnalysis)>1e-5||Math.abs(object.scaleY-object.autoCut.scaleAtAnalysis)>1e-5||object.skewX||object.skewY)throw Error('Artwork was resized: reopen Automatic cuts and update contours to preserve millimeter settings.');
    return cutPaths(object.autoCut).map(path=>path.map(p=>transformPoint(p,object.calcTransformMatrix())));
  }
  const paths = object.cutMode === 'rectangle' ? [rectPath(object.width, object.height)] : object.cutMode === 'custom' ? object.customPaths : object.alphaPaths;
  if (!paths?.length) throw Error('This sticker has no outline. Choose Rectangle or draw a custom outline.');
  return expand(paths.map(p => p.map(point => transformPoint(point, object.calcTransformMatrix()))), (object.borderMM || 0) * PX_PER_MM);
}
function allPaths() { return media==='photo'?[]:exactJob ? exactJob.contours : canvas.getObjects().flatMap(pathsFor); }
function paintCuts() {
  overlay.clearRect(0, 0, W, H);
  let paths = [];
  try { paths = allPaths(); $('sheet-summary').textContent = exactJob ? `Prepared job · ${paths.length} cuts` : `${canvas.getObjects().length} stickers · ${paths.length} cuts`; }
  catch (e) { $('layout-message').textContent = e.message; }
  if ($('show-cuts').checked) {
    overlay.strokeStyle = '#008960'; overlay.lineWidth = 3;
    for (const p of paths) { overlay.beginPath(); p.forEach(([x,y],i) => i ? overlay.lineTo(x,y) : overlay.moveTo(x,y)); overlay.closePath(); overlay.stroke(); }
  }
  if (drawing) { overlay.strokeStyle = '#ba6d2b'; overlay.beginPath(); drawing.forEach(([x,y],i)=>i?overlay.lineTo(x,y):overlay.moveTo(x,y)); overlay.stroke(); for(const [x,y] of drawing) {overlay.beginPath();overlay.arc(x,y,5,0,Math.PI*2);overlay.fill();} }
  if (paths.length) try { validateLayout(paths); $('layout-message').textContent = 'Outlines fit the sheet. Review the preview before printing.'; } catch(e) { $('layout-message').textContent = e.message; }
  else $('layout-message').textContent = media==='photo'?'Photo paper: printing only. No cut commands will be sent.':'Add a transparent PNG or open a prepared job to begin.';
}
function updateSelection() {
  const o = canvas.getActiveObject();
  $('selection').textContent = exactJob ? 'Prepared job is locked to preserve its original print and cuts.' : o ? (o.label || 'Sticker') : 'Select artwork on the sheet to edit it.';
  if (o) { $('cut-mode').value = o.cutMode; $('border').value = o.borderMM; $('border-value').textContent = `${o.borderMM} mm`; }
}
canvas.on('object:modified', changed);
// Preview transforms during dragging; expensive overlap validation only after release.
let renderPending = false;
canvas.on('after:render', () => { if (!renderPending) { renderPending=true; requestAnimationFrame(()=>{renderPending=false;paintCuts();}); } });
canvas.on('selection:created', updateSelection); canvas.on('selection:updated', updateSelection); canvas.on('selection:cleared', updateSelection);
canvas.on('mouse:down', event => { if (drawing) { drawing.push([event.scenePoint.x,event.scenePoint.y]); paintCuts(); } });
function stopDrawing() { drawing=null;drawingObject=null; $('drawing-help').hidden=true; $('finish-draw').hidden=true; controls(); paintCuts(); }
document.addEventListener('keydown', e => { if(e.key==='Escape') stopDrawing(); });
$('draw').onclick = () => { try { drawingObject=selected();drawing=[];canvas.discardActiveObject();$('drawing-help').hidden=false;$('finish-draw').hidden=false; controls(); } catch(e) {notice(e.message);} };
$('finish-draw').onclick = () => {
  if (!drawing || drawing.length<3) return notice('Click at least three points around the sticker.');
  try { validateContours([drawing]); const inv=util.invertTransform(drawingObject.calcTransformMatrix()); drawingObject.customPaths=[drawing.map(p=>transformPoint(p,inv))];drawingObject.cutMode='custom'; canvas.setActiveObject(drawingObject); stopDrawing();changed();updateSelection(); } catch(e) { notice(e.message); }
};
function urlOf(file) { return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);}); }
async function addImage(url, label, fullSheet = false) {
  const object = await FabricImage.fromURL(url);
  if (object.width*object.height>40_000_000) throw Error('Image is too large. Resize it below 40 megapixels.');
  const w=Math.max(1,Math.round(object.width*Math.min(1,600/Math.max(object.width,object.height)))), h=Math.max(1,Math.round(object.height*w/object.width));
  const mask=document.createElement('canvas');mask.width=w;mask.height=h;const ctx=mask.getContext('2d',{willReadFrequently:true});ctx.drawImage(object.getElement(),0,0,w,h);
  const rgba=ctx.getImageData(0,0,w,h).data;
  let transparent=false;for(let i=3;i<rgba.length;i+=4) if(rgba[i]<128){transparent=true;break;}
  object.alphaPaths=transparent?traceAlpha(rgba,w,h,object.width,object.height):[rectPath(object.width,object.height)];
  object.cutMode=transparent?'alpha':'rectangle';object.borderMM=fullSheet?0:1;object.label=label;
  const s=fullSheet?W/object.width:media==='photo'?Math.min(W/object.width,sheetHeight()/object.height):Math.min(450/object.width,600/object.height);
  object.set({ originX:'center',originY:'center',left:fullSheet?W/2:350+(canvas.getObjects().length%2)*450,top:fullSheet?H/2:420+Math.floor(canvas.getObjects().length/2)*500,scaleX:s,scaleY:fullSheet?H/object.height:s,cornerColor:'#285342',borderColor:'#285342',cornerSize:24,transparentCorners:false });
  if(media==='photo')object.set({left:W/2,top:sheetHeight()/2});
  canvas.add(object);canvas.setActiveObject(object);object.setCoords();canvas.requestRenderAll();changed();updateSelection();
  return object;
}
function clearSheet() {stopDrawing(); exactJob=null;canvas.clear();canvas.backgroundColor='#fff';changed();updateSelection();}
$('images').onchange = e => run(async()=>{
  if(exactJob) throw Error('Choose New sheet before adding images to a prepared job.');
  for(const file of e.target.files) {if(file.size>20*1024*1024)throw Error('Each image must be smaller than 20 MB.');await addImage(await urlOf(file),file.name);}
  e.target.value='';
});
$('clear').onclick=()=>{if(canvas.getObjects().length && !confirm('Start a new sheet? Download your job first if you want to keep it.'))return;clearSheet();};
$('media').onchange=()=>{if(exactJob){$('media').value=media;return notice('Choose New sheet before changing the paper for a prepared job.');}stopDrawing();setMedia($('media').value);};
function fitObject(o){const margin=media==='photo'?0:60,s=fitScale(o.width,o.height,o.angle,W-2*margin,sheetHeight()-2*margin);o.set({left:W/2,top:sheetHeight()/2,scaleX:s,scaleY:s,skewX:0,skewY:0});}
$('fit').onclick=()=>{try{const o=selected();fitObject(o);o.setCoords();canvas.requestRenderAll();changed();}catch(e){notice(e.message);}};
$('rotate').onclick=()=>{try{const o=selected();o.rotate((o.angle+90)%360);if(media==='photo')fitObject(o);o.setCoords();canvas.requestRenderAll();changed();notice(media==='photo'?'Rotated 90° clockwise and fitted to the photo sheet without cropping.':'Rotated 90° clockwise.');}catch(e){notice(e.message);}};
$('cut-mode').onchange=()=>{try{const o=selected();if($('cut-mode').value==='custom'&&!o.customPaths)throw Error('Draw a custom outline first.');if($('cut-mode').value==='auto'&&!o.autoCut)throw Error('Choose Automatic cuts to detect outlines first.');o.cutMode=$('cut-mode').value;changed();}catch(e){notice(e.message);updateSelection();}};
$('border').oninput=()=>{try{const o=selected();o.borderMM=Number($('border').value);$('border-value').textContent=`${o.borderMM} mm`;changed();}catch{}};
$('delete').onclick=()=>{try{canvas.remove(selected());canvas.discardActiveObject();changed();}catch(e){notice(e.message);}};
$('duplicate').onclick=()=>run(async()=>{const o=selected(),copy=await o.clone();for(const key of ['alphaPaths','customPaths','cutMode','borderMM','label','autoCut'])copy[key]=structuredClone(o[key]);copy.set({left:o.left+50,top:o.top+50,scaleX:o.scaleX,scaleY:o.scaleY,angle:o.angle,skewX:o.skewX,skewY:o.skewY});canvas.add(copy);canvas.setActiveObject(copy);copy.setCoords();changed();});
$('show-cuts').onchange=paintCuts;
$('save-project').onclick=()=>run(async()=>{
  if(exactJob)throw Error('Prepared jobs preserve exact bytes. Download the job instead, or import an editable sheet image.');
  const data={format:'pixcut-studio',version:1,media,objects:canvas.getObjects().map(o=>({...o.toObject(['alphaPaths','customPaths','cutMode','borderMM','label','autoCut']),...Object.fromEntries(['left','top','scaleX','scaleY','angle','skewX','skewY'].map(k=>[k,o[k]]))}))};
  download(encoder.encode(JSON.stringify(data)),'pixcut-project.json','application/json');notice('Saved artwork, transforms, contour parameters, grouping, and manual outlines.');
});
$('open-project').onchange=e=>run(async()=>{
  const file=e.target.files[0];e.target.value='';if(!file)return;if(file.size>60*1024*1024)throw Error('Project exceeds 60 MB.');
  const data=JSON.parse(await file.text());
  if(data.format!=='pixcut-studio'||data.version!==1||!['photo','sticker'].includes(data.media)||!Array.isArray(data.objects)||data.objects.length>100)throw Error('Unsupported project.');
  const images=[];
  for(const item of data.objects){
    if(!/^data:image\/(png|jpeg|webp|svg\+xml);base64,/.test(item.src)||!['alpha','rectangle','custom','auto'].includes(item.cutMode))throw Error('Project must contain embedded images and supported cuts.');
    const o=await FabricImage.fromURL(item.src);if(o.width*o.height>40_000_000)throw Error('Project image too large.');
    for(const key of ['left','top','scaleX','scaleY','angle','skewX','skewY'])if(!Number.isFinite(item[key])||Math.abs(item[key])>100000)throw Error('Invalid image transform.');
    o.set(Object.fromEntries(['left','top','scaleX','scaleY','angle','skewX','skewY','flipX','flipY','originX','originY'].map(k=>[k,item[k]])));
    for(const key of ['alphaPaths','customPaths','cutMode','borderMM','label','autoCut'])o[key]=item[key];
    if(!Number.isFinite(o.borderMM)||o.borderMM<0||o.borderMM>5)throw Error('Invalid border.');
    if(o.autoCut)validateAnalysisData(o.autoCut);
    images.push(o);
  }
  if(canvas.getObjects().length&&!confirm('Replace this sheet with the saved project?'))return;
  clearSheet();setMedia(data.media);for(const o of images){canvas.add(o);o.setCoords();}if(images[0])canvas.setActiveObject(images[0]);canvas.requestRenderAll();changed();updateSelection();notice('Editable project restored. Review the cuts before printing.');
});
$('contours-file').onchange=e=>run(async()=>{
  if(exactJob||canvas.getObjects().length!==1)throw Error('Import contours onto one full-sheet image in a new sheet.');
  const contours=JSON.parse(await e.target.files[0].text());validateLayout(contours);
  const o=canvas.getObjects()[0];o.set({left:W/2,top:H/2,angle:0,scaleX:W/o.width,scaleY:H/o.height,flipX:false,flipY:false});
  const inv=util.invertTransform(o.calcTransformMatrix());o.customPaths=contours.map(p=>p.map(v=>transformPoint(v,inv)));o.cutMode='custom';o.borderMM=0;o.setCoords();canvas.requestRenderAll();changed();updateSelection();e.target.value='';
});
$('job-file').onchange=e=>run(async()=>{
  const files=[...e.target.files];if(files.some(f=>f.size>20*1024*1024))throw Error('Job exceeds 20 MB.');
  const bin=files.find(f=>f.name.endsWith('.bin'));
  let bytes;
  if(bin)bytes=new Uint8Array(await bin.arrayBuffer());else {const plt=files.find(f=>f.name.endsWith('.plt')),jpg=files.find(f=>/\.jpe?g$/i.test(f.name));if(!jpg)throw Error('Select a .bin job, a JPEG + PLT pair, or a 1200×1800 photo JPEG.');bytes=plt?concat(new Uint8Array(await plt.arrayBuffer()),new Uint8Array(await jpg.arrayBuffer())):new Uint8Array(await jpg.arrayBuffer());}
  const job=inspectJob(bytes);if(job.mode==='sticker')validateLayout(job.contours);
  const image=await urlOf(new Blob([job.jpeg],{type:'image/jpeg'}));
  // Decode before replacing the previous sheet.
  const decoded=await FabricImage.fromURL(image); if(decoded.width!==W||decoded.height!==(job.mode==='photo'?1800:H))throw Error('Invalid print image dimensions.');
  clearSheet();setMedia(job.mode);exactJob=job;decoded.set({left:0,top:0,originX:'left',originY:'top',selectable:false,evented:false});canvas.add(decoded);canvas.requestRenderAll();changed();updateSelection();e.target.value='';notice('Prepared job loaded. Original bytes are preserved.');
});
$('demo').onclick=()=>run(async()=>{
  if(exactJob)clearSheet();
  const svg='<svg xmlns="http://www.w3.org/2000/svg" width="360" height="360"><circle cx="180" cy="180" r="155" fill="#f6cc69"/><path d="M95 202 Q180 290 265 202" fill="none" stroke="#294f42" stroke-width="13" stroke-linecap="round"/><circle cx="126" cy="146" r="12" fill="#294f42"/><circle cx="234" cy="146" r="12" fill="#294f42"/><circle cx="90" cy="184" r="18" fill="#e99173"/><circle cx="270" cy="184" r="18" fill="#e99173"/></svg>';
  await addImage('data:image/svg+xml;base64,'+btoa(svg),'Sunny little sticker');
});
async function printCanvas() {
  const render = new StaticCanvas(null,{width:W,height:sheetHeight(),backgroundColor:'#fff',enableRetinaScaling:false});
  try { for(const o of canvas.getObjects())render.add(await o.clone());render.renderAll();return render.toCanvasElement(1); }
  finally {await render.dispose();}
}
async function compile() {
  if(drawing)throw Error('Finish the custom outline first.');
  if(exactJob)return exactJob;
  if(!canvas.getObjects().length)throw Error('Add artwork before exporting.');
  const paths=allPaths();if(media==='sticker')validateLayout(paths);
  const rendered=await printCanvas();const blob=await new Promise(resolve=>rendered.toBlob(resolve,'image/jpeg',.92));
  if(!blob)throw Error('Could not encode the sheet.');
  const jpeg=new Uint8Array(await blob.arrayBuffer());return inspectJob(media==='photo'?jpeg:concat(makePLT(paths),jpeg));
}
function download(bytes,name,type='application/octet-stream') { const a=document.createElement('a'),url=URL.createObjectURL(new Blob([bytes],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000); }
$('export').onclick=()=>run(async()=>{const job=await compile();download(job.bytes,job.mode==='photo'?'pixcut-photo-4x6.jpg':'pixcut-sheet.bin');notice(job.mode==='photo'?'Downloaded a 1200×1800 photo. No cutting data.':`Downloaded a job with ${job.contours.length} cut outlines.`);});
$('preview').onclick=()=>run(async()=>{
  const job=await compile(),image=await createImageBitmap(new Blob([job.jpeg],{type:'image/jpeg'}));
  const view=document.createElement('canvas');view.width=W;view.height=sheetHeight();const ctx=view.getContext('2d');ctx.drawImage(image,0,0);image.close();ctx.strokeStyle='#008960';ctx.lineWidth=3;
  for(const p of job.contours){ctx.beginPath();p.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();}
  if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(await new Promise(r=>view.toBlob(r)));$('preview-image').src=previewURL;$('preview-dialog').showModal();
});
$('close-preview').onclick=()=>$('preview-dialog').close();
$('accept-preview').onclick=()=>{$('preview-dialog').close();notice('Preview reviewed. Confirm the paper is loaded when you’re ready to print.');};
$('approved').onchange=controls;
async function connect(kind) {
  if(printer)await printer.close();printer=null;
  const candidate=kind==='usb'?new USBPrinter():new BLEPrinter();
  await candidate.connect();printer=candidate;
  $('connection').textContent=`● Connected via ${kind==='usb'?'USB':'Bluetooth'}`;
  const s=await printer.status();notice(`Printer: ${s.state==='20'?'ready':`state ${s.state}`} · alerts ${s.alerts}`);
}
$('usb').onclick=()=>run(()=>connect('usb'));$('ble').onclick=()=>run(()=>connect('ble'));
$('disconnect').onclick=()=>run(async()=>{await printer?.close();printer=null;$('connection').textContent='○ Not connected';notice('Disconnected. Any uploaded job may continue on the printer.');});
$('status').onclick=()=>run(async()=>{if(!printer)throw Error('Connect first.');const s=await printer.status();notice(`Printer state ${s.state}, phase ${s.substate}, alerts ${s.alerts}`);});
async function monitor() {
  if(!printer||!jobRecord)throw Error('Connect to the printer with the existing job first.');
  if(printer.kind!==jobRecord.kind)throw Error(`Reconnect using ${jobRecord.kind.toUpperCase()} to monitor this job.`);
  const deviceKey=printer.device?.serialNumber||printer.device?.id;
  if(jobRecord.deviceKey&&deviceKey!==jobRecord.deviceKey)throw Error('This is a different printer. Reconnect to the printer that received the job.');
  if(jobRecord.id===null){const s=await printer.status();notice(`The job-creation response was lost. Printer state ${s.state}, alerts ${s.alerts}. Check the printer before clearing the record; no automatic resend.`);return;}
  monitoring=true;
  $('stop-monitor').hidden=false;
  try {
    let sawActive=!!jobRecord.sawActive;
    for(let i=0;i<180;i++) {
      if(!monitoring){notice('Monitoring stopped. The printer job may continue; its record is saved.');return;}
      const status=await printer.status();
      if(status.state==='40'){sawActive=true;remember({...jobRecord,sawActive});}
      if(status.alerts!=='::0') {notice(`Job ${jobRecord.id} needs attention (${status.alerts}). Check loading, then monitor this same job. No new copy has been sent.`);return;}
      // BLE's large job-info response can truncate on some platforms. Never
      // equate a return to idle with confirmed completion.
      if(printer.kind==='usb') {
        const info=await printer.jobInfo(jobRecord.id),state=String(info?.['job-state']);
        if(state==='9'){notice(`Job ${jobRecord.id} completed successfully.`);remember(null);return;}
        if(state==='7'){notice(`Job ${jobRecord.id} stopped. Check the printer before clearing this job.`);return;}
        notice(info?.['cut-contours']?`Cutting ${info['cutting-progress']} of ${info['cut-contours']} outlines · job ${jobRecord.id}`:`Job ${jobRecord.id} · printer phase ${status.substate}`);
      } else if(sawActive&&status.state==='20') {notice(`Printer is ready again. If your sheet finished printing, choose “My sheet finished printing” below.`);return;}
      else notice(`Job ${jobRecord.id} · printer phase ${status.substate}`);
      await sleep(5000);
    }
    notice('Monitoring timed out. The printer may still be working. Monitor the existing job again; do not resend.');
  } catch(e) {
    log(`Status check: ${e.message}`);
    notice(jobRecord?.uploaded
      ? 'Your sheet was sent to the printer, but live status is unavailable. This does not mean printing failed. If the sheet has finished, confirm below. Otherwise, wait for the printer or reconnect to check status. Do not send another copy.'
      : 'Live status is unavailable. Check the printer: if your sheet finished, confirm below. Otherwise reconnect to check the existing job before sending another copy.');
  } finally {monitoring=false;$('stop-monitor').hidden=true;}
}
$('stop-monitor').onclick=()=>{monitoring=false;};
$('print').onclick=()=>run(async()=>{
  if(!printer||jobRecord||!$('approved').checked)throw Error('Connect, review the sheet, and resolve any existing job first.');
  const job=await compile();
  if(printer.kind==='ble'&&Math.ceil(job.bytes.length/156)>65535)throw Error('This job is too large for BLE. Use USB.');
  $('progress').hidden=false;$('progress').value=0;
  let wake;
  try{
    try{wake=await navigator.wakeLock?.request('screen');}catch{}
    await printer.start(job,id=>{remember({id,kind:printer.kind,deviceKey:printer.device?.serialNumber||printer.device?.id,created:new Date().toISOString()});notice(id===null?'Creating one print/cut job…':`Created job ${id}. Uploading one sheet…`);},p=>{$('progress').value=p;});
    remember({...jobRecord,uploaded:true});
    notice(`Your sheet was sent to the printer. Waiting for it to finish…`);await monitor();
  }finally{await wake?.release();$('approved').checked=false;}
});
$('resume').onclick=()=>run(monitor);
$('confirm-finished').onclick=()=>run(async()=>{if(!jobRecord)return;if(!confirm('Has your sheet fully finished printing and left the printer? This marks it complete locally; it does not send or cancel anything.'))return;remember(null);$('approved').checked=false;$('progress').hidden=true;notice('Print confirmed complete by you. Ready to prepare another sheet.');});
$('cancel').onclick=()=>run(async()=>{if(!jobRecord||!printer)return;if(jobRecord.id===null)throw Error('The job ID is unknown. Check the printer directly.');if(jobRecord.deviceKey&&(printer.device?.serialNumber||printer.device?.id)!==jobRecord.deviceKey)throw Error('Reconnect to the original printer.');if(!confirm(`Cancel printer job ${jobRecord.id}?`))return;await printer.command('cancel-job',{'job-id':jobRecord.id});notice('Cancellation requested. Check status before clearing the record.');});
$('forget').onclick=()=>run(async()=>{if(!confirm('Have you checked that the old job is finished or cancelled? Clearing this record does not cancel a printer job.'))return;remember(null);notice('Local job record cleared.');});
window.addEventListener('beforeunload',e=>{if(busy||jobRecord){e.preventDefault();e.returnValue='';}});
$('support').textContent=`${navigator.usb?'USB available':'USB unavailable'} · ${navigator.bluetooth?'Bluetooth available':'Bluetooth unavailable'}. Use Chrome with HTTPS or localhost.`;
if(jobRecord)notice(`Saved job ${jobRecord.id} may still be active. Reconnect and monitor it before sending another sheet.`);
installCutEditor({selected,changed:()=>{changed();updateSelection();},notice});
controls();paintCuts();
