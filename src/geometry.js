import Clipper from 'clipper-lib';
import { contours as contourGenerator } from 'd3-contour';
import { validateContours } from './protocol.js';
export const PX_PER_MM = 300 / 25.4;
export function fitScale(width, height, angle, pageWidth, pageHeight) {
  const radians = angle * Math.PI / 180, c = Math.abs(Math.cos(radians)), s = Math.abs(Math.sin(radians));
  return Math.min(pageWidth / (width*c + height*s), pageHeight / (width*s + height*c));
}
export function transformPoint([x, y], [a, b, c, d, e, f]) { return [a * x + c * y + e, b * x + d * y + f]; }
export function rectPath(w, h) { return [[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]]; }
export function traceAlpha(rgba, width, height, originalWidth, originalHeight) {
  const values = Float32Array.from({ length: width * height }, (_, i) => rgba[i * 4 + 3] / 255);
  // Outer rings only: no internal holes cut into stickers.
  return contourGenerator().size([width, height]).thresholds([.5])(values)[0].coordinates
    .filter(p => Math.abs(Clipper.Clipper.Area(p[0].map(([X,Y]) => ({X,Y})))) > 4)
    .map(p => p[0].map(([x,y]) => [x / width * originalWidth - originalWidth/2, y / height * originalHeight - originalHeight/2]));
}
const scale = 100;
const integers = paths => paths.map(p => p.map(([x,y]) => ({ X: Math.round(x*scale), Y: Math.round(y*scale) })));
const floats = paths => paths.map(p => p.map(({X,Y}) => [X/scale,Y/scale]));
export function expand(paths, border) {
  if (!border) return paths;
  const offset = new Clipper.ClipperOffset(2, 20);
  const input = integers(paths);
  for (const p of input) if (!Clipper.Clipper.Orientation(p)) p.reverse();
  offset.AddPaths(input, Clipper.JoinType.jtRound, Clipper.EndType.etClosedPolygon);
  const result = new Clipper.Paths(); offset.Execute(result, border*scale);
  return floats(Clipper.Clipper.CleanPolygons(result, 35));
}
export function validateLayout(paths) {
  validateContours(paths);
  for(const path of paths)validateSimplePath(path);
  const input = integers(paths);
  // Reject intersecting or nested stickers, rather than silently changing cuts.
  const bounds = paths.map(p => ({ x0:Math.min(...p.map(v=>v[0])), y0:Math.min(...p.map(v=>v[1])), x1:Math.max(...p.map(v=>v[0])), y1:Math.max(...p.map(v=>v[1])) }));
  for (let i=0; i<input.length; i++) for (let j=i+1; j<input.length; j++) {
    const a=bounds[i],b=bounds[j];
    if (a.x1<b.x0 || b.x1<a.x0 || a.y1<b.y0 || b.y1<a.y0) continue;
    const clip = new Clipper.Clipper(), result = new Clipper.Paths();
    clip.AddPath(input[i],Clipper.PolyType.ptSubject,true); clip.AddPath(input[j],Clipper.PolyType.ptClip,true);
    clip.Execute(Clipper.ClipType.ctIntersection,result,Clipper.PolyFillType.pftNonZero,Clipper.PolyFillType.pftNonZero);
    if (result.some(p => Math.abs(Clipper.Clipper.Area(p)) > scale)) throw Error(`Cut outlines ${i+1} and ${j+1} overlap. Separate the stickers or reduce their border.`);
  }
}
export function validateSimplePath(path){
  const points=path.filter((p,i)=>!i||p[0]!==path[i-1][0]||p[1]!==path[i-1][1]);
  if(points.length>1&&String(points[0])===String(points.at(-1)))points.pop();
  if(points.length<3)throw Error('A cut outline has too few distinct points.');
  const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  let area=0;
  for(let i=0;i<points.length;i++){
    const a=points[i],b=points[(i+1)%points.length];area+=a[0]*b[1]-a[1]*b[0];
    for(let j=i+2;j<points.length;j++){
      if(i===0&&j===points.length-1)continue;
      const c=points[j],d=points[(j+1)%points.length];
      if(Math.max(a[0],b[0])<Math.min(c[0],d[0])||Math.max(c[0],d[0])<Math.min(a[0],b[0])||Math.max(a[1],b[1])<Math.min(c[1],d[1])||Math.max(c[1],d[1])<Math.min(a[1],b[1]))continue;
      const abC=cross(a,b,c),abD=cross(a,b,d),cdA=cross(c,d,a),cdB=cross(c,d,b);
      if(abC*abD<=0&&cdA*cdB<=0)throw Error('A cut outline crosses or touches itself. Move or delete the intersecting vertices.');
    }
  }
  if(Math.abs(area)<.01)throw Error('A cut outline has no usable area.');
}
export function validateSpacing(paths,gap){
  if(gap<=0)return;
  const expanded=paths.map(p=>integers(expand([p],gap/2)));
  for(let i=0;i<expanded.length;i++)for(let j=i+1;j<expanded.length;j++){
    const clip=new Clipper.Clipper(),out=[];clip.AddPaths(expanded[i],Clipper.PolyType.ptSubject,true);clip.AddPaths(expanded[j],Clipper.PolyType.ptClip,true);clip.Execute(Clipper.ClipType.ctIntersection,out,Clipper.PolyFillType.pftNonZero,Clipper.PolyFillType.pftNonZero);
    if(out.some(p=>Math.abs(Clipper.Clipper.Area(p))>scale))throw Error(`Outlines ${i+1} and ${j+1} are closer than the requested spacing. Reduce spacing, merge their groups, or edit the cuts.`);
  }
}
