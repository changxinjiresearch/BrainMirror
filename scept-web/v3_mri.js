'use strict';
/* SCEPT v3 MRI: NIfTI-1 volume parser, orthogonal voxel slices, 3D projection
   and controlled reversible 3D ROI. No interpolation into anatomy is claimed.
   All voxel arrays remain local to the browser. */
(function(){
const DEFAULT_URL='https://niivue.github.io/niivue-demo-images/mni152.nii.gz';
const $=id=>document.getElementById(id);
const s={volume:null,edited:null,baseline:null,offset:null,dimensions:null,center:[0,0,0],slice:[0,0,0],radius:10,amount:.15,limits:[0,1],name:null,affine:null,spacing:null,source:null,revision:0,rotation:.6,tilt:.35,cloud:[],wait:null};
const index=(x,y,z)=>x+s.dimensions[0]*(y+s.dimensions[1]*z);
async function decodeFile(buffer,name){
 if(/\.gz$/i.test(name)){
  if(!('DecompressionStream' in window))throw Error('此浏览器不支持 gzip 解压，请导入 .nii 文件');
  buffer=await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 }
 if(buffer.byteLength<352)throw Error('NIfTI 数据损坏：文件过小');
 const h=new DataView(buffer);
 const le=h.getInt32(0,true)===348?true:h.getInt32(0,false)===348?false:null;
 if(le===null)throw Error('无法识别 NIfTI-1 头部');
 const magic=String.fromCharCode(...new Uint8Array(buffer,344,3));if(magic!=='n+1')throw Error('当前支持单文件 NIfTI-1');
 const rank=h.getInt16(40,le);if(rank!==3&&rank!==4)throw Error('支持三维 MRI 或单帧四维 MRI');
 const dimensions=[h.getInt16(42,le),h.getInt16(44,le),h.getInt16(46,le)];
 if(dimensions.some(x=>x<8)||dimensions.reduce((a,b)=>a*b,1)>12000000)throw Error('MRI 尺寸超出本机安全范围');
 if(rank===4&&h.getInt16(48,le)!==1)throw Error('多时间点影像须先单独选择一个时间点');
 const datatype=h.getInt16(70,le),bitpix=h.getInt16(72,le);
 const spec={2:[1,'getUint8'],4:[2,'getInt16'],8:[4,'getInt32'],16:[4,'getFloat32'],64:[8,'getFloat64'],512:[2,'getUint16']}[datatype];
 if(!spec||bitpix!==spec[0]*8)throw Error('MRI 数据类型暂不支持：'+datatype);
 const off=Math.ceil(h.getFloat32(108,le)),len=dimensions.reduce((a,b)=>a*b,1);
 if(off<352||off+len*spec[0]>buffer.byteLength)throw Error('MRI 体素数据不足');
 const slope=h.getFloat32(112,le),intercept=h.getFloat32(116,le),m=slope===0?1:slope,t=slope===0?0:intercept;
 const values=new Float32Array(len);
 let min=Infinity,max=-Infinity;
 for(let i=0;i<len;i++){
  const v=h[spec[1]](off+i*spec[0],le)*m+t;
  if(!Number.isFinite(v))throw Error('发现无效 MRI 强度');
  values[i]=v;min=Math.min(v,min);max=Math.max(v,max);
 }
 const sp=[h.getFloat32(80,le),h.getFloat32(84,le),h.getFloat32(88,le)];
 const spacing=sp.map(x=>Number.isFinite(x)&&x>0?x:null);
 if(spacing.some(v=>v===null))throw Error('体素尺寸缺失或无效');
 const samples=[],step=Math.max(1,Math.floor(len/20000));
 for(let i=0;i<len;i+=step)samples.push(values[i]);
 samples.sort((a,b)=>a-b);
 const lo=samples[Math.floor(samples.length*.02)],hi=samples[Math.floor(samples.length*.98)];
 if(hi<=lo)throw Error('MRI 强度范围异常');
 const sform=h.getInt16(254,le),qform=h.getInt16(252,le);
 const affine=sform>0?[0,1,2].map(r=>Array.from({length:4},(_,c)=>h.getFloat32(280+16*r+4*c,le))):null;
 return{values,dimensions,spacing,limits:[lo,hi],affine,orient:sform>0?'sform':qform>0?'qform':'未知'};
}
const sample=(v,lo,hi)=>Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
function voxelValue(x,y,z,edited){const d=s.dimensions;if(!d)return 0;return(edited?s.edited:s.baseline)[index(Math.min(d[0]-1,Math.max(0,x)),Math.min(d[1]-1,Math.max(0,y)),Math.min(d[2]-1,Math.max(0,z)))]}
function voxelIsRoi(x,y,z){const c=s.center,r=s.radius;return(x-c[0])**2+(y-c[1])**2+(z-c[2])**2<=r*r}
function computeEdit(){
 const d=s.dimensions;if(!d)return;const values=new Float32Array(s.baseline),r=s.radius,c=s.center,v=s.amount*(s.limits[1]-s.limits[0]);
 let count=0;
 for(let z=Math.max(0,Math.floor(c[2]-r));z<=Math.min(d[2]-1,Math.ceil(c[2]+r));z++)
 for(let y=Math.max(0,Math.floor(c[1]-r));y<=Math.min(d[1]-1,Math.ceil(c[1]+r));y++)
 for(let x=Math.max(0,Math.floor(c[0]-r));x<=Math.min(d[0]-1,Math.ceil(c[0]+r));x++){
  if(!voxelIsRoi(x,y,z))continue;values[index(x,y,z)]+=v;count++;
 }
 s.edited=values;s.revision++;
 $('mriInfo').textContent=d.join('×')+' 体素 · '+s.spacing.join('×')+' mm · 3D ROI '+count+' 体素 · '+s.orient+' · 探索性强度修改';
 render();
}
function renderPlane(plane){
 const d=s.dimensions;
 const specs=plane===0?{width:d[0],height:d[1],axis:(u,v)=>[u,d[1]-1-v,s.slice[2]],click:(u,v)=>[u,d[1]-1-v,s.slice[2]]}:
 plane===1?{width:d[0],height:d[2],axis:(u,v)=>[u,s.slice[1],d[2]-1-v],click:(u,v)=>[u,s.slice[1],d[2]-1-v]}:
 {width:d[1],height:d[2],axis:(u,v)=>[s.slice[0],u,d[2]-1-v],click:(u,v)=>[s.slice[0],u,d[2]-1-v]};
 const canvas=$(['viewAxial','viewCoronal','viewSagittal'][plane]),ctx=canvas.getContext('2d');
 const tmp=document.createElement('canvas');tmp.width=specs.width;tmp.height=specs.height;
 const tctx=tmp.getContext('2d');const im=tctx.createImageData(specs.width,specs.height);
 const lo=s.limits[0],hi=s.limits[1],data=s.edited;
 for(let v=0;v<specs.height;v++)for(let u=0;u<specs.width;u++){
  const p=specs.axis(u,v),ii=(v*specs.width+u)*4;
  let intensity=Math.floor(255*sample(data[index(...p)],lo,hi));
  im.data[ii]=intensity;im.data[ii+1]=intensity;im.data[ii+2]=intensity;im.data[ii+3]=255;
  if(voxelIsRoi(...p)){
    im.data[ii]=Math.min(255,intensity+45);im.data[ii+1]=Math.min(255,intensity+65);im.data[ii+2]=Math.max(0,intensity-45);
  }
 }
 tctx.putImageData(im,0,0);ctx.fillStyle='#071720';ctx.fillRect(0,0,canvas.width,canvas.height);
 ctx.imageSmoothingEnabled=false;ctx.drawImage(tmp,0,0,canvas.width,canvas.height);
 canvas.__size=[specs.width,specs.height];canvas.__select=specs.click;
}
function updateCloud(){
 const d=s.dimensions,arr=[],step=Math.max(2,Math.ceil(Math.cbrt(s.baseline.length/13000)));
 const lo=s.limits[0],hi=s.limits[1],cut=lo+(hi-lo)*.55;
 for(let zz=0;zz<d[2];zz=zz+step)for(let yy=0;yy<d[1];yy=yy+step)for(let xx=0;xx<d[0];xx=xx+step){
  const intensity=s.edited[index(xx,yy,zz)];
  if(intensity>cut){
   const coords=[(xx/d[0]-.5)*2,(yy/d[1]-.5)*2,(zz/d[2]-.5)*2];
   arr.push({p:coords,intensity:sample(intensity,lo,hi),inRoi:voxelIsRoi(xx,yy,zz)});
  }
 }
 s.cloud=arr.slice(0,16000);
}
function drawCloud(){
 const canvas=$('view3d'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
 ctx.fillStyle='#061a24';ctx.fillRect(0,0,w,h);
 if(!s.dimensions)return;
 const ca=Math.cos(s.rotation),sa=Math.sin(s.rotation),ct=Math.cos(s.tilt),st=Math.sin(s.tilt);
 const points=s.cloud.map(a=>{const p=a.p,x=p[0]*ca-p[1]*sa,y=p[0]*sa+p[1]*ca,z=p[2]*ct-y*st;return{x:w/2+x*110,y:h/2+(p[2]*st+y*ct)*105,depth:z,intensity:a.intensity,roi:a.inRoi}});
 points.sort((a,b)=>a.depth-b.depth);
 for(const p of points){const g=Math.floor(85+p.intensity*145);ctx.fillStyle=p.roi?'rgba(81,235,179,.85)':`rgba(${g},${Math.min(255,g+20)},${Math.min(255,g+37)},.32)`;ctx.fillRect(p.x,p.y,p.roi?3:2,p.roi?3:2)}
 ctx.fillStyle='#86a7ad';ctx.font='12px sans-serif';ctx.fillText('MRI 体素点云（非脑表面分割）',12,19);
}
function render(){
 if(!s.dimensions)return;
 for(let plane=0;plane<3;plane++)renderPlane(plane);
 updateCloud();drawCloud();
 $('sliceI').value=String(s.slice[0]);$('sliceJ').value=String(s.slice[1]);$('sliceK').value=String(s.slice[2]);
 $('mriRadiusValue').textContent=s.radius+' 体素';
 $('mriAmountValue').textContent=(s.amount>=0?'+':'')+s.amount.toFixed(2);
 $('roiCoords').textContent='三维 ROI 中心：['+s.center.join(', ')+'] · 三视图切片坐标：['+s.slice.join(', ')+']';
 window.SCEPT_PATCH?.refresh();
}
async function install(data,name,source){
 s.dimensions=data.dimensions;s.spacing=data.spacing;s.limits=data.limits;s.affine=data.affine;s.orient=data.orient;
 s.baseline=data.values;s.edited=new Float32Array(data.values);s.name=name;s.source=source;
 s.center=data.dimensions.map(x=>Math.floor(x/2));
 s.slice=[...s.center];
 s.radius=Math.max(3,Math.min(10,Math.floor(Math.min(...data.dimensions)/6)));
 s.amount=.15;
 for(const [id,axis]of [['sliceI',0],['sliceJ',1],['sliceK',2]]){
  const el=$(id);el.max=String(data.dimensions[axis]-1);el.value=String(s.slice[axis]);
 }
 $('mriRadius').value=String(s.radius);$('mriAmount').value='.15';
 computeEdit();
 $('mriSource').textContent=name;
 $('mriStatus').textContent='真实 NIfTI 已加载 · '+data.dimensions.join('×');
}
async function loadDefault(){
 $('mriStatus').textContent='正在获取真实 MRI 衍生模板…';
 try{
  const res=await fetch(DEFAULT_URL,{mode:'cors'});
  if(!res.ok)throw Error('HTTP '+res.status);
  await install(await decodeFile(await res.arrayBuffer(),'mni152.nii.gz'),'MNI152 脑 MRI 模板','niivue-demo-images/mni152.nii.gz');
 }catch(e){
  $('mriStatus').textContent='MRI 自动加载失败：'+e.message+'，请手动导入 .nii/.nii.gz';
  $('mriInfo').textContent='未加载 MRI；不会使用合成条纹假装 MRI。';
 }
}
$('mriFile').onchange=async e=>{const f=e.target.files?.[0];if(!f)return;try{
 if(f.size>120000000)throw Error('单文件上限 120 MB');
 await install(await decodeFile(await f.arrayBuffer(),f.name),f.name,'local_browser_file');
 }catch(ex){$('mriStatus').textContent='MRI 导入失败：'+ex.message}};
for(const [id,axis]of [['sliceI',0],['sliceJ',1],['sliceK',2]])$(id).oninput=e=>{
 s.slice[axis]=Number(e.target.value);s.revision++;render();
};
$('mriRadius').oninput=e=>{s.radius=Number(e.target.value);computeEdit()};
$('mriAmount').oninput=e=>{s.amount=Number(e.target.value);computeEdit()};
$('mriUndo').onclick=()=>{$('mriAmount').value='0';s.amount=0;computeEdit()};
for(const id of ['viewAxial','viewCoronal','viewSagittal']){
 const canvas=$(id);canvas.addEventListener('pointerdown',e=>{
  if(!s.dimensions)return;const rect=canvas.getBoundingClientRect(),p=canvas.__size;
  const u=Math.min(p[0]-1,Math.max(0,Math.floor((e.clientX-rect.left)/rect.width*p[0]))),
  v=Math.min(p[1]-1,Math.max(0,Math.floor((e.clientY-rect.top)/rect.height*p[1])));
  s.center=canvas.__select(u,v);computeEdit();
 });
}
let drag=null;const c=$('view3d');
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};
c.onpointermove=e=>{if(!drag)return;s.rotation+=(e.clientX-drag[0])*.01;s.tilt+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];drawCloud()};
c.onpointerup=()=>drag=null;
function sampleSlice(edited=true,axis=2){
 if(!s.dimensions)throw Error('MRI 未加载');
 const d=s.dimensions,a=s.slice[axis],lo=s.limits[0],hi=s.limits[1],out=new Float32Array(3*224*224);
 const select=(u,v)=>{
  const x=axis===0?a:Math.round(u*(d[0]-1)/223),y=axis===1?a:Math.round(v*(d[1]-1)/223),z=axis===2?a:Math.round(v*(d[2]-1)/223);
  return axis===2?[x,Math.round(v*(d[1]-1)/223),z]:axis===1?[x,y,Math.round(v*(d[2]-1)/223)]:[x,Math.round(u*(d[1]-1)/223),z];
 };
 for(let v=0;v<224;v++)for(let u=0;u<224;u++){
  const p=select(u,v),val=sample(voxelValue(...p,edited),lo,hi),i=v*224+u;
  out[i]=out[224*224+i]=out[2*224*224+i]=val;
 }
 return out;
}
function makeCnnInput(edited=true){
 if(!s.dimensions)throw Error('MRI 尚未加载');
 const d=s.dimensions,depth=s.slice[2],lo=s.limits[0],hi=s.limits[1];
 const out=new Float32Array(256*256*3);
 for(let y=0;y<256;y++)for(let x=0;x<256;x++){
  const vx=Math.min(d[0]-1,Math.round((x/255)*(d[0]-1)));
  const vy=Math.min(d[1]-1,Math.round((y/255)*(d[1]-1)));
  const intensity=Math.floor(255*sample(voxelValue(vx,vy,depth,edited),lo,hi));
  const idx=(y*256+x)*3;out[idx]=out[idx+1]=out[idx+2]=intensity;
 }
 return out;
}
async function checksumVolume(){return s.baseline?crypto.subtle.digest('SHA-256',s.baseline.buffer).then(buf=>Array.from(new Uint8Array(buf)).map(v=>v.toString(16).padStart(2,'0')).join('')):null}
window.SCEPT_MRI={loadDefault,get state(){return{loaded:!!s.baseline,dimensions:s.dimensions,spacing:s.spacing,affine:s.affine,orient:s.orient,source:s.source,name:s.name,center:[...s.center],slice:[...s.slice],radius:s.radius,amount:s.amount,revision:s.revision}},makeSlice:sampleSlice,makeCnnInput,sourceHash:checksumVolume,decodeFile,applyEdit:computeEdit};
loadDefault();
})();