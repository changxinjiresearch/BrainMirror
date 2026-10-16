'use strict';
const $ = id => document.getElementById(id);
const state={trace:null,frame:0,node:42,brainYaw:0.5,brainPitch:-0.28,netYaw:0.6,netPitch:-0.18,brainScale:1,netScale:1,brainProjected:[],netProjected:[],error:null,mode:'template',volume:null,mask:null,surfacePoints:[],modelId:'brainae',anatomyBusy:false};
const style={cyan:'#6de0d3',blue:'#66c7e7',orange:'#f5a46c',muted:'#7b9cb6'};
const IXI_PUBLIC_URL='https://raw.githubusercontent.com/changxinjiresearch/BrainMirror/main/data/ixi_population_467.json';
const IXI_PUBLIC_SHA256='36b25fbf1f6749b309cc93adc811ee2137fa83fdab89cf72ba87886e542cda21';
function nodeParts(i){return [Math.floor(i/64),Math.floor(i%64/16),Math.floor(i%16/4),i%4];}
function frame(){return state.trace.frames[state.frame]}
function baseline(){return state.trace.frames[0]}
function font(ctx,color,size=12,weight='normal'){ctx.fillStyle=color;ctx.font=`${weight} ${size}px system-ui,Segoe UI,sans-serif`;}
function prepCanvas(canvas){let rect=canvas.getBoundingClientRect();const w=Math.max(1,rect.width),h=Math.max(1,rect.height),dpr=Math.min(2,devicePixelRatio||1);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return {ctx,w,h};}
function project(p,yaw,pitch,scale,w,h){let [x,y,z]=p;let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);let xx=cy*x+sy*z,zz=-sy*x+cy*z,yy=cp*y-sp*zz,zd=sp*y+cp*zz;let k=scale/(1+zd*.024);return [w/2+xx*k,h/2-yy*k,zd]}
function drawBrain(){
 const {ctx,w,h}=prepCanvas($('brainCanvas'));
 if(!state.volume){drawSlices();return;}
 const v=state.volume,pts=state.surfacePoints,center=v.shape.map(n=>(n-1)/2);
 const scale=Math.min(w,h)*.037*state.brainScale*(32/Math.max(...v.shape)) * 2.1;
 const projected=pts.map(item=>({p:project(item.p.map((x,i)=>x-center[i]),state.brainYaw,state.brainPitch,scale,w,h),item})).sort((a,b)=>a.p[2]-b.p[2]);
 for(const {p,item} of projected){let t=Math.max(.1,Math.min(1,item.v));const r=item.roi?248:Math.round(60+t*80),g=item.roi?156:Math.round(110+t*90),bl=item.roi?80:Math.round(159+t*76);ctx.beginPath();ctx.fillStyle=`rgba(${r},${g},${bl},${.34+t*.45})`;ctx.arc(p[0],p[1],1.5+t*1.4,0,Math.PI*2);ctx.fill()}
 font(ctx,'#799bba',11);ctx.fillText(`REAL VOLUME · ${v.shape.join('×')} (sampled)`,16,h-24);drawSlices();
}
function index3(shape,x,y,z){return x+shape[0]*(y+shape[1]*z)}
function colorIntensity(t){return Math.max(0,Math.min(255,Math.round(t*255)))}
function sampleVolume(v,x,y,z){return v.data[index3(v.shape,x,y,z)]||0}
function drawSlices(){
 if(!state.volume)return;
 const v=state.volume;
 const affine=v.affine;
 // RAS coordinate rows identify the actual physical scan plane.
 // Do not call NIfTI axis 2 "axial" unless it maps to superior/inferior.
 const rasAxis=Array.isArray(affine)&&affine.length>=3
  ? [0,1,2].map(r=>{
      const a=affine[r].slice(0,3).map(Math.abs);
      return a.indexOf(Math.max(...a));
    })
  : [0,1,2];
 if(new Set(rasAxis).size!==3)throw Error('Ambiguous NIfTI affine orientation');
 const sign=ras=>Array.isArray(affine)&&affine[ras][rasAxis[ras]]<0?-1:1;
 const planes=[
  {id:'axialSlice',fix:rasAxis[2],u:rasAxis[0],vr:rasAxis[1],uflip:sign(0)<0,vflip:sign(1)<0},
  {id:'coronalSlice',fix:rasAxis[1],u:rasAxis[0],vr:rasAxis[2],uflip:sign(0)<0,vflip:sign(2)<0},
  {id:'sagittalSlice',fix:rasAxis[0],u:rasAxis[1],vr:rasAxis[2],uflip:sign(1)<0,vflip:sign(2)<0}
 ];
 for(const plane of planes){
  const {ctx,w,h}=prepCanvas($(plane.id));
  const [width,height]=[v.shape[plane.u],v.shape[plane.vr]];
  const [cw,ch]=[Math.min(256,width),Math.min(256,height)];
  const img=ctx.createImageData(cw,ch);
  const coords=[0,0,0];
  coords[plane.fix]=Math.min(v.shape[plane.fix]-1,
      Math.max(0,Math.round((v.shape[plane.fix]-1)*Number($('slicePosition').value)/100)));
  for(let py=0;py<ch;py++)for(let px=0;px<cw;px++){
    const uu=Math.min(width-1,Math.floor(px*width/cw));
    const vv=Math.min(height-1,Math.floor((ch-1-py)*height/ch));
    coords[plane.u]=plane.uflip?width-1-uu:uu;
    coords[plane.vr]=plane.vflip?height-1-vv:vv;
    const raw=sampleVolume(v,...coords),q=colorIntensity(raw),pos=4*(py*cw+px);
    const isroi=state.mask&&state.mask.data[index3(state.mask.shape,...coords)]===Number($('roiLabel').value)&&Number($('roiLabel').value)>0;
    img.data[pos]=isroi?235:q;
    img.data[pos+1]=isroi?161:q;
    img.data[pos+2]=isroi?80:q;
    img.data[pos+3]=255;
  }
  const off=document.createElement('canvas');
  off.width=cw;off.height=ch;
  off.getContext('2d').putImageData(img,0,0);
  ctx.imageSmoothingEnabled=true;
  const factor=Math.min(w/cw,h/ch);
  ctx.drawImage(off,(w-cw*factor)/2,(h-ch*factor)/2,cw*factor,ch*factor);
 }
}
function recomputeSurface(){if(!state.volume)return;const v=state.volume,[nx,ny,nz]=v.shape,thr=Number($('surfaceThreshold').value)/100;let pts=[];
 const stride=Math.max(1,Math.ceil(Math.max(nx,ny,nz)/90));for(let z=1;z<nz-1;z+=stride)for(let y=1;y<ny-1;y+=stride)for(let x=1;x<nx-1;x+=stride){const val=sampleVolume(v,x,y,z);if(val<thr)continue;
 if(sampleVolume(v,x-1,y,z)>thr&&sampleVolume(v,x+1,y,z)>thr&&sampleVolume(v,x,y-1,z)>thr&&sampleVolume(v,x,y+1,z)>thr&&sampleVolume(v,x,y,z-1)>thr&&sampleVolume(v,x,y,z+1)>thr)continue;
 const mask=state.mask?state.mask.data[index3(v.shape,x,y,z)]===Number($('roiLabel').value)&&Number($('roiLabel').value)>0:false;
 pts.push({p:[x,y,z],v:val,roi:mask});}
 const cap=5400;if(pts.length>cap){let step=Math.ceil(pts.length/cap);pts=pts.filter((_,i)=>i%step===0)}state.surfacePoints=pts;$('brainCanvasTop').textContent=`VOXEL-DERIVED PREVIEW · ${pts.length} SAMPLES`;
}
function networkPositions(){let a=[];for(let i=0;i<512;i++){let[c,z,y,x]=nodeParts(i);a.push([(c-3.5)*1.3+(x-1.5)*.28,(y-1.5)*.9+Math.sin(c*2.4)*.18,(z-1.5)*.9+(x-1.5)*.4])}return a}
const networkPts=networkPositions();
function drawNetwork(){if(!state.trace)return;const {ctx,w,h}=prepCanvas($('networkCanvas')),f=frame(),b=baseline(),scale=Math.min(w,h)*.080*state.netScale;
const vals=f.a.map((v,i)=>v-b.a[i]);let maxdelta=Math.max(0.000001,...vals.map(Math.abs)),maxabs=Math.max(0.000001,...f.a.map(Math.abs));
let projected=networkPts.map(p=>project(p,state.netYaw,state.netPitch,scale,w,h));state.netProjected=projected;
let dots=projected.map((p,i)=>({p,i})).sort((a,b)=>a.p[2]-b.p[2]);
for(const {p,i} of dots){let value=$('viewMode').value==='delta'?vals[i]:f.a[i],den=$('viewMode').value==='delta'?maxdelta:maxabs,m=Math.min(1,Math.abs(value)/den),positive=value>=0;let col=$('viewMode').value==='absolute'||positive?'104,218,205':'242,159,104';ctx.beginPath();ctx.fillStyle=`rgba(${col},${.22+.75*m})`;ctx.arc(p[0],p[1],1.6+m*4,0,Math.PI*2);ctx.fill();if(i===state.node){ctx.beginPath();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.arc(p[0],p[1],8+m*4,0,Math.PI*2);ctx.stroke();}}
font(ctx,'#7a9aaf',10);ctx.fillText('Node count: 512 | Edges not inferred',16,h-36);
}
function drawCurve(){if(!state.trace)return;const {ctx,w,h}=prepCanvas($('lineChart')),margin={left:40,right:15,top:19,bottom:27};const values=state.trace.frames.map(f=>f.a[state.node]);const min=Math.min(...values),max=Math.max(...values),range=Math.max(0.02,max-min);const low=min-.15*range,high=max+.15*range;const xx=i=>margin.left+i*(w-margin.left-margin.right)/5,yy=v=>h-margin.bottom-(v-low)/(high-low)*(h-margin.top-margin.bottom);
ctx.strokeStyle='#274358';ctx.lineWidth=1;for(let j=0;j<4;j++){let y=margin.top+j*(h-margin.top-margin.bottom)/3;ctx.beginPath();ctx.moveTo(margin.left,y);ctx.lineTo(w-margin.right,y);ctx.stroke();font(ctx,'#698ca3',10);ctx.fillText((high-j*(high-low)/3).toFixed(3),3,y+3)};
ctx.beginPath();for(let i=0;i<6;i++){if(i===0)ctx.moveTo(xx(i),yy(values[i]));else ctx.lineTo(xx(i),yy(values[i]));}ctx.strokeStyle=style.cyan;ctx.lineWidth=2;ctx.stroke();
for(let i=0;i<6;i++){ctx.beginPath();ctx.arc(xx(i),yy(values[i]),i===state.frame?5:3,0,Math.PI*2);ctx.fillStyle=i===state.frame?'#f0bd74':style.cyan;ctx.fill();font(ctx,'#7499ad',10);ctx.fillText(state.trace.frames[i].s.toFixed(1),xx(i)-8,h-5);}
}
function drawEdgeContributions(){let el=$('edgeChart'),canvas=el.querySelector('canvas');if(!canvas){el.innerHTML='';canvas=document.createElement('canvas');el.appendChild(canvas)}const {ctx,w,h}=prepCanvas(canvas),f=frame();const contributions=f.inputs.map((v,i)=>v*state.trace.weights[i]);const sorted=contributions.map((v,i)=>({v,i})).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,16);const max=Math.max(...sorted.map(e=>Math.abs(e.v)),.00001);let half=w/2,barH=6.2,gap=2;sorted.forEach(({v,i},j)=>{const y=8+j*(barH+gap);let len=Math.abs(v)/max*(half-36);ctx.fillStyle=v>=0?style.cyan:style.orange;ctx.fillRect(v<0?half-len:half,y,len,barH)});ctx.strokeStyle='#5c7a8c';ctx.beginPath();ctx.moveTo(half,3);ctx.lineTo(half,h-3);ctx.stroke();font(ctx,'#829bb0',9);ctx.fillText('− weighted contribution',10,h-3);ctx.fillText('+ contribution',half+9,h-3);
let weighted=contributions.reduce((a,b)=>a+b,0)+state.trace.bias,actual=f.pre;const err=Math.abs(weighted-actual);$('equation').textContent=`Σ (stored rounded input × weight) + bias = ${weighted.toFixed(7)}\nRecorded pre-activation: ${actual.toFixed(7)} · |difference| = ${err.toExponential(2)}`;}
function render(){if(!state.trace)return;
 const d=state.trace,f=frame(),b=baseline(),unit=nodeParts(state.node),active=state.modelId==='brainae';
 $('networkCanvas').style.visibility=active?'visible':'hidden';$('modelEmpty').classList.toggle('hidden',active);$('modelMetric').textContent=active?'BrainAE':state.modelId.toUpperCase();
 $('modelStatus').textContent=active?'真实 PyTorch 历史记录 · 合成输入':'尚未训练或接入';$('frameBadge').textContent=active?`Run ${state.frame+1}/6`:'未连接';
 $('strength').disabled=!active;$('prevBtn').disabled=!active;$('nextBtn').disabled=!active;$('viewMode').disabled=!active;
 $('selectedNodeBadge').textContent=active?'#'+String(state.node).padStart(3,'0'):'—';
 $('strength').value=state.frame;$('strengthLabel').textContent=f.s.toFixed(1)+' voxels';$('activationVal').textContent=active?f.a[state.node].toFixed(6):'—';$('deltaVal').textContent=active?(f.a[state.node]-b.a[state.node]).toFixed(6):'—';
 $('unitLabel').textContent=active?`conv3[${unit.join(',')}]`:'模型未启用';$('mseVal').textContent=active?f.mse.toFixed(6):'—';$('ablationVal').textContent=active?(state.node===42?f.ablation.toExponential(3):'not measured'):'—';
 $('unitInfo').innerHTML=active?`<span><b>Layer</b> conv3</span><span><b>Channel</b> ${unit[0]}</span><span><b>Coords</b> [z${unit[1]}, y${unit[2]}, x${unit[3]}]</span><span><b>Source</b> synthetic</span>`:'当前模型没有有效运行记录。';
 if(active){$('edgesBadge').textContent=state.node===42?'216 RECORDED EDGES':'UNAVAILABLE';if(state.node===42){$('edgeMessage').textContent='固定节点 conv3[0,2,2,2]：真实卷积权重×输入贡献。';drawEdgeContributions();$('edgeChart').style.visibility='visible';$('equation').style.visibility='visible';}else{$('edgeMessage').textContent='当前模型记录没有该单元的输入连线；禁止编造。';$('edgeChart').innerHTML='<button class="outline" id="focusBtn">查看固定节点连接 →</button>';$('equation').style.visibility='hidden';$('focusBtn').onclick=()=>{state.node=42;render()}}drawNetwork();drawCurve();}
 else{$('edgeMessage').textContent='尚无已审核权重或激活；请首先训练并接入该模型。';$('edgeChart').innerHTML='';$('equation').textContent='';const a=prepCanvas($('lineChart'));}
 drawBrain();
}
function setupRotate(spaceId,which){const el=$(spaceId),canvas=el.querySelector('canvas');let down=false,drag=false,px=0,py=0;canvas.addEventListener('pointerdown',e=>{down=true;drag=false;px=e.clientX;py=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!down)return;let dx=e.clientX-px,dy=e.clientY-py;if(Math.abs(dx)+Math.abs(dy)>2)drag=true;state[which+'Yaw']+=dx*.008;state[which+'Pitch']=Math.max(-1.4,Math.min(1.4,state[which+'Pitch']+dy*.008));px=e.clientX;py=e.clientY;which==='brain'?drawBrain():drawNetwork()});canvas.addEventListener('pointerup',e=>{down=false;if(which==='net'&&!drag){const rect=canvas.getBoundingClientRect();let x=e.clientX-rect.left,y=e.clientY-rect.top;let closest=-1,d2=200;for(let i=0;i<state.netProjected.length;i++){const p=state.netProjected[i];let dist=(p[0]-x)**2+(p[1]-y)**2;if(dist<d2){closest=i;d2=dist}}if(closest>=0){state.node=closest;render();}}});canvas.addEventListener('wheel',e=>{e.preventDefault();state[which+'Scale']=Math.max(.55,Math.min(2,state[which+'Scale']*(e.deltaY<0?1.075:.925)));which==='brain'?drawBrain():drawNetwork()},{passive:false});}
const TEMPLATE_URL='https://templateflow.s3.amazonaws.com/tpl-MNI152Lin/tpl-MNI152Lin_res-02_T1w.nii.gz';
function status(msg,isError=false){$('anatomyReceipt').textContent=msg;$('anatomyReceipt').style.borderColor=isError?'#ae6670':'#29515c'}
function isPreview(v){return v&&v.schema==='brainmirror.anatomy-preview.v1'&&Array.isArray(v.shape)&&v.shape.length===3&&typeof v.intensities_b64==='string'}
function b64bytes(base64){const raw=atob(base64);let result=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)result[i]=raw.charCodeAt(i);return result}
function decodePreview(obj){if(!isPreview(obj))throw Error('不是 BrainMirror anatomy-preview.v1 格式');const shape=obj.shape.map(Number),n=shape.reduce((a,b)=>a*b,1);if(n<512||n>8e6)throw Error('预览体积大小非法');let bytes=b64bytes(obj.intensities_b64);if(bytes.length!==n)throw Error('预览数值长度错误');const data=Float32Array.from(bytes,v=>v/255);return {shape,data,source:String(obj.provenance?.source||'External preview'),kind:String(obj.provenance?.kind||'external'),rawDims:shape,meta:obj.provenance||{},affine:obj.affine||null};}
async function readBytes(buffer,filename){let bytes=new Uint8Array(buffer);if(filename.toLowerCase().endsWith('.gz')){if(!('DecompressionStream' in window))throw Error('当前浏览器不支持 gzip 解压。请使用 Chrome 新版或先解压成 .nii。');const stream=new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));buffer=await new Response(stream).arrayBuffer();if(buffer.byteLength>240e6)throw Error('体积解压超过内存安全上限 240 MB');bytes=new Uint8Array(buffer);}return buffer}
function parseNifti(buffer,name,asLabels=false){if(buffer.byteLength<352)throw Error('NIfTI 文件太短');const dv=new DataView(buffer);let le=dv.getInt32(0,true)===348;if(!le && dv.getInt32(0,false)!==348)throw Error('不是 NIfTI-1 单文件');const endian=le;const ndim=dv.getInt16(40,endian);if(ndim!==3)throw Error(`当前仅支持 3D NIfTI：ndim=${ndim}`);const dims=[1,2,3].map(i=>dv.getInt16(40+2*i,endian));if(dims.some(x=>x<8||x>1024))throw Error('NIfTI shape 异常');const datatype=dv.getInt16(70,endian),bitpix=dv.getInt16(72,endian);const types={2:[1,(o)=>dv.getUint8(o)],4:[2,(o)=>dv.getInt16(o,endian)],8:[4,(o)=>dv.getInt32(o,endian)],16:[4,(o)=>dv.getFloat32(o,endian)],64:[8,(o)=>dv.getFloat64(o,endian)],256:[1,(o)=>dv.getInt8(o)],512:[2,(o)=>dv.getUint16(o,endian)],768:[4,(o)=>dv.getUint32(o,endian)]};const type=types[datatype];if(!type||bitpix!==type[0]*8)throw Error('当前不支持此 NIfTI datatype='+datatype);const offset=Math.max(352,Math.ceil(dv.getFloat32(108,endian)));const n=dims.reduce((a,b)=>a*b,1);if(offset+n*type[0]>buffer.byteLength)throw Error('NIfTI voxel data 截断');const scale0=dv.getFloat32(112,endian),scale=Number.isFinite(scale0)&&scale0!==0?scale0:1,inter=dv.getFloat32(116,endian)||0;
// Downsample only for browser display, original bytes are not changed.
const maxAxis=Math.max(...dims),ratio=Math.max(1,maxAxis/96),shape=dims.map(x=>Math.max(8,Math.ceil(x/ratio)));const data=new Float32Array(shape.reduce((a,b)=>a*b,1));const vals=[];for(let z=0;z<shape[2];z++)for(let y=0;y<shape[1];y++)for(let x=0;x<shape[0];x++){let ox=Math.min(dims[0]-1,Math.floor(x*dims[0]/shape[0])),oy=Math.min(dims[1]-1,Math.floor(y*dims[1]/shape[1])),oz=Math.min(dims[2]-1,Math.floor(z*dims[2]/shape[2]));let v=type[1](offset+(ox+dims[0]*(oy+dims[1]*oz))*type[0])*scale+inter;if(!Number.isFinite(v))v=0;data[index3(shape,x,y,z)]=v;if(v>0 && Number.isFinite(v))vals.push(v)}
if(!asLabels && vals.length<128)throw Error('可用非零体素过少');if(!asLabels){vals.sort((a,b)=>a-b);}const low=vals[Math.floor(vals.length*.02)],high=vals[Math.floor(vals.length*.98)];if(!asLabels){if(!(high>low))throw Error('体素强度动态范围异常');for(let i=0;i<data.length;i++)data[i]=Math.max(0,Math.min(1,(data[i]-low)/(high-low)));}else{for(let i=0;i<data.length;i++)if(data[i]<0||!Number.isInteger(data[i]))throw Error('解剖标签应为非负整数');}
const sform=dv.getInt16(254,endian),qform=dv.getInt16(252,endian);const affine=sform?Array.from({length:3},(_,i)=>Array.from({length:4},(_,j)=>dv.getFloat32(280+i*16+j*4,endian))):null;
return {shape,data,rawDims:dims,datatype,qform,sform,affine,source:name,kind:'local_nifti',meta:{normalization:'visual p2-p98, within this scan',nonzero_count:vals.length,downsampling:'nearest voxel',source_file:name},display:'real T1 voxel preview'};}
async function decodeFile(file){if(file.name.endsWith('.json'))return decodePreview(JSON.parse(await file.text()));const buf=await readBytes(await file.arrayBuffer(),file.name);return parseNifti(buf,file.name)}
function updateAnatomyMeta(){const v=state.volume;if(!v){$('anatomyStatus').textContent='等待载入';return;}$('brainEmpty').classList.add('hidden');$('anatomyStatus').textContent=v.kind==='reference'?'MNI152 (参考)':v.kind.includes('group')?'IXI 467 平均脑':'本地 MRI';$('anatomyStats').textContent=v.shape.join('×')+' · 预览采样';$('anatomySourceTag').textContent=v.kind==='reference'?'MNI152 PUBLIC TEMPLATE':v.kind.includes('group')?'IXI 467 · UNREVIEWED':'LOCAL VOLUME / PRIVATE';$('anatomyInfo').textContent=`来源：${v.source}。真实体素抽样 / 几何与强度供浏览用途；表面点云为阈值近似，不是医学脑分割。此输入没有传给右侧 BrainAE。`;
$('provAnatomy').textContent=`${v.source}; ${v.rawDims.join('×')} 原始尺寸；预览尺寸 ${v.shape.join('×')}；浏览器端加载，未上传。`;status(`已载入 ${v.source}
Preview shape: ${v.shape.join('×')}
Mode: ${state.mode}
NO ONLINE MODEL INFERENCE · NOT COUPLED`);recomputeSurface();
 if(window.BrainMirror3D){
  try{
    const result=window.BrainMirror3D.loadVolume(v);
    $('anatomySourceTag').textContent=v.kind.includes('group')?'IXI 467 · CONTINUOUS 3D · UNREVIEWED':'MRI-DERIVED 3D SURFACE';
    $('anatomyInfo').textContent+=' 已从体素支持区域构建 '+result.polygons+' 个连续网格三角形。此表面不等于经 FreeSurfer 认证的皮层重建。';
  }catch(e){status('3D 表面生成失败，已保留 MRI 原始切片：'+e.message,true);}
 }
 render()}
async function importVolume(file){if(!file)return;state.anatomyBusy=true;status(`本地读取 ${file.name}，可能需要数秒…`);try{const v=await decodeFile(file);state.volume=v;state.mask=null;state.mode=(v.kind==='reference'||v.kind.includes('group'))?'template':'subject';syncMode();updateAnatomyMeta();}catch(e){status('导入失败：'+e.message,true)}finally{state.anatomyBusy=false}}
async function importMask(file){if(!file)return;try{if(!state.volume)throw Error('请先加载 MRI');let v,labels;
 if(file.name.endsWith('.json')){const obj=JSON.parse(await file.text());v=decodePreview(obj);if(!obj.labels_b64)throw Error('缺少 labels_b64');const raw=b64bytes(obj.labels_b64),n=v.shape.reduce((a,b)=>a*b,1);
 if(obj.label_dtype==='uint16'){if(raw.length!==n*2)throw Error('标签长度不匹配');const view=new DataView(raw.buffer);labels=new Uint16Array(n);for(let i=0;i<n;i++)labels[i]=view.getUint16(i*2,true)}else{if(raw.length!==n)throw Error('标签长度不匹配');labels=raw;}}
 else{const buf=await readBytes(await file.arrayBuffer(),file.name);v=parseNifti(buf,file.name,true);labels=v.data;}
 if(v.shape.join(',')!==state.volume.shape.join(','))throw Error('NIfTI 预览采样网格与 MRI 不一致');if(v.rawDims.join(',')!==state.volume.rawDims.join(','))throw Error('原始矩阵尺寸不一致');
 if(v.affine&&state.volume.affine&&Math.max(...v.affine.flat().map((x,i)=>Math.abs(x-state.volume.affine.flat()[i])))>.01)throw Error('affine 不匹配，禁止任意叠加');
 state.mask={shape:v.shape,data:labels};$('roiStatus').textContent='标签已导入，可输入 label ID 定位；仅做可视化。';recomputeSurface();drawBrain();status('解剖标签载入：'+file.name+'；形状与空间已核验。此标签不会自动指定 AI 概念节点。');
 }catch(e){status('脑区导入失败：'+e.message,true)}}
async function loadIXIPopulation(){
 if(state.anatomyBusy)return;
 state.anatomyBusy=true;
 const button=$('loadIXIBtn');
 if(button)button.disabled=true;
 status('正在载入 IXI 467 人群体平均脑（未完成独立解剖审核）…');
 try{
  const response=await fetch(IXI_PUBLIC_URL,{mode:'cors',cache:'no-store'});
  if(!response.ok)throw Error('公开参考数据尚未就绪 (HTTP '+response.status+')');
  const raw=await response.text();
  const hashbuf=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(raw));
  const hash=Array.from(new Uint8Array(hashbuf),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==IXI_PUBLIC_SHA256)throw Error('IXI 预览 SHA256 不匹配，拒绝加载');
  const json=JSON.parse(raw);
  if(json?.provenance?.kind!=='ixi_affine_group_candidate'||json.provenance.subjects_included!==467||
     json.provenance.status!=='RESEARCH_CANDIDATE_UNREVIEWED'||json.provenance.license!=='CC BY-SA 3.0')throw Error('群体数据来源或验收状态不符合发布记录');
  const v=decodePreview(json);
  if(v.data.some(x=>!Number.isFinite(x)))throw Error('非有限体素');
  state.volume=v;state.mask=null;state.mode='template';
  syncMode();updateAnatomyMeta();
  status('已载入真实 IXI 467 人群体平均脑（研究候选版，未获独立解剖审核）');
 }catch(e){
  status('暂无法自动载入 IXI：'+e.message+'。可在下方导入本地 JSON / NIfTI，或重试。',true);
 }finally{
  state.anatomyBusy=false;
  if(button)button.disabled=false;
 }
}
async function loadReference(){if(state.anatomyBusy)return;state.anatomyBusy=true;status('正在从 TemplateFlow 公共镜像请求 MNI152 2mm 参考模板…');$('loadTemplateBtn').disabled=true;
try{const response=await fetch(TEMPLATE_URL,{mode:'cors'});if(!response.ok)throw Error(`HTTP ${response.status}`);const buf=await response.arrayBuffer();const unpacked=await readBytes(buf,'reference.nii.gz');const v=parseNifti(unpacked,'TemplateFlow MNI152Lin res-02 T1w');v.kind='reference';v.meta.source_url=TEMPLATE_URL;v.meta.ixi_average=false;state.volume=v;state.mode='template';syncMode();updateAnatomyMeta();}catch(e){status('公共模板自动加载失败：'+e.message+'。可以从 TemplateFlow 下载合法模板或在左侧导入已审核 NIfTI；IXI 群体模板尚未生成。',true)}finally{state.anatomyBusy=false;$('loadTemplateBtn').disabled=false}}
function syncMode(){document.querySelectorAll('[data-brain-mode]').forEach(b=>b.classList.toggle('active',b.dataset.brainMode===state.mode));$('modeBadge').textContent=state.mode==='template'?'群体参考':state.mode==='subject'?'真实个体':'结构实验';}
function events(){setupRotate('brainSpace','brain');setupRotate('networkSpace','net');$('strength').addEventListener('input',e=>{state.frame=Number(e.target.value);render()});$('viewMode').addEventListener('change',drawNetwork);$('prevBtn').onclick=()=>{state.frame=Math.max(0,state.frame-1);render()};$('nextBtn').onclick=()=>{state.frame=Math.min(5,state.frame+1);render()};
 $('modelSelect').onchange=e=>{state.modelId=e.target.value;render()};$('surfaceThreshold').oninput=e=>{$('thresholdDisplay').textContent=e.target.value+'%';recomputeSurface();drawBrain()};$('slicePosition').oninput=e=>{$('sliceLocation').textContent=e.target.value+'%';drawSlices()};$('roiLabel').oninput=()=>{recomputeSurface();drawBrain()};
 for(const button of document.querySelectorAll('[data-brain-mode]'))button.onclick=()=>{state.mode=button.dataset.brainMode;syncMode();if(state.mode==='experiment'&&!state.mask)status('结构实验模式已打开，但缺少经过空间核对的脑区标签；变形预览保持禁用。');};
 $('loadIXIBtn').onclick=loadIXIPopulation; $('loadTemplateBtn').onclick=loadReference;$('loadTemplateBtn2').onclick=loadReference;$('volumeFile').onchange=e=>importVolume(e.target.files?.[0]);$('maskFile').onchange=e=>importMask(e.target.files?.[0]);$('resetBrainBtn').onclick=()=>{state.brainYaw=.5;state.brainPitch=-.28;state.brainScale=1;drawBrain()};
 $('exportBtn').onclick=()=>{if(!state.trace){status('无模型记录可导出',true);return}let data={schema:'brainmirror.web.v0.4.export',provenance:state.trace.provenance,anatomy_provenance:state.volume?{source:state.volume.source,kind:state.volume.kind,raw_dims:state.volume.rawDims,preview_shape:state.volume.shape,linked_to_model:false}:null,model_id:state.modelId,model_is_coupled_to_current_mri:false,trace_sha_prefix:state.trace.sha,frame:frame(),baseline:baseline(),selected_unit:nodeParts(state.node),selected_activation:frame().a[state.node],connection_weights:state.node===42?state.trace.weights:null,source_note:'Actual prerecorded PyTorch synthetic traces, NOT current MRI inference'};const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='brainmirror-experiment-record.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)};
 document.querySelectorAll('[data-scroll]').forEach(b=>b.onclick=()=>document.getElementById(b.dataset.scroll)?.scrollIntoView({behavior:'smooth'}));window.addEventListener('resize',render);
}
async function init(){events();const response=await fetch('./trace.json',{cache:'no-store'});if(!response.ok)throw Error('Trace HTTP '+response.status);let d=await response.json();if(d.dataset!=='SYNTHETIC PHANTOM; NO REAL HUMAN MRI'||d.frames.length!==6||d.frames.some(f=>f.a.length!==512||f.inputs.length!==216)||d.weights.length!==216)throw Error('Trace source invalid');state.trace=d;$('provenanceModel').textContent=`${d.model}：${d.provenance.training_examples} 个合成样本，${d.provenance.epochs} epochs，seed ${d.provenance.seed}；无真实脑 MRI 权重。`;$('traceFingerprint').textContent='PyTorch trace prefix: '+d.sha;render();
 try{
  const viewer=await import('./brain3d.js');
  viewer.mount({container:$('brainSpace')});
  window.BrainMirror3D=viewer;
 }catch(e){status('连续 3D 渲染器启动失败：'+e.message+'；请检查网络是否允许加载 Three.js。',true)}
 await loadIXIPopulation();
}
init().catch(e=>{status('模型记录无法加载：'+e.message,true);console.error(e)});