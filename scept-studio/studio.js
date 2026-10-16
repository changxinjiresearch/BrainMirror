'use strict';
/* SCEPT Studio v1.1. Every plot comes from on-device ONNX Runtime forward passes.
 * Supported observation surfaces are exported ONNX graph outputs, not arbitrary hidden units.
 * No training, fabricated activations, or background upload.
 */
const el=id=>document.getElementById(id);
const $all=q=>Array.from(document.querySelectorAll(q));
const ORT_VERSION='1.22.0';
const ORT_BASE='https://cdn.jsdelivr.net/npm/onnxruntime-web@'+ORT_VERSION+'/dist/';
const presets={
 mnist:{name:'MNIST-8 · ONNX Model Zoo',url:'https://huggingface.co/onnxmodelzoo/mnist-8/resolve/main/mnist-8.onnx',dims:[1,1,28,28],normalize:'unit',source:'draw',sha:'2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf',license:'Apache-2.0'},
 vit:{name:'ViT-base · pretrained ImageNet',url:'https://huggingface.co/Xenova/vit-base-patch16-224/resolve/main/onnx/model_quantized.onnx',dims:[1,3,224,224],normalize:'signed',source:'file',sha:'fe13a717a54c0f6ef1a966e0dd67a82e532e0898018fc50f7c37d437c225074a',license:'Apache-2.0'},
 squeeze:{name:'SqueezeNet 1.1 · ONNX Model Zoo',url:'https://huggingface.co/onnxmodelzoo/squeezenet1.1-7/resolve/main/squeezenet1.1-7.onnx',dims:[1,3,224,224],normalize:'imagenet',source:'file',sha:'1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088',license:'Apache-2.0'}
};
let ort=null,session=null,modelHash='',modelName='',modelSource='',manifest=null,modelUrl='',actualProvider='wasm';
let sourceMode='draw',sourceVolume=null,sourceTensor=null,baseline=null,lastRecord=null;
let history=[],drawActive=false,center={x:.5,y:.5},runBusy=false;
const mainCanvas=el('inputCanvas'),c2d=mainCanvas.getContext('2d',{willReadFrequently:true}),viz=el('vizCanvas'),vctx=viz.getContext('2d');
const encoder=new TextEncoder();
function notice(msg,failed){el('runMessage').textContent=msg;el('runMessage').style.color=failed?'#ff9e9e':'#9ce9d8';if(failed) console.error('[SCEPT]',msg)}
function setState(msg,ready){el('stateLabel').textContent=msg;el('statusLed').className=ready?'ready':'';el('runExperiment').disabled=!ready||runBusy;el('selfTest').disabled=!ready||runBusy}
function compact(v){if(typeof v!=='number'||!Number.isFinite(v))return '—';if(Math.abs(v)>=1000)return v.toExponential(2);return v.toFixed(Math.abs(v)<1?.5:3)}
function typedBytes(a){return new Uint8Array(a.buffer,a.byteOffset,a.byteLength)}
async function hashBuffer(bytes){const digest=await crypto.subtle.digest('SHA-256',bytes);return Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('')}
async function hashFloat(a){return hashBuffer(typedBytes(a))}
function numberDims(){const dims=el('shape').value.split(/[,x×\s]+/).map(v=>Number(v));if(!dims.length||dims.some(v=>!Number.isInteger(v)||v<1)||dims.reduce((a,b)=>a*b,1)>35000000)throw Error('输入形状无效或超出 3500 万元素安全限额');return dims}
function product(d){return d.reduce((a,b)=>a*b,1)}
function sessionMetadata(name){return session&&session.inputMetadata&&session.inputMetadata[name]||null}
function baseInputName(){return el('inputName').value.trim()||session?.inputNames.find(n=>!n.includes('__scept_mask'))}
function reportedShapes(){const first=baseInputName(),meta=sessionMetadata(first);const dimensions=meta?.dims||meta?.dimensions;if(Array.isArray(dimensions)&&dimensions.every(v=>Number.isInteger(v)&&v>0)){el('shape').value=dimensions.join(',');return dimensions}return null}
function showModel(){el('modelName').textContent=modelName||'未选择模型';el('modelSha').textContent=modelHash?modelHash.slice(0,17)+'…':'—';el('modelSha').title=modelHash;el('modelIO').textContent=session?session.inputNames.length+' input / '+session.outputNames.length+' output':'—';el('runExperiment').disabled=!session}
async function ensureOrt(){if(ort)return ort;notice('正在加载 ONNX Runtime Web (CPU/WASM)…');ort=await import(ORT_BASE+'esm/ort.min.js');ort.env.wasm.wasmPaths=ORT_BASE;ort.env.wasm.numThreads=1;return ort}
function metadataType(name){return sessionMetadata(name)?.type||'float32'}
function bytesToTensorData(values,type){switch(type){case 'float64':return Float64Array.from(values);case 'int64':return BigInt64Array.from(values,v=>BigInt(Math.round(v)));case 'int32':return Int32Array.from(values);case 'uint8':return Uint8Array.from(values);case 'bool':return Uint8Array.from(values,v=>v?1:0);default:return Float32Array.from(values)}}
async function setupModel(bytes,description,provenance,preset){
 if(runBusy)throw Error('实验运行中不能替换模型');
 setState('正在解析权重…',false);const lib=await ensureOrt();
 const data=bytes instanceof Uint8Array?bytes:new Uint8Array(bytes);
 const hash=await hashBuffer(data);
 if(preset&&preset.sha!==hash)throw Error('下载的预训练模型 SHA-256 与官方文件不一致，拒绝加载');
 const webgpu=el('webgpu').checked;
 let newSession,provider='wasm';
 if(webgpu){try{
  const gpuOrt=await import(ORT_BASE+'esm/ort.webgpu.min.js');
  gpuOrt.env.wasm.wasmPaths=ORT_BASE;
  newSession=await gpuOrt.InferenceSession.create(data,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
  ort=gpuOrt;provider='webgpu / wasm fallback';
 }catch(e){console.warn('WebGPU unavailable, falling back to WASM',e)}}
 if(!newSession)newSession=await lib.InferenceSession.create(data,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
 if(session?.release)await session.release();
 session=newSession;actualProvider=provider;modelHash=hash;modelName=description;modelSource=provenance;modelUrl=preset?.url||'local-file';el('backend').textContent=provider.toUpperCase();
 const first=session.inputNames.find(n=>!n.includes('__scept_mask'))||session.inputNames[0];
 el('inputName').value=first||'';
 if(preset){el('shape').value=preset.dims.join(',');el('normalize').value=preset.normalize;setMode(preset.source)}
 else if(manifest?.input?.shape)el('shape').value=manifest.input.shape.join(',');
 else reportedShapes();
 populateMaskInputs();showModel();setState('已加载预训练模型',true);
 notice('模型已在本地加载：'+description+' · '+(data.length/1048576).toFixed(2)+' MiB · SHA '+hash.slice(0,12));
}
async function loadPreset(which){
 const p=presets[which];notice('下载并校验官方模型：'+p.name+' …');
 try{const response=await fetch(p.url,{cache:'default'});if(!response.ok)throw Error('下载失败 HTTP '+response.status);
  const buf=new Uint8Array(await response.arrayBuffer());await setupModel(buf,p.name,'ONNX Model Zoo · '+p.url,p);
  if(which==='mnist')renderDigit();
 }catch(e){setState('模型加载失败',false);notice('模型加载失败：'+e.message+'。如遇跨域或网络限制，可下载 ONNX 文件后手动导入。',true)}
}
function populateMaskInputs(){
 const sel=el('maskInput');sel.innerHTML='<option value="">仅输入扰动</option>';
 for(const name of session?.inputNames||[])if(name.includes('__scept_mask')){
  const opt=document.createElement('option');opt.value=name;opt.textContent=name+' · 内部干预';sel.append(opt)
 }
}
function setMode(mode){
 sourceMode=mode;
 $all('[data-source]').forEach(b=>b.classList.toggle('active',b.dataset.source===mode));
 el('mriTools').hidden=mode!=='mri';
 el('sourceNote').textContent={
  draw:'手绘数字适合 MNIST；图像点击位置设为干预中心。',file:'上传 jpg/png/webp 等自然图像；请核对模型要求的预处理和通道顺序。',
  mri:'支持 NIfTI-1 .nii/.nii.gz 基本数据类型；三维张量按 Z/Y/X 读取，需验证模型预处理与空间方向。',
  tensor:'输入 JSON 格式：{"dims":[1,1,28,28],"data":[...]}；必须与已加载模型要求一致。'
 }[mode];
}
for(const b of $all('[data-source]'))b.addEventListener('click',()=>setMode(b.dataset.source));
el('demoMnist').addEventListener('click',()=>loadPreset('mnist'));
el('demoSqueeze').addEventListener('click',()=>loadPreset('squeeze'));
el('demoVit').addEventListener('click',()=>{notice('ViT-base 模型约 88 MB，可能需要较长加载时间及较大浏览器内存');loadPreset('vit')});
el('webgpu').addEventListener('change',()=>{if(session)notice('后端选项已修改；请重新导入模型使更改生效')});
el('modelFile').addEventListener('change',async e=>{const f=e.target.files?.[0];if(!f)return;try{await setupModel(new Uint8Array(await f.arrayBuffer()),f.name,'local://'+f.name,null)}catch(err){setState('模型载入失败',false);notice(err.message,true)}});
el('manifestFile').addEventListener('change',async e=>{try{const f=e.target.files?.[0];if(!f)return;const m=JSON.parse(await f.text());if(m.schema!=='scept.model.v1'||!m.input||!Array.isArray(m.input.shape))throw Error('无效 manifest：需要 schema=scept.model.v1 及 input.shape');
 manifest=m;el('shape').value=m.input.shape.join(',');if(m.input.name)el('inputName').value=m.input.name;if(m.input.normalize)el('normalize').value=m.input.normalize;notice('模型配置已导入：'+f.name);
 }catch(err){notice(err.message,true)}});
function erase(){c2d.fillStyle='#05080e';c2d.fillRect(0,0,224,224);sourceTensor=null;sourceVolume=null}
erase();
function paint(ev){const r=mainCanvas.getBoundingClientRect();const x=(ev.clientX-r.left)/r.width*224,y=(ev.clientY-r.top)/r.height*224;
 c2d.lineWidth=22;c2d.lineCap='round';c2d.lineJoin='round';c2d.strokeStyle='#fff';if(!paint.previous)paint.previous=[x,y];c2d.beginPath();c2d.moveTo(...paint.previous);c2d.lineTo(x,y);c2d.stroke();paint.previous=[x,y];center={x:x/224,y:y/224};
}
mainCanvas.addEventListener('pointerdown',e=>{drawActive=true;mainCanvas.setPointerCapture(e.pointerId);if(sourceMode!=='draw')setMode('draw');paint.previous=null;paint(e)});
mainCanvas.addEventListener('pointermove',e=>{if(drawActive)paint(e)});
['pointerup','pointercancel','lostpointercapture'].forEach(n=>mainCanvas.addEventListener(n,()=>{drawActive=false;paint.previous=null}));
el('clearCanvas').addEventListener('click',()=>{erase();setMode('draw')});
function renderDigit(){erase();setMode('draw');c2d.font='bold 195px Arial';c2d.textAlign='center';c2d.textBaseline='middle';c2d.fillStyle='#fff';c2d.fillText('8',112,120);center={x:.48,y:.48}}
el('sampleDigit').addEventListener('click',renderDigit);
async function openImage(file){const bitmap=await createImageBitmap(file);erase();setMode('file');c2d.fillStyle='#05080e';c2d.fillRect(0,0,224,224);
 const s=Math.max(224/bitmap.width,224/bitmap.height),w=bitmap.width*s,h=bitmap.height*s;c2d.drawImage(bitmap,(224-w)/2,(224-h)/2,w,h);bitmap.close?.();center={x:.5,y:.5};notice('已加载本地图像，未上传服务器：'+file.name)}
async function niftiVolume(file){
 let buf=await file.arrayBuffer();let bytes=new Uint8Array(buf);if(bytes[0]===31&&bytes[1]===139){if(typeof DecompressionStream==='undefined')throw Error('浏览器不支持 gzip 解压');buf=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()}
 const v=new DataView(buf);if(buf.byteLength<352)throw Error('NIfTI 文件太短');
 const little=v.getInt32(0,true)===348;if(!little&&v.getInt32(0,false)!==348)throw Error('不是有效的 NIfTI-1 header');
 const ndim=v.getInt16(40,little),dims=Array.from({length:Math.min(ndim,4)},(_,i)=>v.getInt16(42+i*2,little));
 if(ndim<3||dims.some(d=>d<1))throw Error('只支持 NIfTI-1 三维或四维数据');
 const [w,h,d]=dims;if(w*h*d>35000000)throw Error('MRI 大于 3500 万体素，当前浏览器工作台拒绝加载');
 const type=v.getInt16(70,little),offset=Math.floor(v.getFloat32(108,little)),step={2:1,4:2,8:4,16:4,64:8,512:2}[type];if(!step||offset+w*h*d*step>buf.byteLength)throw Error('不支持的 NIfTI 数据类型或损坏的数据区');
 const arr=new Float32Array(w*h*d),dv=new DataView(buf,offset);for(let i=0;i<arr.length;i++){const at=i*step;arr[i]=type===2?dv.getUint8(at):type===4?dv.getInt16(at,little):type===8?dv.getInt32(at,little):type===16?dv.getFloat32(at,little):type===64?dv.getFloat64(at,little):dv.getUint16(at,little)}
 if(!arr.every(Number.isFinite))throw Error('MRI 包含非有限像素');
 sourceVolume={data:arr,w,h,d,filename:file.name,spacing:[v.getFloat32(80,little),v.getFloat32(84,little),v.getFloat32(88,little)],affineWarning:'NIfTI 原始存储轴向预览；本模块尚不支持临床级 affine 重采样'};
 sourceTensor=null;setMode('mri');el('slice').max=d-1;el('slice').value=Math.floor(d/2);renderSlice();notice('已本地加载 NIfTI '+w+'×'+h+'×'+d+'。空间方向/预处理需与模型要求核对。')
}
function renderSlice(){const v=sourceVolume;if(!v)return;const z=+el('slice').value,cv=document.createElement('canvas');cv.width=v.w;cv.height=v.h;const cx=cv.getContext('2d'),img=cx.createImageData(v.w,v.h);
 let lo=Infinity,hi=-Infinity;for(let y=0;y<v.h;y++)for(let x=0;x<v.w;x++){const p=v.data[z*v.w*v.h+y*v.w+x];if(p<lo)lo=p;if(p>hi)hi=p}
 const range=hi-lo||1;for(let i=0;i<v.w*v.h;i++){let q=Math.max(0,Math.min(255,255*(v.data[z*v.w*v.h+i]-lo)/range));img.data[i*4]=q;img.data[i*4+1]=q;img.data[i*4+2]=q;img.data[i*4+3]=255}
 cx.putImageData(img,0,0);c2d.fillStyle='#000';c2d.fillRect(0,0,224,224);c2d.drawImage(cv,0,0,224,224);el('sliceLabel').textContent='Z='+z+'/'+(v.d-1)+' · 仅原始存储方向预览';
}
el('slice').addEventListener('input',renderSlice);
el('sourceFile').addEventListener('change',async e=>{const f=e.target.files?.[0];if(!f)return;try{
 if(/\.nii(\.gz)?$/i.test(f.name))await niftiVolume(f);
 else if(f.name.toLowerCase().endsWith('.json')){const j=JSON.parse(await f.text());if(!Array.isArray(j.dims)||!Array.isArray(j.data)||product(j.dims)!==j.data.length)throw Error('Tensor JSON 中 dims 和 data 长度不匹配');
  if(j.data.some(v=>!Number.isFinite(Number(v))))throw Error('Tensor JSON 存在非有限数据');sourceTensor={dims:j.dims,data:Float32Array.from(j.data),filename:f.name};sourceVolume=null;setMode('tensor');notice('已加载 Tensor JSON：'+f.name)}
 else await openImage(f);
 }catch(err){notice('输入文件读取失败：'+err.message,true)}});
['strength','radius'].forEach(id=>el(id).addEventListener('input',()=>{el('strengthText').textContent=(+el('strength').value).toFixed(2);el('radiusText').textContent=Math.round(+el('radius').value*100)+'%'}));
function imageInput(dims){
 const H=dims[dims.length-2],W=dims[dims.length-1],C=dims.length===4?dims[1]:1;
 if(dims.length!==4||dims[0]!==1||![1,3,4].includes(C))throw Error('图像模式当前只支持 [1,1/3/4,H,W]；其他模型请导入 Tensor JSON 或 NIfTI');
 const t=document.createElement('canvas');t.width=W;t.height=H;const x=t.getContext('2d',{willReadFrequently:true});x.drawImage(mainCanvas,0,0,W,H);const im=x.getImageData(0,0,W,H).data;
 const output=new Float32Array(product(dims)),norm=el('normalize').value;
 for(let c=0;c<C;c++)for(let i=0;i<H*W;i++){
  const j=4*i,raw=C===1?(im[j]*.299+im[j+1]*.587+im[j+2]*.114)/255:im[j+c]/255;
  const val=norm==='imagenet'?(raw-[.485,.456,.406][Math.min(c,2)])/([.229,.224,.225][Math.min(c,2)]):norm==='signed'?2*raw-1:raw;
  output[c*H*W+i]=val;
 }
 return output;
}
function volumeInput(dims){const v=sourceVolume;if(!v)throw Error('先导入 NIfTI MRI');if(dims[0]!==1||dims[1]!==1||![4,5].includes(dims.length))throw Error('MRI 当前仅支持单通道 [1,1,D,H,W] 或 [1,1,H,W]；更复杂预处理请导入 Tensor JSON');
 const is3d=dims.length===5,D=is3d?dims[2]:1,H=dims[dims.length-2],W=dims[dims.length-1],a=new Float32Array(product(dims));
 let total=0,total2=0;const Z0=+el('slice').value;
 for(let z=0;z<D;z++)for(let y=0;y<H;y++)for(let x=0;x<W;x++){
  const zz=is3d?Math.min(v.d-1,Math.floor((z+.5)*v.d/D)):Z0,yy=Math.min(v.h-1,Math.floor((y+.5)*v.h/H)),xx=Math.min(v.w-1,Math.floor((x+.5)*v.w/W)),val=v.data[zz*v.w*v.h+yy*v.w+xx],j=z*H*W+y*W+x;
  a[j]=val;total+=val;total2+=val*val;
 }
 if(el('normalize').value==='zscore'){const mean=total/a.length,std=Math.sqrt(Math.max(total2/a.length-mean*mean,1e-12));for(let i=0;i<a.length;i++)a[i]=(a[i]-mean)/std}
 return a;
}
function getInput(dims){
 if(sourceMode==='tensor'){if(!sourceTensor)throw Error('请先上传 Tensor JSON');if(dims.join(',')!==sourceTensor.dims.join(','))throw Error('Tensor JSON dims 与输入设置不匹配');return Float32Array.from(sourceTensor.data)}
 if(sourceMode==='mri')return volumeInput(dims);
 return imageInput(dims);
}
function modifyInput(base,dims){
 const mode=el('perturb').value,r=+el('radius').value,s=+el('strength').value;
 const copy=Float32Array.from(base);if(mode==='none'||s===0)return copy;
 const H=dims[dims.length-2],W=dims[dims.length-1],area=H*W,C=copy.length/area,cx=center.x*W,cy=center.y*H;
 for(let c=0;c<C;c++)for(let y=0;y<H;y++)for(let x=0;x<W;x++){
  const dist=Math.hypot((x-cx)/W,(y-cy)/H);if(dist>r)continue;
  const k=c*area+y*W+x;
  if(mode==='occlusion')copy[k]*=(1-s);
  else if(mode==='intensity')copy[k]+=s;
  else if(mode==='noise'){const u=Math.sin((k+1)*12.9898)*43758.5453;copy[k]+=(u-Math.floor(u)-.5)*s}
 }
 return copy;
}
function additionalFeeds(lib,maskName,ablate){
 const result={};
 for(const n of session.inputNames){if(n===baseInputName())continue;
  if(n.includes('__scept_mask')){
   const m=sessionMetadata(n),raw=m?.dims||m?.dimensions;
   if(!raw||raw.some(x=>!Number.isInteger(x)||x<1))throw Error('干预 mask 的维度无法自动确定：'+n);
   const len=product(raw);if(len>35000000)throw Error('干预 mask 太大');const arr=new Float32Array(len).fill(1);
   if(n===maskName&&ablate){const idx=Number(el('ablateIndex').value);if(!Number.isInteger(idx)||idx<0||idx>=len)throw Error('内部干预索引超出范围（0-'+(len-1)+'）');arr[idx]=0}
   result[n]=new lib.Tensor('float32',arr,raw);
  }else throw Error('模型具有额外输入 '+n+'。请导出单输入 ONNX 或仪器化模型；不会自动伪造未知输入。');
 }
 return result;
}
async function runOnce(lib,tensor,shape,maskName,ablate){
 const name=baseInputName(),type=metadataType(name),data=bytesToTensorData(tensor,type),feeds={[name]:new lib.Tensor(type,data,shape),...additionalFeeds(lib,maskName,ablate)};
 const t=performance.now(),outs=await session.run(feeds),elapsed=performance.now()-t;
 const result={};for(const [name,obj] of Object.entries(outs)){const nums=Array.from(obj.data,x=>typeof x==='bigint'?Number(x):Number(x));if(nums.some(n=>!Number.isFinite(n)))throw Error('输出张量 '+name+' 存在 NaN/Infinity');
   result[name]={dims:obj.dims,type:obj.type,values:nums};}
 return {outputs:result,elapsed};
}
function stats(a,b){const n=Math.min(a.length,b.length);let sum=0,max=0,changed=0;
 for(let i=0;i<n;i++){const d=b[i]-a[i];sum+=d*d;max=Math.max(max,Math.abs(d));if(Math.abs(d)>1e-8)changed++}
 return {l2:Math.sqrt(sum),max,count:n,changed};
}
function drawTopology(){
 const W=viz.width,H=viz.height;vctx.clearRect(0,0,W,H);vctx.fillStyle='#0a1322';vctx.fillRect(0,0,W,H);
 const nodes=manifest?.graph?.nodes;
 vctx.fillStyle='#9eb4cc';vctx.font='14px sans-serif';
 if(!Array.isArray(nodes)||!nodes.length){vctx.fillText('真实 ONNX 拓扑尚未导入：请上传由 scept_instrument_onnx.py 生成的模型配置 JSON',32,55);return}
 const shown=nodes.slice(0,30),pos=new Map();const cols=6,space=(W-115)/cols;
 shown.forEach((n,i)=>{const col=Math.floor(i/5),row=i%5;pos.set(n.id,{x:56+col*space,y:52+row*67})});
 const producer=new Map();
 for(const n of nodes)for(const k of n.outputs||[])producer.set(k,n.id);
 vctx.strokeStyle='#447e84';vctx.lineWidth=1;
 for(const node of shown){const to=pos.get(node.id);for(const input of node.inputs||[]){const from=pos.get(producer.get(input));if(!from||from.x===to.x&&from.y===to.y)continue;
 vctx.beginPath();vctx.moveTo(from.x+38,from.y);vctx.lineTo(to.x-38,to.y);vctx.stroke()}}
 for(const node of shown){const p=pos.get(node.id);vctx.fillStyle='#122c39';vctx.strokeStyle='#408f8c';vctx.beginPath();vctx.roundRect(p.x-39,p.y-21,78,43,7);vctx.fill();vctx.stroke();
 vctx.fillStyle='#dcf0ff';vctx.font='11px sans-serif';vctx.textAlign='center';vctx.fillText(String(node.op).slice(0,12),p.x,p.y-3);vctx.fillStyle='#83abb5';vctx.font='9px sans-serif';vctx.fillText('node '+node.id,p.x,p.y+12)}
 vctx.textAlign='left';vctx.fillStyle='#89a4bb';vctx.font='11px sans-serif';vctx.fillText('实际 ONNX 张量依赖 · 共 '+nodes.length+' 个节点 · 仅显示前 '+shown.length+' 个',16,H-12);
}
function drawPlot(){
 if(el('displayMode').value==='topology'){drawTopology();return}
 if(!lastRecord)return;const key=el('observedOutput').value,entry=lastRecord.tensors[key];if(!entry)return;
 const mode=el('displayMode').value,a=entry.before,b=entry.after;
 const W=viz.width,H=viz.height;vctx.clearRect(0,0,W,H);vctx.fillStyle='#0a1322';vctx.fillRect(0,0,W,H);
 const values=(mode==='base'?a:mode==='edited'?b:b.map((v,i)=>v-a[i]));const maxItems=150;
 const stride=Math.max(1,Math.ceil(values.length/maxItems)),points=[];
 for(let i=0;i<values.length;i+=stride){let val=0;for(let j=i;j<Math.min(i+stride,values.length);j++)val+=values[j];points.push({index:i,v:val/Math.min(stride,values.length-i)})}
 const max=Math.max(1e-8,...points.map(p=>Math.abs(p.v))),cy=H/2;
 vctx.strokeStyle='#243752';vctx.lineWidth=1;for(let i=0;i<7;i++){const y=30+i*(H-60)/6;vctx.beginPath();vctx.moveTo(43,y);vctx.lineTo(W-22,y);vctx.stroke()}
 vctx.strokeStyle='#51718b';vctx.beginPath();vctx.moveTo(43,cy);vctx.lineTo(W-22,cy);vctx.stroke();
 const gap=(W-70)/Math.max(points.length,1),bar=Math.max(1,gap-2);
 points.forEach((p,i)=>{const h=p.v/max*(H*.37),x=48+i*gap;
 vctx.fillStyle=p.v>=0?'#49d4c0':'#ec9b7c';vctx.fillRect(x,h>=0?cy-h:cy,bar,Math.abs(h));});
 vctx.font='12px sans-serif';vctx.fillStyle='#91acc3';vctx.fillText('+'+compact(max),8,35);vctx.fillText('0',22,cy-9);vctx.fillText('−'+compact(max),8,H-21);
 vctx.fillStyle='#a1b9ce';vctx.fillText(mode==='delta'?'Δ = edited − baseline':mode.toUpperCase(),60,19);
 vctx.fillText('实际张量元素 '+values.length+' · 每柱采样/聚合 '+stride+' 个',W-340,H-14);
 const s=stats(a,b);el('statL2').textContent=compact(s.l2);el('statMax').textContent=compact(s.max);el('statCount').textContent=s.count.toLocaleString();
}
function setOutputOptions(r){const s=el('observedOutput');s.innerHTML='';for(const k of Object.keys(r.tensors)){const op=document.createElement('option');op.value=k;op.textContent=k+' · '+r.tensors[k].dims.join('×');s.appendChild(op)}
 el('emptyViz').style.display='none';drawPlot()
}
el('observedOutput').addEventListener('change',drawPlot);el('displayMode').addEventListener('change',drawPlot);
function db(){return new Promise((resolve,reject)=>{const q=indexedDB.open('scept-studio-v1',1);q.onupgradeneeded=()=>q.result.createObjectStore('runs',{keyPath:'run_id'});q.onsuccess=()=>resolve(q.result);q.onerror=()=>reject(q.error)})}
async function dbOperation(action,record){try{const d=await db();return await new Promise((res,rej)=>{const t=d.transaction('runs',action==='read'?'readonly':'readwrite'),s=t.objectStore('runs');const q=action==='put'?s.put(record):action==='get'?s.get(record):action==='read'?s.getAll():s.clear();q.onsuccess=()=>res(q.result);q.onerror=()=>rej(q.error);t.oncomplete=()=>d.close()})}catch(e){console.warn('IndexedDB storage unavailable:',e);return action==='read'?[]:null}}
function historyItem(r){const elx=document.createElement('div');elx.className='history-record';const info=document.createElement('div'),title=document.createElement('strong'),small=document.createElement('small'),btn=document.createElement('button');
 title.textContent=r.model.name;small.textContent=new Date(r.timestamp).toLocaleString()+' · '+r.mode+' · '+r.run_id.slice(0,8);info.append(title,small);btn.textContent='载入记录';btn.addEventListener('click',async()=>{const stored=await dbOperation('get',r.run_id);if(stored?.tensors){lastRecord=stored;setOutputOptions(stored);el('evidencePill').textContent='STORED TRACE';notice('已打开保存的运行结果 '+r.run_id)}else notice('记录数据不可用，请重新执行实验。',true)});
 elx.append(info,btn);return elx}
function renderHistory(){const host=el('historyList');host.innerHTML='';if(!history.length){host.textContent='尚无运行记录。';return}for(const r of history.slice(0,20))host.append(historyItem(r))}
el('clearHistory').addEventListener('click',async()=>{if(!confirm('确定清空此设备保存的 SCEPT 实验记录吗？'))return;await dbOperation('clear');history=[];renderHistory();notice('本机记录已清空')});
async function readHistory(){const all=await dbOperation('read');history=all.map(x=>({run_id:x.run_id,model:x.model,timestamp:x.timestamp,mode:x.mode})).sort((a,b)=>b.timestamp.localeCompare(a.timestamp));renderHistory()}
async function saveRun(r){await dbOperation('put',r);history.unshift({run_id:r.run_id,model:r.model,timestamp:r.timestamp,mode:r.mode});history=history.slice(0,20);renderHistory()}
function exportFile(obj,mime,name){const url=URL.createObjectURL(new Blob([obj],{type:mime}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
el('downloadJson').addEventListener('click',()=>{if(lastRecord)exportFile(JSON.stringify(lastRecord),'application/json','SCEPT-'+lastRecord.run_id+'.json')});
el('downloadCsv').addEventListener('click',()=>{if(!lastRecord)return;let rows=['tensor,index,baseline,edited,delta'];for(const [key,t] of Object.entries(lastRecord.tensors)){for(let i=0;i<t.before.length;i++)rows.push(JSON.stringify(key)+','+i+','+t.before[i]+','+t.after[i]+','+(t.after[i]-t.before[i]))}exportFile(rows.join('\n'),'text/csv','SCEPT-'+lastRecord.run_id+'.csv')});
async function execute(){
 if(!session||runBusy)return;runBusy=true;el('runExperiment').disabled=true;setState('正在推理…',false);
 try{
 const lib=await ensureOrt(),shape=numberDims(),inputName=baseInputName();
 if(!session.inputNames.includes(inputName))throw Error('模型中找不到输入张量：'+inputName);
 const original=getInput(shape),selectedMask=el('maskInput').value;
 const edited=selectedMask?Float32Array.from(original):modifyInput(original,shape);
 if(original.length!==product(shape))throw Error('输入形状与数据长度不一致');
 notice('执行基线前向传播（真实 ONNX）…');
 const b=await runOnce(lib,original,shape,selectedMask,false);
 notice('执行干预前向传播（真实 ONNX）…');
 const a=await runOnce(lib,edited,shape,selectedMask,true);
 const outputNames=Object.keys(b.outputs);if(outputNames.join(',')!==Object.keys(a.outputs).join(','))throw Error('两个运行的输出结构不一致');
 const tensors={},fingerprints={};
 for(const key of outputNames){const tb=b.outputs[key],ta=a.outputs[key];if(tb.values.length!==ta.values.length)throw Error('输出张量维度不一致：'+key);
  const cap=2000000,trimmed=tb.values.length>cap;
  const valBefore=tb.values.slice(0,cap),valAfter=ta.values.slice(0,cap);
  tensors[key]={dims:tb.dims,type:tb.type,before:valBefore,after:valAfter,complete:!trimmed,original_length:tb.values.length};
  fingerprints[key]={before_sha256:await hashFloat(Float32Array.from(tb.values)),after_sha256:await hashFloat(Float32Array.from(ta.values)),full_length:tb.values.length,export_truncated:trimmed};
 }
 const source={kind:sourceMode,filename:sourceMode==='mri'?sourceVolume?.filename:sourceMode==='tensor'?sourceTensor?.filename:null,warning:sourceMode==='mri'?sourceVolume?.affineWarning:null};
 const inputSha=await hashFloat(original),editedSha=await hashFloat(edited),runId=crypto.randomUUID();
 const changed=Object.fromEntries(Object.entries(tensors).map(([k,v])=>[k,stats(v.before,v.after)]));
 const r={schema:'scept.trace.v1',run_id:runId,timestamp:new Date().toISOString(),runtime:'onnxruntime-web@'+ORT_VERSION,
  execution_provider_requested:el('webgpu').checked?'webgpu':'wasm',execution_provider_session:actualProvider,location:'local browser',scientific_validation:'NOT_PERFORMED',
  evidence_level:selectedMask?'L1_NUMERIC_INTERNAL_INTERVENTION':'L1_NUMERIC_INPUT_PERTURBATION',
  warning:'L1 数值变化不是自动成立的因果解释；只有模型暴露的输出张量属于已观察状态。',
  model:{name:modelName,sha256:modelHash,source:modelSource,url:modelUrl,trainable_parameters_modified:false},
  input:{source,dims:shape,input_name:inputName,normalize:el('normalize').value,baseline_sha256:inputSha,edited_sha256:editedSha,
    center,perturbation:selectedMask?null:{kind:el('perturb').value,strength:+el('strength').value,radius:+el('radius').value}},
  intervention:selectedMask?{kind:'ONNX_MASK_SINGLE_ELEMENT',input:selectedMask,index:Number(el('ablateIndex').value)}:null,
  execution:{baseline_ms:b.elapsed,intervention_ms:a.elapsed},
  output_fingerprints:fingerprints,summary:changed,tensors};
 lastRecord=r;setOutputOptions(r);el('statTime').textContent=compact(b.elapsed+a.elapsed)+'ms';el('evidencePill').textContent='L1 · REAL FORWARD';
 el('downloadJson').disabled=false;el('downloadCsv').disabled=false;
 await saveRun(r);notice('✓ 完成两次真实前向传播 · '+runId.slice(0,8)+' · '+outputNames.length+' 个可观测输出。结果已保存到本机。');
 }catch(err){notice('实验失败：'+err.message,true);console.error(err)}
 finally{runBusy=false;setState(session?'模型就绪':'无模型',!!session)}
}
el('runExperiment').addEventListener('click',execute);
el('selfTest').addEventListener('click',async()=>{
 if(runBusy||!session)return;
 const old=el('perturb').value,mask=el('maskInput').value,oldMask=mask;
 el('perturb').value='none';el('maskInput').value='';
 await execute();
 const ok=lastRecord?.summary&&Object.values(lastRecord.summary).every(s=>s.max<1e-6);
 notice(ok?'恒等对照 PASS：同模型同输入重复计算的输出差异小于 1e-6':'恒等对照 FAIL 或未执行：请审查实际输出。',!ok);
 el('perturb').value=old;el('maskInput').value=oldMask;
});
renderDigit();readHistory();setState('未加载模型',false);
