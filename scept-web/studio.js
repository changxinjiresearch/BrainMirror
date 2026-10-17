'use strict';
/* SCEPT Chinese research studio. Frozen pretrained ViT; local two-pass CPU inference.
   Scientific contract: synthetic image-like tensor and exploratory perturbations,
   not MRI diagnosis or causal explanation evidence. */
(function(){
const $=id=>document.getElementById(id);
const SOURCE='https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/resolve/ebffd7b9c0c53b51bada44116abd8f01aed41be6/onnx/model.onnx';
const SOURCE_HASH='32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3';
const PROBE='/vit/encoder/layer.6/attention/attention/value/MatMul_output_0';
const SIDE=224,PIXELS=SIDE*SIDE,SHAPE=[1,3,SIDE,SIDE],EPSILON=1e-7;
const st={session:null,graph:null,sha:null,base:null,edited:null,roiX:112,roiY:112,radius:18,amount:0.15,kind:'合成数值输入',audit:null,busy:false,error:null,sourceVerified:false};
const ctx=$('inputCanvas').getContext('2d');
let history=[];try{history=JSON.parse(localStorage.getItem('scept-studio-history')||'[]')}catch{};
function sha256(buffer){return crypto.subtle.digest('SHA-256',buffer).then(x=>Array.from(new Uint8Array(x)).map(b=>b.toString(16).padStart(2,'0')).join(''))}
function fmt(x){return Number.isFinite(x)?(Math.abs(x)<.00001&&x!==0?x.toExponential(2):x.toFixed(5)):'—'}
function setStatus(msg,mode='loading'){ $('modelStateText').textContent=msg; $('modelState').dataset.mode=mode; }
function warn(msg){st.error=msg;$('notice').textContent=msg;$('notice').classList.add('error');$('progressText').textContent='请检查网络连接或刷新后重试';}
function clearWarning(){st.error=null;$('notice').classList.remove('error');$('notice').textContent=st.kind==='合成数值输入'?'默认使用合成数值测试输入，并非真实 MRI。这里展示的是模型计算变化，不代表医学诊断或因果解释。':'当前是用户本地导入的普通图片。这里只做简化的 224×224、0–1 预处理，未经验证与预训练模型原始预处理完全一致，不能作为可靠分类结论。';}
function bytesJoin(...blocks){let size=0;for(const b of blocks)size+=b.length;const out=new Uint8Array(size);let pos=0;for(const b of blocks){out.set(b,pos);pos+=b.length}return out}
function varint(n){let arr=[];do{let v=n%128;n=Math.floor(n/128);if(n)v+=128;arr.push(v)}while(n);return Uint8Array.from(arr)}
function readVar(a,p,end){let n=0,m=1;for(let i=0;i<9&&p<end;i++){let v=a[p++];n+=(v&127)*m;if(!(v&128))return{value:n,end:p};m*=128}throw Error('模型文件的数值格式无效')}
function fields(a,start,end,fn){let p=start;while(p<end){const keyStart=p,k=readVar(a,p,end);p=k.end;const f=Math.floor(k.value/8),w=k.value%8;
 if(w===2){const l=readVar(a,p,end),s=l.end,t=s+l.value;if(!Number.isSafeInteger(t)||t>end)throw Error('模型文件结构损坏');fn({field:f,start:keyStart,payloadStart:s,payloadEnd:t,fieldEnd:t});p=t}
 else if(w===0)p=readVar(a,p,end).end;
 else if(w===1)p+=8;
 else if(w===5)p+=4;
 else throw Error('暂不支持这种 ONNX 文件字段类型');
 if(p>end)throw Error('模型文件越界');
}}
function addObservedOutput(original,name){
 const a=new Uint8Array(original);if(a.byteLength>150000000)throw Error('模型过大');
 let graph=null;fields(a,0,a.length,x=>{if(x.field===7)graph=x});if(!graph)throw Error('预训练模型缺少计算图');
 let entry=null,has=false;const decoder=new TextDecoder();
 fields(a,graph.payloadStart,graph.payloadEnd,x=>{
   if(x.field!==13&&x.field!==12)return;
   let current=null;fields(a,x.payloadStart,x.payloadEnd,t=>{if(t.field===1)current=decoder.decode(a.subarray(t.payloadStart,t.payloadEnd))});
   if(current===name){if(x.field===12)has=true;else entry=a.slice(x.payloadStart,x.payloadEnd)}
 });
 if(has)return original;
 if(!entry)throw Error('找不到第 6 层中间观测张量');
 const extra=bytesJoin(varint(98),varint(entry.length),entry);
 const fullGraph=bytesJoin(a.subarray(graph.payloadStart,graph.payloadEnd),extra);
 return bytesJoin(a.subarray(0,graph.start),varint(58),varint(fullGraph.length),fullGraph,a.subarray(graph.fieldEnd)).buffer;
}
function synthetic(){
 const input=new Float32Array(PIXELS*3);
 for(let i=0;i<input.length;i++)input[i]=((i*73)%256)/255;
 return input;
}
function renderInput(){
 if(!st.base)return;const edited=makeEdited();
 const image=ctx.createImageData(SIDE,SIDE);
 for(let i=0;i<PIXELS;i++){const j=4*i;image.data[j]=Math.max(0,Math.min(255,edited[i]*255));image.data[j+1]=Math.max(0,Math.min(255,edited[PIXELS+i]*255));image.data[j+2]=Math.max(0,Math.min(255,edited[2*PIXELS+i]*255));image.data[j+3]=255;}
 ctx.putImageData(image,0,0);
 ctx.strokeStyle='#71f4c1';ctx.lineWidth=2.2;ctx.setLineDash([5,3]);ctx.beginPath();ctx.arc(st.roiX,st.roiY,st.radius,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle='#b5fff1';ctx.beginPath();ctx.arc(st.roiX,st.roiY,2.4,0,Math.PI*2);ctx.fill();
}
function makeEdited(){
 if(!st.base)return null;
 const out=new Float32Array(st.base),r2=st.radius*st.radius,x0=Math.max(0,Math.floor(st.roiX-st.radius)),x1=Math.min(SIDE-1,Math.ceil(st.roiX+st.radius)),y0=Math.max(0,Math.floor(st.roiY-st.radius)),y1=Math.min(SIDE-1,Math.ceil(st.roiY+st.radius));
 for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
  if((x-st.roiX)**2+(y-st.roiY)**2>r2)continue;
  const idx=y*SIDE+x;for(let ch=0;ch<3;ch++)out[ch*PIXELS+idx]+=st.amount;
 }
 return out;
}
function showSelectedInput(){
 $('radiusValue').textContent=st.radius+' px';$('amountValue').textContent=(st.amount>=0?'+':'')+st.amount.toFixed(2);
 $('inputType').textContent=st.kind;
 $('startBtn').textContent=st.session?(st.audit?'重新计算':'开始实验'):'正在准备模型…';
 renderInput();clearWarning();
}
function updateFromPointer(e){
 const rect=$('inputCanvas').getBoundingClientRect();
 const x=(e.clientX-rect.left)*SIDE/rect.width,y=(e.clientY-rect.top)*SIDE/rect.height;
 st.roiX=Math.round(Math.max(0,Math.min(SIDE-1,x)));st.roiY=Math.round(Math.max(0,Math.min(SIDE-1,y)));
 showSelectedInput();$('inputTip').textContent='已选择新的修改位置：('+st.roiX+', '+st.roiY+')。点击「重新计算」查看真实模型响应。';
}
$('inputCanvas').addEventListener('pointerdown',updateFromPointer);
$('inputCanvas').addEventListener('keydown',e=>{
 if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;
 e.preventDefault();st.roiX=Math.max(0,Math.min(223,st.roiX+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0)));st.roiY=Math.max(0,Math.min(223,st.roiY+(e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0)));showSelectedInput();
});
$('roiRadius').oninput=e=>{st.radius=Number(e.target.value);showSelectedInput()};
$('perturbAmount').oninput=e=>{st.amount=Number(e.target.value);showSelectedInput()};
$('resetBtn').onclick=()=>{st.base=synthetic();st.kind='合成数值输入';st.roiX=112;st.roiY=112;st.radius=18;st.amount=.15;$('roiRadius').value='18';$('perturbAmount').value='.15';showSelectedInput()};
$('photoInput').onchange=async e=>{
 const file=e.target.files?.[0];if(!file)return;
 if(file.size>12000000){warn('图片不能超过 12 MB');return}
 try{const bitmap=await createImageBitmap(file),cv=document.createElement('canvas');cv.width=SIDE;cv.height=SIDE;const cx=cv.getContext('2d',{willReadFrequently:true});
 cx.drawImage(bitmap,0,0,SIDE,SIDE);bitmap.close?.();const pixels=cx.getImageData(0,0,SIDE,SIDE).data,d=new Float32Array(PIXELS*3);
 for(let i=0;i<PIXELS;i++)for(let ch=0;ch<3;ch++)d[ch*PIXELS+i]=pixels[i*4+ch]/255;
 st.base=d;st.kind='用户图片 · 简化预处理';st.roiX=112;st.roiY=112;st.audit=null;showSelectedInput();
 }catch(ex){warn('图片读取失败：'+ex.message)}
};
function renderHistory(){const root=$('historyList');root.replaceChildren();if(!history.length){root.innerHTML='<div class="empty-record">尚未运行实验。点击「开始实验」生成第一条记录。</div>';return}
 for(const h of history.slice(0,6)){const row=document.createElement('div');row.className='record-row';const a=document.createElement('span'),b=document.createElement('small');a.textContent=h.date+' · '+h.kind;b.textContent=h.changed+' 个隐藏数值变化';row.append(a,b);root.append(row)}
}
function plotDeltas(a,b){
 const root=$('tensorPlot');root.replaceChildren();
 const canvas=document.createElement('canvas');canvas.width=960;canvas.height=220;canvas.setAttribute('aria-label','实际中间层前64个元素的数值变化条形图');
 const ctx=canvas.getContext('2d'),pad=22,mid=canvas.height-30,H=165,W=canvas.width;
 ctx.fillStyle='#071923';ctx.fillRect(0,0,W,canvas.height);const n=Math.min(64,a.length),vals=[];
 for(let i=0;i<n;i++)vals.push(Math.abs(b[i]-a[i]));
 const max=Math.max(1e-12,...vals);const stride=(W-2*pad)/n;
 ctx.strokeStyle='#31555c';ctx.beginPath();ctx.moveTo(pad,mid);ctx.lineTo(W-pad,mid);ctx.stroke();
 for(let i=0;i<n;i++){const h=Math.max(1,vals[i]/max*H);ctx.fillStyle=vals[i]>EPSILON?'#62d5b5':'#426c70';ctx.fillRect(pad+i*stride,mid-h,Math.max(2,stride-3),h)}
 ctx.fillStyle='#9fbabc';ctx.font='12px system-ui';ctx.textAlign='left';ctx.fillText('真实激活差异 · |Δ|（前 64 项）',pad,18);ctx.fillText('元素 0',pad,213);ctx.textAlign='right';ctx.fillText('元素 63',W-pad,213);
 root.append(canvas);
}
function saveBlob(content,name,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2500)}
$('exportJson').onclick=()=>{if(st.audit)saveBlob(JSON.stringify(st.audit,null,2),'scept-transformer-audit.json','application/json')};
$('exportCsv').onclick=()=>{const rec=st.audit?.observations?.[PROBE];if(!rec)return;const rows=['index,baseline,edited,delta'];for(let i=0;i<rec.before.length;i++)rows.push([i,rec.before[i],rec.after[i],rec.after[i]-rec.before[i]].join(','));saveBlob(rows.join('\n'),'scept-transformer-hidden-delta.csv','text/csv')};
async function loadModel(){
 if(st.busy)return;st.busy=true;$('startBtn').disabled=true;
 try{
  setStatus('正在下载预训练模型…');
  $('progressText').textContent='首次访问需要下载约 23 MB 模型文件';
  if(!window.ort)throw Error('浏览器推理组件加载失败，请检查网络');
  const res=await fetch(SOURCE,{mode:'cors',cache:'force-cache'});if(!res.ok)throw Error('模型下载失败，HTTP '+res.status);
  const orig=await res.arrayBuffer();setStatus('正在核对模型文件…');
  const digest=await sha256(orig);if(digest!==SOURCE_HASH)throw Error('模型校验失败，出于安全原因已中止');
  st.sourceVerified=true;$('sourceSha').textContent=digest;
  setStatus('正在准备内部观测…');
  const modified=addObservedOutput(orig,PROBE);st.sha=await sha256(modified);$('instrumentSha').textContent=st.sha;
  st.graph=window.SCEPT_INSPECT_ONNX(modified);
  st.graph&&window.SCEPT_GRAPH3D?.render(st.graph,null);
  $('layerLabel').textContent='ViT / 第 6 层';
  $('probeName').textContent=PROBE;
  setStatus('正在初始化浏览器推理…');
  window.ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  window.ort.env.wasm.numThreads=1;
  st.session=await window.ort.InferenceSession.create(new Uint8Array(modified),{executionProviders:['wasm']});
  if(!st.session.inputNames.includes('pixel_values')||!st.session.outputNames.includes(PROBE))throw Error('预训练模型的输入或内部观测点不匹配');
  setStatus('模型已就绪','ready');$('progressText').textContent='可以开始：用同一套模型权重比较输入修改前后';
  $('startBtn').disabled=false;$('startBtn').textContent='开始实验';clearWarning();
  if(new URLSearchParams(location.search).get('run')==='transformer')await execute();
 }catch(e){setStatus('模型准备失败','error');warn('模型准备失败：'+e.message);$('startBtn').textContent='重新加载模型';$('startBtn').disabled=false}
 finally{st.busy=false}
}
async function execute(){
 if(!st.session){await loadModel();return}if(st.busy)return;
 st.busy=true;$('startBtn').disabled=true;$('startBtn').textContent='正在运行真实模型…';setStatus('计算进行中…');
 try{
  const edited=makeEdited(),baseline=st.base;
  const feeds0={pixel_values:new ort.Tensor('float32',baseline,SHAPE)},feeds1={pixel_values:new ort.Tensor('float32',edited,SHAPE)};
  $('progressText').textContent='第 1 次：计算原始输入…';
  const a=await st.session.run(feeds0);
  $('progressText').textContent='第 2 次：计算修改后的输入…';
  const b=await st.session.run(feeds1);
  const obs={},allNames=st.session.outputNames;
  for(const name of allNames){const x=a[name],y=b[name];if(!x?.data||!y?.data||x.data.length!==y.data.length)throw Error('模型输出张量不可比较：'+name);
   obs[name]={shape:Array.from(x.dims),before:Array.from(x.data,Number),after:Array.from(y.data,Number)};
  }
  const x=obs[PROBE];if(!x||x.before.length!==37824)throw Error('第 6 层观测张量长度不符合预期');
  let delta=0,max=0,changed=0;
  for(let i=0;i<x.before.length;i++){const diff=Math.abs(x.after[i]-x.before[i]);delta+=diff;max=Math.max(max,diff);if(diff>EPSILON)changed++}
  st.edited=edited;
  st.audit={schema:'scept.studio.v1',time:new Date().toISOString(),evidence_level:'L1_REAL_NUMERIC',model:'ViT tiny pre-trained FP32',source_model_sha256:SOURCE_HASH,observed_model_sha256:st.sha,input_kind:st.kind,input_shape:SHAPE,input_original_sha256:await sha256(baseline.buffer),input_modified_sha256:await sha256(edited.buffer),roi:{x:st.roiX,y:st.roiY,radius_px:st.radius,delta:st.amount},runtime:'ONNX Runtime Web / CPU WASM',notes:'Real pre-trained ViT hidden activations, not a medical MRI result, not proof of causal decision dependence.',observations:obs};
  $('changedMetric').textContent=changed.toLocaleString('zh-CN')+' / 37,824';
  $('meanMetric').textContent=fmt(delta/x.before.length);$('maxMetric').textContent=fmt(max);
  plotDeltas(x.before,x.after);window.SCEPT_GRAPH3D?.render(st.graph,obs);
  $('exportJson').disabled=false;$('exportCsv').disabled=false;
  history.unshift({date:new Date().toLocaleString('zh-CN'),kind:st.kind,changed});history=history.slice(0,15);
  try{localStorage.setItem('scept-studio-history',JSON.stringify(history))}catch{}
  renderHistory();setStatus('真实观测完成','ready');
  $('progressText').textContent='运行成功 · 已更新真实网络与第 6 层数据';
  $('inputTip').textContent='已比较修改前后的真实模型计算。拖动或点击左侧输入区域后，再点击「重新计算」。';
 }catch(e){setStatus('运行未完成','error');warn('实验失败：'+e.message)}
 finally{st.busy=false;$('startBtn').disabled=false;$('startBtn').textContent=st.session?'重新计算':'重新加载模型'}
}
$('startBtn').onclick=()=>st.session?execute():loadModel();
st.base=synthetic();showSelectedInput();renderHistory();loadModel();
window.SCEPT_STUDIO={getSnapshot:()=>({modelReady:!!st.session,sourceVerified:st.sourceVerified,hasAudit:!!st.audit,changed:$('changedMetric').textContent,kind:st.kind}),run:execute};
})();
