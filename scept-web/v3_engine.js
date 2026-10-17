'use strict';
/* SCEPT v3 engine: frozen pretrained ViT with multiple genuine ONNX graph probes.
 * The MRI is authentic MNI152-derived atlas data. ViT is NOT MRI-trained.
 * This validates engineering observability and independent interventions only.
 * Network colors are measured activation-group deltas; no synthetic signals.
 */
(function(){
const $=id=>document.getElementById(id);
const SOURCE='https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/resolve/ebffd7b9c0c53b51bada44116abd8f01aed41be6/onnx/model.onnx';
const SOURCE_HASH='32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3';
const PROBE_LAYERS=[2,4,6,8,10];
const PROBE_NAME=i=>`/vit/encoder/layer.${i}/attention/attention/value/MatMul_output_0`;
const st={session:null,graph:null,modelSha:null,sourceHash:SOURCE_HASH,probes:[],results:null,selected:6,history:[],loading:true,running:false,failed:null};
let mouse=null;
try{st.history=JSON.parse(localStorage.getItem('scept-v3-audit-list')||'[]')}catch{}
const fmt=n=>Number.isFinite(n)?Math.abs(n)>.000001?n.toFixed(5):n.toExponential(2):'—';
const sha256=async buf=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buf))).map(v=>v.toString(16).padStart(2,'0')).join('');
function concat(...blocks){let size=0;for(const b of blocks)size+=b.length;const result=new Uint8Array(size);let p=0;for(const b of blocks){result.set(b,p);p+=b.length}return result}
function varint(n){const out=[];do{let t=n%128;n=Math.floor(n/128);if(n)t+=128;out.push(t)}while(n);return Uint8Array.from(out)}
function getVar(b,p,end){let n=0,m=1;for(let i=0;i<10&&p<end;i++){const v=b[p++];n+=(v&127)*m;if(!(v&128))return{n,p};m*=128}throw Error('错误的 ONNX protobuf 编码')}
function fields(b,start,end,fn){let p=start;while(p<end){const begin=p,t=getVar(b,p,end);p=t.p;const w=t.n%8,f=Math.floor(t.n/8);
 if(w===2){const len=getVar(b,p,end),s=len.p,e=s+len.n;if(e>end||!Number.isSafeInteger(e))throw Error('ONNX 长度无效');fn({f,begin,s,e});p=e;}
 else if(w===0)p=getVar(b,p,end).p;else if(w===1)p+=8;else if(w===5)p+=4;else throw Error('不支持的 ONNX 字段');
 if(p>end)throw Error('ONNX 边界无效');
}}
function observeRealLayers(input){
 const b=new Uint8Array(input);if(b.byteLength>120000000)throw Error('当前演示模型超出大小上限');
 let graph=null;fields(b,0,b.length,x=>{if(x.f===7)graph=x});if(!graph)throw Error('模型缺少计算图');
 const wanted=new Map(PROBE_LAYERS.map(i=>[PROBE_NAME(i),i])),found=new Map(),existing=new Set(),dec=new TextDecoder();
 fields(b,graph.s,graph.e,x=>{
  if(x.f!==13&&x.f!==12)return;
  let name=null;fields(b,x.s,x.e,y=>{if(y.f===1)name=dec.decode(b.subarray(y.s,y.e))});
  if(name&&wanted.has(name)){
   if(x.f===12)existing.add(name);
   else found.set(name,b.slice(x.s,x.e));
  }
 });
 let patched=b.subarray(graph.s,graph.e),exposed=[];
 for(const [name,layer]of wanted){
  if(existing.has(name)){exposed.push({layer,name});continue}
  const raw=found.get(name);if(!raw)continue;
  patched=concat(patched,varint(98),varint(raw.length),raw);
  exposed.push({layer,name});
 }
 if(exposed.length<3)throw Error('无法同时开放至少 3 个实际 Transformer 层，请检查模型版本');
 return{model:concat(b.subarray(0,graph.begin),varint(58),varint(patched.length),patched,b.subarray(graph.e)).buffer,exposed};
}
function status(msg,kind='pending'){
 $('readyLabel').textContent=msg;
 $('readyLabel').dataset.state=kind;
 $('runProgress').textContent=msg;
}
function visibleError(msg){st.failed=msg;$('warning').textContent=msg;}
async function init(){
 try{
  status('正在下载 ViT 冻结权重…');
  if(!window.ort)throw Error('浏览器无法下载 ONNX 推理库');
  const res=await fetch(SOURCE,{mode:'cors',cache:'force-cache'});
  if(!res.ok)throw Error('模型下载失败 HTTP '+res.status);
  const raw=await res.arrayBuffer();
  status('核对训练权重数字指纹…');
  if(await sha256(raw)!==SOURCE_HASH)throw Error('原始预训练权重 SHA-256 校验失败');
  $('sourceModelHash').textContent=SOURCE_HASH;
  status('开放多个真实隐藏层…');
  const patched=observeRealLayers(raw);
  st.probes=patched.exposed;
  st.modelSha=await sha256(patched.model);
  $('instrumentedHash').textContent=st.modelSha;
  st.graph=window.SCEPT_INSPECT_ONNX(patched.model);
  status('初始化本地 CPU 推理…');
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  ort.env.wasm.numThreads=1;
  st.session=await ort.InferenceSession.create(new Uint8Array(patched.model),{executionProviders:['wasm']});
  if(!st.probes.every(p=>st.session.outputNames.includes(p.name)))throw Error('实际模型未返回完整观测张量输出');
  const sel=$('layerSelect');sel.replaceChildren();
  for(const p of st.probes){const o=document.createElement('option');o.value=String(p.layer);o.textContent='第 '+p.layer+' 层';sel.add(o)}
  st.selected=st.probes.some(x=>x.layer===6)?6:st.probes[0].layer;sel.value=String(st.selected);
  status('模型就绪 · 等待 MRI','ready');st.loading=false;readyIfPossible();
 }catch(e){st.loading=false;st.failed=e.message;status('模型初始化失败','error');visibleError('MRI 工作台仍可查看，但模型无法运行：'+e.message)}
}
function readyIfPossible(){
 const mri=window.SCEPT_MRI?.state;
 const yes=!!st.session&&!!mri?.loaded;
 $('runExperiment').disabled=!yes;
 $('runExperiment').textContent=yes?'开始实验':st.failed?'模型加载失败':'等待 MRI / 模型';
 if(yes && !st.results && !st.running)status('模型和 MRI 已就绪','ready');
 return yes;
}
function metrics(before,after){
 if(before.length!==after.length)throw Error('模型输出形状不一致');
 let sum=0,max=0,changed=0,up=0,down=0;
 const chunkCount=8,groups=[];
 for(let k=0;k<chunkCount;k++)groups.push({sum:0,len:0});
 for(let i=0;i<before.length;i++){
  const delta=after[i]-before[i],abs=Math.abs(delta);
  if(!Number.isFinite(delta))throw Error('观测点出现 NaN 或 Infinity');
  sum+=abs;max=Math.max(max,abs);if(abs>1e-7){changed++;delta>0?up++:down++}
  const g=groups[Math.min(7,Math.floor(i*8/before.length))];g.sum+=abs;g.len++;
 }
 return{mean:sum/before.length,max,changed,up,down,total:before.length,groups:groups.map(g=>g.len?g.sum/g.len:0)};
}
async function run(){
 if(st.running)return;
 if(!readyIfPossible())return;
 st.running=true;$('runExperiment').disabled=true;$('runExperiment').textContent='计算中…';
 try{
  const before=window.SCEPT_MRI.makeSlice(false,2),after=window.SCEPT_MRI.makeSlice(true,2);
  $('runProgress').textContent='模型前向计算 1/2：原始脑切片';
  const a=await st.session.run({pixel_values:new ort.Tensor('float32',before,[1,3,224,224])});
  $('runProgress').textContent='模型前向计算 2/2：ROI 修改后脑切片';
  const b=await st.session.run({pixel_values:new ort.Tensor('float32',after,[1,3,224,224])});
  const layers=[];
  for(const p of st.probes){
   const xa=a[p.name],xb=b[p.name];
   if(!xa?.data||!xb?.data)throw Error('缺少模型第 '+p.layer+' 层的真实张量');
   const baseline=Array.from(xa.data,Number),edited=Array.from(xb.data,Number);
   const summary=metrics(baseline,edited);
   layers.push({layer:p.layer,probe:p.name,shape:Array.from(xa.dims),summary,before:baseline,after:edited});
  }
  st.results={layers,at:new Date().toISOString(),model_sha256:st.modelSha,pretrained_source_sha256:SOURCE_HASH,raw_mri_sha256:await window.SCEPT_MRI.sourceHash(),mri:window.SCEPT_MRI.state,baseline_input_sha256:await sha256(before.buffer),intervened_input_sha256:await sha256(after.buffer),experiment_type:'TRUE_MRI_ATLAS_INPUT__PRETRAINED_NATURAL_IMAGE_VIT_OUT_OF_DOMAIN',evidence:'L1 genuine numeric activation changes, NOT a medically valid or causal model explanation',preprocessing:'single native voxel i-j slice z=center, nearest-neighbor 224x224, 2-98 percentile clipped to 0..1, three repeated grayscale channels; ViT natural-image preprocessor NOT matched'};
  st.history.unshift({at:st.results.at,mri:st.results.mri.name,layers:layers.length,changed:layers.find(x=>x.layer===st.selected)?.summary.changed??null});st.history=st.history.slice(0,20);
  try{localStorage.setItem('scept-v3-audit-list',JSON.stringify(st.history))}catch{}
  drawAll();listHistory();$('saveAudit').disabled=false;$('saveLayers').disabled=false;
  $('runProgress').textContent='真实计算完成 · '+layers.length+' 个观测层可查看';
  $('networkState').textContent='已观测 '+layers.length+' 个层级';
  status('MRI → 真实多层观测完成','ready');
  $('interpretation').textContent='真实计算成功：选中的 MRI 三维 ROI 被修改，同一冻结 ViT 在 '+layers.length+' 个实际隐藏层记录了逐元素数值差异。由于该 ViT 未针对 MRI 训练，本结果只证明观测机制，不构成医学解释或病灶判断。';
 }catch(e){status('实验失败','error');visibleError('真实模型推理失败：'+e.message)}
 finally{st.running=false;readyIfPossible();$('runExperiment').textContent=st.results?'重新计算':'开始实验'}
}
function drawAll(){
 const layers=st.results?.layers??[];
 const selected=layers.find(x=>x.layer===st.selected);
 $('metricLayers').textContent=String(layers.length)+' 层';
 $('metricChanged').textContent=selected?selected.summary.changed.toLocaleString('zh-CN')+' / '+selected.summary.total.toLocaleString('zh-CN'):'—';
 $('metricMean').textContent=selected?fmt(selected.summary.mean):'—';
 const root=$('layerTable');root.replaceChildren();
 if(layers.length){
  const peak=Math.max(1e-12,...layers.map(x=>x.summary.mean));
  for(const l of layers){
   const row=document.createElement('div');row.className='layer-row'+(l.layer===st.selected?' selected':'');row.tabIndex=0;
   const label=document.createElement('span');label.textContent='第 '+l.layer+' 层';
   const outer=document.createElement('span');outer.className='bar';const bar=document.createElement('i');bar.style.width=(100*l.summary.mean/peak).toFixed(2)+'%';outer.append(bar);
   const value=document.createElement('strong');value.textContent=fmt(l.summary.mean);row.append(label,outer,value);
   row.onclick=()=>choose(l.layer);row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(l.layer)}};
   root.append(row);
  }
 }
 drawNetwork();drawResponse();
}
function choose(layer){st.selected=layer;$('layerSelect').value=String(layer);drawAll()}
$('layerSelect').onchange=e=>choose(Number(e.target.value));
function drawNetwork(){
 const canvas=$('networkCanvas'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
 ctx.fillStyle='#061823';ctx.fillRect(0,0,w,h);
 const p=st.results?.layers??[],look=new Map(p.map(x=>[x.layer,x])),max=Math.max(1e-9,...p.map(x=>x.summary.mean));
 const yBase=180,step=(w-102)/11;
 for(let layer=0;layer<12;layer++){
  const x=52+layer*step,info=look.get(layer),strength=info?Math.min(1,info.summary.mean/max):0;
  if(layer<11){ctx.strokeStyle='#29515b';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x+11,yBase);ctx.lineTo(x+step-11,yBase);ctx.stroke()}
  for(let group=0;group<8;group++){
   const y=yBase+(group-3.5)*24;
   const g=info?.summary.groups[group]??null;
   const c=g===null?null:Math.min(1,g/max),color=c===null?'#54737d':`rgba(${Math.floor(57+100*c)},${Math.floor(132+97*c)},${Math.floor(140-40*c)},1)`;
   ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,group===4&&layer===st.selected?8:6,0,2*Math.PI);ctx.fill();
   if(layer===st.selected&&info){ctx.strokeStyle='#b1f5dd';ctx.lineWidth=1.4;ctx.stroke()}
  }
  ctx.fillStyle=info?'#b6ede0':'#6f9098';ctx.textAlign='center';ctx.font='12px sans-serif';ctx.fillText('L'+layer,x,312);
 }
 ctx.fillStyle='#8ab9b4';ctx.font='12px sans-serif';ctx.textAlign='left';
 ctx.fillText('模型中的层级顺序 · 只给有真实观测张量的层上色',16,24);
 ctx.fillText('每个圆点 = 该层真实张量的 1/8 分块均值；不是单一生物神经元',16,345);
}
function drawResponse(){
 const canvas=$('responseChart'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
 ctx.fillStyle='#081923';ctx.fillRect(0,0,w,h);
 const layer=st.results?.layers.find(x=>x.layer===st.selected);
 if(!layer){ctx.fillStyle='#85aeb2';ctx.fillText('等待获取真实模型内部激活',14,30);return}
 let vmax=1e-12;for(let i=0;i<64;i++)vmax=Math.max(vmax,Math.abs(layer.after[i]-layer.before[i]));
 const gap=(w-35)/64;
 for(let i=0;i<64;i++){const d=Math.abs(layer.after[i]-layer.before[i]),bar=d/vmax*125;ctx.fillStyle=d>1e-7?'#62d5ba':'#49716e';ctx.fillRect(18+i*gap,h-23-bar,Math.max(2,gap-3),Math.max(1,bar))}
 ctx.fillStyle='#accbc6';ctx.font='12px sans-serif';ctx.fillText('第 '+st.selected+' 层：真实 |激活差异| 前 64 项',14,20);
}
function networkSelect(e){
 if(!st.results)return;
 const canvas=$('networkCanvas'),rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*canvas.width/rect.width;
 const layer=Math.max(0,Math.min(11,Math.round((x-52)/((canvas.width-102)/11))));
 if(st.results.layers.some(p=>p.layer===layer))choose(layer);
}
$('networkCanvas').onpointerdown=networkSelect;
function listHistory(){
 const root=$('records');root.replaceChildren();if(!st.history.length){root.textContent='尚无本次实验记录。';return}
 for(const r of st.history.slice(0,8)){const el=document.createElement('div');el.className='record';const a=document.createElement('span'),b=document.createElement('span');a.textContent=new Date(r.at).toLocaleString('zh-CN')+' · '+r.mri;b.textContent=r.layers+' 层观测';el.append(a,b);root.append(el)}
}
function save(content,name,type){const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1500)}
$('saveAudit').onclick=()=>{if(st.results)save(JSON.stringify(st.results,null,2),'scept-v3-mri-multilayer.json','application/json')};
$('saveLayers').onclick=()=>{if(!st.results)return;const rows=['layer,tensor_name,total,changed,mean_abs_delta,max_abs_delta,up,down'];for(const l of st.results.layers){const s=l.summary;rows.push([l.layer,l.probe,s.total,s.changed,s.mean,s.max,s.up,s.down].join(','))}save(rows.join('\n'),'scept-v3-layer-summary.csv','text/csv')};
$('runExperiment').onclick=run;
$('reloadTemplate').onclick=async()=>{await SCEPT_MRI.loadDefault();readyIfPossible()};
window.SCEPT_V3={getSnapshot:()=>({ready:!!st.session&&!!SCEPT_MRI.state.loaded,model:!!st.session,mri:SCEPT_MRI.state.loaded,probeLayers:st.probes.map(p=>p.layer),hasExperiment:!!st.results,selected:st.selected,runId:st.results?.at||null,changed:st.results?.layers.find(x=>x.layer===st.selected)?.summary.changed??null}),run};
listHistory();drawNetwork();drawResponse();init();
let attempts=0;
const check=setInterval(()=>{readyIfPossible();if(++attempts>240||readyIfPossible()){clearInterval(check)}},500);
})();