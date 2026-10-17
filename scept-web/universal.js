'use strict';
/* U0-U2 generic model workstation: no special casing model input names/shapes. */
(function(){
const $=id=>document.getElementById(id),registry=SCEPT_CORE.createRegistry();
const adapter=SCEPT_ONNX_ADAPTER.make({wasmPaths:'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'});
registry.register('model',adapter);
const dataAdapter=SCEPT_DATA_ADAPTER.make(SCEPT_CORE);
registry.register('data',dataAdapter);
const PRESETS={
 vit:{id:'pretrained-vit-tiny',url:'https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/resolve/ebffd7b9c0c53b51bada44116abd8f01aed41be6/onnx/model.onnx',sha256:'32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3'},
 cnn:{id:'mri-cnn-experimental',url:'https://huggingface.co/Udit013/brain-tumor-efficientnetb3/resolve/main/efficientnetb3.onnx',sha256:'320b1ef648f3da5216f6ee0374349324b3cb00a76bebf7dbd1c3796e6155991c'}
};
let state={bytes:null,hash:null,inspect:null,source:null,session:null,loaded:null,audit:null};
function note(id,msg,kind='pending'){const el=$(id);el.textContent=msg;el.dataset.state=kind}
function tableCells(parent,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td)}parent.append(tr)}
function safeData(buf){if(buf.byteLength>160*1024*1024)throw Error('Model exceeds 160 MB local resource guard');return buf}
async function inspectModel(){
 $('inspect').disabled=true;$('load').disabled=true;$('run').disabled=true;$('export').disabled=true;
 try{
  if(state.session){await state.session.dispose();state.session=null}
  state.bytes=null;state.inspect=null;state.audit=null;
  const source=$('modelPreset').value,selected=PRESETS[source];
  let buffer;
  if(selected){
   note('modelStatus','下载公开预训练模型并核对来源指纹…');
   const res=await fetch(selected.url,{cache:'force-cache'});
   if(!res.ok)throw Error('Model HTTP '+res.status);
   buffer=safeData(await res.arrayBuffer());
  }else{
   const file=$('modelFile').files?.[0];if(!file)throw Error('请先上传本地 .onnx 模型');
   if(file.size>160*1024*1024)throw Error('Model exceeds 160 MB local resource guard');
   buffer=await file.arrayBuffer();
  }
  const hash=await SCEPT_CORE.sha256(buffer);
  if(selected&&hash!==selected.sha256)throw Error('公开模型 SHA-256 校验失败，拒绝加载');
  state={...state,bytes:buffer,hash,source,loaded:null};
  const graph=SCEPT_INSPECT_ONNX(buffer);
  if(!graph.inputs.length||!graph.outputs.length)throw Error('无法识别 ONNX 输入/输出');
  state.inspect=graph;
  $('modelFacts').textContent='实际模型 SHA-256: '+hash+' · '+graph.totalNodes+' 个算子 · '+graph.edges.length+' 条数据依赖 · '+Object.keys(graph.intermediateMetadata).length+' 个具名可检查中间张量';
  const inputs=$('inputs');inputs.replaceChildren();
  for(const name of graph.inputs){const m=graph.inputMetadata[name];tableCells(inputs,[m.name,m.type,m.shape.join('×')])}
  const root=$('probesAvailable');root.replaceChildren();
  const entries=Object.values(graph.intermediateMetadata||{}).filter(e=>
   ['float32','float64'].includes(e.type)&&e.shape.length&&e.shape.every(v=>typeof v==='number'&&v>0)).slice(0,120);
  for(const m of entries){
   const line=document.createElement('label');line.className='probe';
   const cb=document.createElement('input');cb.type='checkbox';cb.value=m.name;cb.dataset.probe='true';
   cb.onchange=()=>{$('probes').value=Array.from(root.querySelectorAll('input:checked')).map(x=>x.value).join('\n')};
   line.append(cb,document.createTextNode(m.name+' ['+m.shape.join('×')+']'));root.append(line);
  }
  if(!entries.length)root.textContent='该 ONNX 文件没有带静态形状的可用中间张量元数据；仍可观察实际 graph outputs。';
  $('load').disabled=false;
  note('modelStatus','模型检查成功。现在可以选择中间张量，再点击「接入并验证模型」。','success');
 }catch(e){note('modelStatus','检查失败：'+e.message,'error');state.bytes=null;state.inspect=null}
 finally{$('inspect').disabled=false}
}
async function attach(){
 if(!state.bytes||!state.inspect)return;
 $('load').disabled=true;$('run').disabled=true;
 try{
  const probes=$('probes').value.split(/[\r\n]+/).map(s=>s.trim()).filter(Boolean);
  if(probes.length>16)throw Error('最多同时选择 16 个真实张量');
  const raw=state.bytes,manifest={modelId:PRESETS[state.source]?.id||'user-uploaded-onnx',adapterId:'onnx-wasm',sha256:state.hash,probes};
  note('modelStatus','正在创建真正的 ONNX WASM 推理会话…');
  state.session=await SCEPT_CORE.openModel(registry,manifest,raw,{wasmPaths:'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'});
  state.loaded=manifest;state.audit=null;$('export').disabled=true;
  $('run').disabled=false;
  note('modelStatus','已接入真实模型：'+state.session.description.inputs.length+' 个输入，'+state.session.description.outputs.length+' 个实际输出（含 '+probes.length+' 个选定观测点）。','success');
 }catch(e){state.session=null;note('modelStatus','模型接入失败：'+e.message,'error')}
 finally{$('load').disabled=false}
}
function buildInputs(){
 const input=state.session.description.inputs;
 const user=$('inputJson').value.trim();
 if(user.length>8000000)throw Error('JSON 输入超过 8 MB：请使用小型验证输入');
 if(user){
  const feeds=JSON.parse(user),keys=Object.keys(feeds);
  if(keys.length!==input.length||keys.some(n=>!input.find(m=>m.name===n)))throw Error('输入 JSON 的字段必须与实际模型输入完全一致');
  const all={};
  for(const meta of input){
   const v=feeds[meta.name],dims=v.dims,type=v.type||meta.type;
   if(!Array.isArray(v.data)||v.data.length>1e6)throw Error('用户输入数组必须存在且不超过 1,000,000 项');
   const klass={float32:Float32Array,float64:Float64Array,int32:Int32Array,int64:BigInt64Array,uint8:Uint8Array,bool:Uint8Array}[type];
   if(!klass)throw Error('尚不支持的输入类型 '+type);
   all[meta.name]={type,dims,data:type==='int64'?new klass(v.data.map(BigInt)):new klass(v.data)};
  }
  return{feeds:all,source:'user-provided-typed-JSON'};
 }
 const overrides=$('shapeJson').value.trim()?JSON.parse($('shapeJson').value):{};
 if(!overrides||Array.isArray(overrides)||typeof overrides!=='object')throw Error('动态维度 JSON 必须是输入名称到尺寸数组的映射');
 if(Object.keys(overrides).some(name=>!input.find(x=>x.name===name)))throw Error('动态维度包含模型中不存在的输入');
 const feeds={};
 for(const meta of input){
  const dims=overrides[meta.name]||meta.shape;
  if(!Array.isArray(dims)||dims.length!==meta.shape.length||
    !dims.every((d,i)=>Number.isSafeInteger(d)&&d>0&&(typeof meta.shape[i]!=='number'||d===meta.shape[i])))
    throw Error('请为 '+meta.name+' 提供合法的具体维度，并保持模型中的固定维度不变');
  const n=dims.reduce((a,b)=>a*b,1);
  if(!Number.isSafeInteger(n)||n>1e6)throw Error('工程合成输入超过 1,000,000 项；请使用模型要求的数据');
  const klass={float32:Float32Array,float64:Float64Array,int32:Int32Array,int64:BigInt64Array,uint8:Uint8Array,bool:Uint8Array}[meta.type];
  if(!klass)throw Error('尚不支持输入类型 '+meta.type);
  const arr=new klass(n);
  if(['float32','float64'].includes(meta.type))for(let i=0;i<n;i++)arr[i]=(i%37)/37;
  feeds[meta.name]={type:meta.type,dims:dims.slice(),data:arr};
 }
 return{feeds,source:'deterministic-synthetic-nonmedical'};
}
async function run(){
 if(!state.session)return;
 $('run').disabled=true;
 try{
  note('runStatus','准备真实 ONNX 输入，运行冻结模型前向计算 A…');
  const {feeds,source}=buildInputs();
  const prepared=dataAdapter.prepare(state.session.description,feeds);
  const amount=Number($('perturb').value);
  const {edited,intervention}=dataAdapter.offset(prepared,state.session.description,{delta:amount});
  const a=await state.session.run(prepared);
  note('runStatus','正在使用相同核心接口运行第二次真实冻结模型推理 B…');
  const b=await state.session.run(edited);
  // U4: one architecture-independent observation API for ViT/CNN/local ONNX.
  // Runtime tensor dimensions, not a baked-in [1,197,192], govern every metric.
  const profileId=$('architectureProfile').value;
  const observations=SCEPT_OBSERVATION.observePair({
   description:state.session.description,baseline:a,edited:b,
   probes:state.loaded.probes||[],profileId,maxAxisUnits:4096
  });
  const n=intervention.elementCount;
  state.audit={schema:'scept.core.u4.cross_architecture.v1',at:new Date().toISOString(),modelId:state.loaded.modelId,
   originalModelSHA256:state.session.sha256,adapter:state.loaded.adapterId,dataAdapter:dataAdapter.id,architectureProfile:profileId,inputs:state.session.description.inputs,
   inputSource:source,intervention,
   outputs:observations,limits:['No diagnosis','No validated explanation','No training','Only explicitly declared outputs']};
  const container=$('results');container.replaceChildren();
  for(const row of observations){
   const box=document.createElement('div');box.className='result';
   const title=document.createElement('b');title.textContent=row.tensor;
   const details=document.createElement('small');details.textContent=row.source+' · ['+row.shape.join('×')+'] · '+row.axes.map(a=>a.role+':'+a.size).join(' / ')+' · 变化 '+row.changed+'/'+row.total+' · 平均 |Δ| '+row.mean_abs_delta.toPrecision(5)+' · 最大 |Δ| '+row.max_abs_delta.toPrecision(5);
   const raw=document.createElement('code');raw.textContent='前 4 项：'+row.first12.slice(0,4).map(x=>x.A.toPrecision(4)+'→'+x.B.toPrecision(4)).join('，');
   box.append(title,details,raw);container.append(box);
  }
  $('export').disabled=false;
  note('runStatus','两次真正的冻结模型推理已完成。所有显示的数值来自 ONNX Runtime。','success');
 }catch(e){note('runStatus','实际推理失败：'+e.message,'error');$('export').disabled=true}
 finally{$('run').disabled=false}
}
$('modelPreset').onchange=()=>{$('architectureProfile').value=$('modelPreset').value==='vit'?'vit-token-channel':$('modelPreset').value==='cnn'?'cnn-nhwc':'generic';$('modelFile').disabled=$('modelPreset').value!=='local';$('inspect').disabled=false;$('load').disabled=true;$('run').disabled=true};
$('modelFile').disabled=true;
$('inspect').onclick=inspectModel;$('load').onclick=attach;$('run').onclick=run;
$('perturb').oninput=()=>{$('perturbLabel').textContent=(Number($('perturb').value)>=0?'+':'')+Number($('perturb').value).toFixed(2)};
$('export').onclick=()=>{
 if(!state.audit)return;
 const url=URL.createObjectURL(new Blob([JSON.stringify(state.audit,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='scept-u2-real-onnx-audit.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);
};
window.SCEPT_UNIVERSAL={registry,inspectModel,attach,run,getSnapshot:()=>({inspected:!!state.inspect,modelId:state.loaded?.modelId||null,ready:!!state.session,
 inputNames:state.session?.description.inputs.map(x=>x.name)||[],outputNames:state.session?.description.outputs.map(x=>x.name)||[],probes:state.loaded?.probes||[],profile:$('architectureProfile').value,hasExperiment:!!state.audit,hash:state.hash}),getAudit:()=>state.audit,observePair:SCEPT_OBSERVATION.observePair};

})();
