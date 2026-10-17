'use strict';
/* U0-U2 generic model workstation: no special casing model input names/shapes. */
(function(){
const $=id=>document.getElementById(id),registry=SCEPT_CORE.createRegistry();
const adapter=SCEPT_ONNX_ADAPTER.make({wasmPaths:'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'});
registry.register('model',adapter);
const dataAdapter=SCEPT_DATA_ADAPTER.make(SCEPT_CORE);
registry.register('data',dataAdapter);
const interventionEngine=SCEPT_INTERVENTION.make(registry,SCEPT_CORE,dataAdapter);
interventionEngine.registerBuiltins();
const observationAdapter=SCEPT_OBSERVATION.make();
registry.register('observation',observationAdapter);
const PRESETS={
 vit:{id:'pretrained-vit-tiny',url:'https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/resolve/ebffd7b9c0c53b51bada44116abd8f01aed41be6/onnx/model.onnx',sha256:'32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3'},
 cnn:{id:'mri-cnn-experimental',url:'https://huggingface.co/Udit013/brain-tumor-efficientnetb3/resolve/main/efficientnetb3.onnx',sha256:'320b1ef648f3da5216f6ee0374349324b3cb00a76bebf7dbd1c3796e6155991c'}
};
let state={bytes:null,hash:null,inspect:null,source:null,session:null,loaded:null,audit:null};
let scheduler=null;
function note(id,msg,kind='pending'){const el=$(id);el.textContent=msg;el.dataset.state=kind}
function tableCells(parent,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td)}parent.append(tr)}
function safeData(buf){if(buf.byteLength>160*1024*1024)throw Error('Model exceeds 160 MB local resource guard');return buf}
async function inspectModel(){
 if(scheduler?.snapshot().status==='running'){note('modelStatus','调度任务仍在执行，请先取消队列','error');return}
 scheduler=null;updateU6Controls();
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
 if(scheduler?.snapshot().status==='running')return;
 scheduler=null;updateU6Controls();
 if(!state.bytes||!state.inspect)return;
 $('load').disabled=true;$('run').disabled=true;
 try{
  const probes=$('probes').value.split(/[\r\n]+/).map(s=>s.trim()).filter(Boolean);
  if(probes.length>16)throw Error('最多同时选择 16 个真实张量');
  const raw=state.bytes,manifest={modelId:PRESETS[state.source]?.id||'user-uploaded-onnx',adapterId:'onnx-wasm',sha256:state.hash,probes};
  note('modelStatus','正在创建真正的 ONNX WASM 推理会话…');
  state.session=await SCEPT_CORE.openModel(registry,manifest,raw,{wasmPaths:'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'});
  state.loaded=manifest;state.audit=null;$('export').disabled=true;
  $('run').disabled=false;updateU6Controls();
  note('modelStatus','已接入真实模型：'+state.session.description.inputs.length+' 个输入，'+state.session.description.outputs.length+' 个实际输出（含 '+probes.length+' 个选定观测点）。','success');
 }catch(e){state.session=null;note('modelStatus','模型接入失败：'+e.message,'error')}
 finally{$('load').disabled=false}
}
function parseReferenceJson(raw){
 const meta=state.session.description.inputs,json=JSON.parse(raw);
 if(!json||Array.isArray(json)||typeof json!=='object')throw Error('参考输入必须是张量名称到类型/维度/数据的映射');
 const feeds={};
 for(const m of meta){
  const v=json[m.name];
  if(!v||!Array.isArray(v.data)||v.data.length>1e6)throw Error('参考输入缺少 '+m.name+'，或元素超过上限');
  const C={float32:Float32Array,float64:Float64Array,int32:Int32Array,int64:BigInt64Array,uint8:Uint8Array,bool:Uint8Array}[v.type||m.type];
  if(!C)throw Error('不支持的参考输入类型 '+m.name);
  feeds[m.name]={type:v.type||m.type,dims:v.dims,data:(v.type||m.type)==='int64'?new C(v.data.map(BigInt)):new C(v.data)};
 }
 return dataAdapter.prepare(state.session.description,feeds);
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
 $('run').disabled=true;state.audit=null;$('export').disabled=true;
 try{
  note('runStatus','准备真实 ONNX 输入，运行冻结模型前向计算 A…');
  const {feeds,source}=buildInputs();
  const prepared=dataAdapter.prepare(state.session.description,feeds);
  const mode=$('interventionMode').value;
  const amount=Number($('perturb').value);
  let experiment;
  if(mode==='legacy-offset'){
   const old=dataAdapter.offset(prepared,state.session.description,{delta:amount});
   experiment={baseline:prepared,edited:old.edited,control:null,intervention:{
    plugin:'legacy-offset',kind:'engineering-numeric-offset',...old.intervention}};
  }else{
   const config=JSON.parse($('interventionJson').value||'{}');
   const referenceFeeds=mode==='u5.replace'?parseReferenceJson($('referenceJson').value):null;
   experiment=await interventionEngine.execute({
    pluginId:mode,description:state.session.description,feeds:prepared,config,referenceFeeds});
  }
  const {baseline,edited,control,intervention}=experiment;
  const a=await state.session.run(baseline);
  note('runStatus','使用相同冻结模型运行实际干预 B'+(control?' 及匹配对照 C':'')+'…');
  const b=await state.session.run(edited);
  const c=control?await state.session.run(control):null;
  // U4: one architecture-independent observation API for ViT/CNN/local ONNX.
  // Runtime tensor dimensions, not a baked-in [1,197,192], govern every metric.
  const profileId=$('architectureProfile').value;
  const observations=observationAdapter.observePair({
   description:state.session.description,baseline:a,edited:b,
   probes:state.loaded.probes||[],profileId,maxAxisUnits:4096
  });
  const controls=c?observationAdapter.observePair({
   description:state.session.description,baseline:a,edited:c,probes:state.loaded.probes||[],profileId,maxAxisUnits:4096
  }):null;
  state.audit={schema:mode==='legacy-offset'?'scept.core.u4.cross_architecture.v1':'scept.core.u5.intervention.v1',at:new Date().toISOString(),modelId:state.loaded.modelId,
   originalModelSHA256:state.session.sha256,adapter:state.loaded.adapterId,dataAdapter:dataAdapter.id,observationAdapter:observationAdapter.id,architectureProfile:profileId,inputs:state.session.description.inputs,
   inputSource:source,intervention,
   outputs:observations,matchedControlOutputs:controls,limits:['No diagnosis','No validated explanation','No training','Only explicitly declared outputs']};
  const container=$('results');container.replaceChildren();
  for(const row of observations){
   const box=document.createElement('div');box.className='result';
   const title=document.createElement('b');title.textContent=row.tensor;
   const details=document.createElement('small');details.textContent=row.source+' · ['+row.shape.join('×')+'] · '+row.axes.map(a=>a.role+':'+a.size).join(' / ')+' · 变化 '+row.changed+'/'+row.total+' · 平均 |Δ| '+row.mean_abs_delta.toPrecision(5)+' · 最大 |Δ| '+row.max_abs_delta.toPrecision(5);
   const raw=document.createElement('code');raw.textContent='前 4 项：'+row.first12.slice(0,4).map(x=>x.A.toPrecision(4)+'→'+x.B.toPrecision(4)).join('，');
   box.append(title,details,raw);container.append(box);
  }
  if(controls){const box=document.createElement('div');box.className='result';
   box.textContent='匹配对照 C：'+controls.map(x=>x.tensor+' 平均 |C−A|='+x.mean_abs_delta.toPrecision(5)).join('；');
   container.append(box)}
  $('export').disabled=false;
  note('runStatus','两次真正的冻结模型推理已完成。所有显示的数值来自 ONNX Runtime。','success');
 }catch(e){note('runStatus','实际推理失败：'+e.message,'error');$('export').disabled=true}
 finally{$('run').disabled=false}
}
const U5_SAMPLES={
 'legacy-offset':{},'u5.noise':{sigma:0.1,seed:42},
 'u5.occlusion':{region:{bounds:[[0,1],[0,1],[0,1],[0,2]]},fill:0},
 'u5.replace':{region:{bounds:[[0,1],[0,1],[0,1],[0,2]]}},
 'u5.matched-control':{region:{bounds:[[0,1],[0,1],[0,1],[0,2]]},
   controlRegion:{bounds:[[0,1],[0,1],[1,2],[0,2]]},meanTolerance:1,fill:0},
 'u5.internal-mask':{maskInput:'scept_mask',indices:[0],value:0}
};
$('interventionMode').onchange=()=>{
 const mode=$('interventionMode').value;
 $('interventionJson').value=JSON.stringify(U5_SAMPLES[mode]||{},null,2);
 $('referenceJson').disabled=mode!=='u5.replace';
 $('perturb').disabled=mode!=='legacy-offset';
 $('interventionStatus').textContent=mode==='u5.internal-mask'?
  '要求 ONNX 图内存在已验证的 Mul 遮罩输入，并且连接至下游算子；不兼容时强制报错。':
  '示例 ROI 仅适用于 [1,1,4,4] 工程 CNN。其他模型需按实际输入维度指定区域。匹配对照需明确检查原始数据的适当协变量。';
};
$('interventionMode').onchange();
function u6StatusSnapshot(){return scheduler?.snapshot()||{status:'idle',total:0,tasks:[],pending:0,running:0,completed:0,failed:0,cancelled:0}}
function updateU6Controls(){
 const s=u6StatusSnapshot(),busy=s.status==='running',ready=!!state.session;
 $('u6Create').disabled=!ready||busy;
 $('u6Run').disabled=!ready||busy||!(s.pending>0||s.cancelled>0);
 $('u6Cancel').disabled=!busy;
 $('u6Retry').disabled=!ready||busy||!s.failed;
 $('u6Json').disabled=!s.total;$('u6Csv').disabled=!s.total;
 $('u6Facts').textContent=s.total?('总任务 '+s.total+' · 完成 '+s.completed+' · 失败 '+s.failed+
  ' · 待运行 '+s.pending+' · 已取消 '+s.cancelled+' · 当前状态 '+s.status):'未建立实验队列';
 const root=$('u6TaskRows');root.replaceChildren();
 for(const t of s.tasks){
  const tr=document.createElement('tr');
  const result=t.result?.outputs?.[0],detail=t.error?(t.error.code+' '+t.error.message):
   result?(result.tensor+': 平均 |Δ| '+result.mean_abs_delta.toPrecision(5)):'—';
  for(const value of [t.id+' / '+t.variant,t.pluginId,String(t.seed),t.status,detail]){
   const td=document.createElement('td');td.textContent=value;tr.append(td);
  }root.append(tr);
 }
}
function saveU6(content,name,type){
 const url=URL.createObjectURL(new Blob([content],{type}));
 const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),10000);
}
$('u6Mode').onchange=()=>{$('u6Parameter').disabled=$('u6Mode').value!=='sweep';
 $('u6Intensities').disabled=$('u6Mode').value!=='sweep';$('u6Controls').disabled=$('u6Mode').value!=='batch'};
$('u6Mode').onchange();
$('u6Plugin').onchange=()=>{$('u6Config').value=JSON.stringify(U5_SAMPLES[$('u6Plugin').value]||{},null,2);
 $('u6Parameter').value=$('u6Plugin').value==='u5.noise'?'sigma':
   $('u6Plugin').value==='u5.internal-mask'?'value':'fill'};
$('u6Plugin').onchange();
$('u6Create').onclick=async()=>{
 $('u6Create').disabled=true;
 try{
  if(!state.session)throw Error('模型尚未加载');
  const {feeds}=buildInputs(),baseSeed=Number($('u6Seed').value);
  const mode=$('u6Mode').value,config=JSON.parse($('u6Config').value||'{}');
  const spec={mode,pluginId:$('u6Plugin').value,config,baseSeed};
  if(mode==='sweep'){spec.parameter=$('u6Parameter').value;
   spec.intensities=JSON.parse($('u6Intensities').value)}
  if(mode==='batch')spec.controls=JSON.parse($('u6Controls').value);
  // Reference tensor comes from the same explicit U5 source; never infer it.
  const referenceFeeds=spec.pluginId==='u5.replace'||mode==='batch'&&spec.controls.some(x=>x.pluginId==='u5.replace')?
   parseReferenceJson($('referenceJson').value):null;
  scheduler=SCEPT_SCHEDULER.make({session:state.session,interventionEngine,
   observationAdapter,dataAdapter,core:SCEPT_CORE,onProgress:updateU6Controls});
  await scheduler.create(spec,{feeds,referenceFeeds,profileId:$('architectureProfile').value});
  note('u6Status','已建立 '+scheduler.snapshot().total+' 项任务；模型和输入指纹已记录。','success');
 }catch(e){scheduler=null;note('u6Status','建立失败：'+e.message,'error')}
 finally{updateU6Controls()}
};
$('u6Run').onclick=async()=>{
 if(!scheduler)return;
 $('u6Run').disabled=true;
 try{
  if(scheduler.snapshot().cancelled)scheduler.resume();
  note('u6Status','正在执行真实 ONNX 任务队列；将逐项记录结果、失败及种子…');
  const s=await scheduler.run();
  note('u6Status','批次结束：完成 '+s.completed+'/'+s.total+' · 失败 '+s.failed+
   ' · 取消 '+s.cancelled+' · 待执行 '+s.pending,s.failed?'error':'success');
 }catch(e){note('u6Status','调度失败：'+e.message,'error')}
 finally{updateU6Controls()}
};
$('u6Cancel').onclick=()=>{if(scheduler){scheduler.cancel();note('u6Status','已请求取消，当前正在执行的模型推理结束后生效')}};
$('u6Retry').onclick=()=>{
 try{scheduler?.retryFailed();note('u6Status','失败任务已恢复待运行，重试会复用原始种子和模型输入','success')}
 catch(e){note('u6Status','恢复失败：'+e.message,'error')}
 updateU6Controls();
};
$('u6Json').onclick=()=>{if(scheduler)saveU6(JSON.stringify(scheduler.exportAudit(),null,2),'scept-u6-audit.json','application/json')};
$('u6Csv').onclick=()=>{if(scheduler)saveU6(scheduler.exportCSV(),'scept-u6-results.csv','text/csv')};
updateU6Controls();
$('modelPreset').onchange=()=>{$('architectureProfile').value=$('modelPreset').value==='vit'?'vit-token-channel':$('modelPreset').value==='cnn'?'generic':'generic';$('modelFile').disabled=$('modelPreset').value!=='local';$('inspect').disabled=false;$('load').disabled=true;$('run').disabled=true};
$('modelFile').disabled=true;
$('inspect').onclick=inspectModel;$('load').onclick=attach;$('run').onclick=run;
$('perturb').oninput=()=>{$('perturbLabel').textContent=(Number($('perturb').value)>=0?'+':'')+Number($('perturb').value).toFixed(2)};
$('export').onclick=()=>{
 if(!state.audit)return;
 const url=URL.createObjectURL(new Blob([JSON.stringify(state.audit,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='scept-u5-real-onnx-audit.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);
};
window.SCEPT_UNIVERSAL={registry,inspectModel,attach,run,getSnapshot:()=>({inspected:!!state.inspect,modelId:state.loaded?.modelId||null,ready:!!state.session,
 inputNames:state.session?.description.inputs.map(x=>x.name)||[],outputNames:state.session?.description.outputs.map(x=>x.name)||[],probes:state.loaded?.probes||[],interventionMode:$('interventionMode').value,availableInterventions:interventionEngine.available().map(x=>x.id),profile:$('architectureProfile').value,hasExperiment:!!state.audit,hash:state.hash}),getAudit:()=>state.audit,getU6Snapshot:()=>u6StatusSnapshot(),createU6:()=>$('u6Create').click(),runU6:()=>$('u6Run').click(),interventionEngine,observePair:observationAdapter.observePair};

})();
