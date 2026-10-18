'use strict';
/* V6 NAF public observatory. One forward/run, real source tensor only.
 * This file NEVER asks for A/B to render a primary fingerprint.
 */
(function(root){if(!root?.document)return;
root.document.addEventListener('DOMContentLoaded',()=>{
 const $=id=>document.getElementById(id),O=root.SCEPT_OBSERVATION,R=root.SCEPT_RUN_RECORD,
 C=root.SCEPT_CAPTURE_ONCE,N=root.SCEPT_NAF,K=root.SCEPT_NAF_CALIBRATION;
 if(!$('v6Run')||!O||!R||!C||!N||!K)return;
 let current=null,capture=null,history=[],reference=null,selected=null,busy=false,angle=.48;
 const runListeners=new Set();
 const stats={runs:0};
 const fmt=x=>Number.isFinite(x)?Number(x).toPrecision(5):'—';
 const say=(s,error=false)=>{const e=$('v6Status');e.textContent=s;e.dataset.error=String(error)};
 const mode=()=>$('v6Model').value;
 const color=value=>{
  if(value===null||!Number.isFinite(value))return '#293e4a';
  const v=Math.min(1,Math.max(0,value));
  return v<=.33?'#356b83':v<=.68?'#53cabb':'#eeab61';
 };
 const show=($id,v)=>{const element=$($id);if(element)element.textContent=v};
 function groups(){
  if(!current)return[];
  if($('v6Metric').value==='update'){
   return current.updates.map(u=>({status:u.status,tensor:u.outputTensor,role:u.role||'token',
    units:u.units.map(x=>({index:x.index,rms:x.rmsUpdate,mean:x.meanSignedUpdate})),
    reason:u.reason,view:'update'}));
  }
  return current.groups.filter(g=>g.role&&g.status==='measured').map(g=>({...g,view:'state'}));
 }
 function value(g,u){
  if($('v6Scale').value==='reference'){
   if(!reference)return null;
   const k=g.tensor+'\u0001'+u.index;
   return K.empirical(u.rms,reference.distributions[k]);
  }
  const m=Number($('v6RawMax').value);
  return K.scale(u.rms,{mode:'raw',rawMax:m});
 }
 function renderFingerprint(){
  const cv=$('v6Fingerprint'),ctx=cv.getContext('2d'),dpr=Math.min(2,devicePixelRatio||1),
  w=Math.max(520,Math.round(cv.getBoundingClientRect().width||800));
  const gs=groups(),h=Math.max(185,55+gs.length*43);
  cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle='#071723';ctx.fillRect(0,0,w,h);
  ctx.fillStyle='#d4e8ec';ctx.font='12px system-ui';
  ctx.fillText($('v6Metric').value==='update'?'真实模块表示更新':'真实单次活动幅度 RMS',95,19);
  const left=125,right=w-18,top=34,rowHeight=43;
  gs.forEach((g,row)=>{
   const y=top+row*rowHeight;
   ctx.fillStyle='#bddbd9';
   const label=g.tensor.split('/').slice(-3).join('/').slice(0,18);
   ctx.fillText(label||g.tensor.slice(0,18),5,y+16);
   if(g.status!=='measured'||!g.units.length){
    ctx.fillStyle='#81979e';ctx.fillText('未观测 / 不可评价',left,y+16);
    return;
   }
   const width=(right-left)/g.units.length;
   for(const u of g.units){
    const x=left+u.index*width,raw=value(g,u);
    ctx.fillStyle=color(raw);
    ctx.fillRect(x,y,Math.max(.65,width*.93),22);
   }
  });
  ctx.font='10px system-ui';ctx.fillStyle='#8eafb5';
  ctx.fillText($('v6Scale').value==='reference'?
    '校准：同源实测参考样本经验秩，不是显著性或病理概率':
    'RAW：全图同一固定 RMS 上限；绝不每次自动拉满色标',left,h-16);
  cv.__layout={left,right,top,rowHeight,gs,dpr};
  show('v6Counts',current?
    '单次运行 '+current.runId.slice(-10)+' · '+current.groups.filter(g=>g.status==='measured').length+
    ' 个已测量张量 · '+current.groups.filter(g=>g.status==='measured')
       .reduce((n,g)=>n+g.units.length,0)+' 个原生观测单元':
    '尚未运行。点击一次观察，不需要 A/B 对照。');
  if(!selected&&gs.length&&gs[0].units.length)selected={tensor:gs[0].tensor,index:0};
  renderDetail();renderCloud();
 }
 function currentGroup(){return groups().find(g=>g.tensor===selected?.tensor)}
 function renderDetail(){
  if(!current||!selected)return;
  const g=currentGroup(),u=g?.units.find(x=>x.index===selected.index);
  if(!u){show('v6Selected','当前视图没有这个观测位置的真实数据。');drawPattern(null);return}
  show('v6Selected',g.tensor+' / '+g.role+' '+selected.index+' · 实际 RMS '+fmt(u.rms)+
    ' · Mean '+fmt(u.mean)+'。这不是解释重要性。');
  if(g.view==='update'){show('v6PatternNote','更新模式显示同一次前向传播中 L5→L6 真实残差表示变化；详细通道更新将在后续专用视图提供。');drawPattern(null);return}
  if(!capture){show('v6PatternNote','原始通道张量已释放；如需再看详情，请重新运行该输入。');drawPattern(null);return}
  try{
   const p=N.getPattern(capture,g.tensor,selected.index,current.profile,{maxValues:65536});
   show('v6PatternNote',p.values.length+' 个真实数值 · '+JSON.stringify(p.coordinates)+
     '；同样大小的 RMS 也可能对应完全不同的通道模式。');
   drawPattern(p.values);
  }catch(e){show('v6PatternNote','真实通道不可评价：'+e.message);drawPattern(null)}
 }
 function drawPattern(vals){
  const cv=$('v6Channels'),ctx=cv.getContext('2d');
  const dpr=Math.min(2,devicePixelRatio||1),w=Math.max(460,Math.round(cv.getBoundingClientRect().width||720)),h=178;
  cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle='#071723';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#43636b';ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  if(!vals||!vals.length)return;
  const max=Math.max(1e-10,...vals.map(Math.abs)),bars=Math.min(vals.length,640),step=Math.ceil(vals.length/bars),size=w/bars;
  for(let i=0;i<bars;i++){
   const v=vals[i*step],height=Math.abs(v)/max*(h/2-12);
   ctx.fillStyle=v>=0?'#5cd2bb':'#eca769';
   ctx.fillRect(i*size,v>=0?h/2-height:h/2,Math.max(1,size*.76),Math.max(.5,height));
  }
  ctx.font='10px system-ui';ctx.fillStyle='#a1cbd0';ctx.fillText(
    vals.length>bars?'仅下采样显示真实值（每 '+step+' 个取第一个）：共 '+vals.length:'真实通道数值 · 正负均有含义；±值色标按选中向量缩放',
    8,h-8);
 }
 function renderCloud(){
  const cv=$('v6Cloud'),ctx=cv.getContext('2d'),dpr=Math.min(2,devicePixelRatio||1);
  const w=Math.max(440,Math.round(cv.getBoundingClientRect().width||670)),h=250;
  cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle='#061923';ctx.fillRect(0,0,w,h);
  const gs=groups().filter(g=>g.status==='measured'&&g.units.length);if(!gs.length)return;
  const points=[];
  gs.forEach((g,layer)=>{
   // Geometric layout is explicitly illustrative, never a learned topology.
   const shown=Math.min(g.units.length,400);
   for(let unit=0;unit<shown;unit++){
    const u=g.units[unit],z=layer-(gs.length-1)/2,x0=(unit%20-9.5)*13,y0=(Math.floor(unit/20)-5)*11,
      x1=x0*Math.cos(angle)-z*25*Math.sin(angle),
      z1=x0*Math.sin(angle)+z*25*Math.cos(angle),scale=1/(1+z1/550),
      x=w/2+x1*scale,y=h/2+(y0*scale),v=value(g,u),isSelected=selected?.tensor===g.tensor&&selected?.index===u.index;
    points.push({x,y,r:isSelected?5:Math.max(1.4,2.1*scale),v,isSelected,tensor:g.tensor,index:u.index});
   }
  });
  points.sort((a,b)=>a.r-b.r);
  for(const p of points){
   ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=p.isSelected?'#fff1a6':color(p.v);ctx.fill();
   if(p.isSelected){ctx.strokeStyle='#fff1a6';ctx.lineWidth=1.5;ctx.stroke()}
  }
  ctx.font='11px system-ui';ctx.fillStyle='#a9cbd2';
  ctx.fillText('真实计算位置 · 3D 排布仅为视图，点间没有虚构连线',8,h-10);
  cv.__points=points;
 }
 $('v6Fingerprint').onclick=e=>{
  const cv=e.currentTarget,l=cv.__layout;if(!l)return;
  const b=cv.getBoundingClientRect(),x=(e.clientX-b.left)/b.width*cv.width/l.dpr,
   y=(e.clientY-b.top)/b.height*cv.height/l.dpr,
   row=Math.floor((y-l.top)/l.rowHeight),g=l.gs[row];if(!g||g.status!=='measured')return;
  const unit=Math.floor((x-l.left)/(l.right-l.left)*g.units.length);
  if(unit<0||unit>=g.units.length)return;
  selected={tensor:g.tensor,index:unit};renderDetail();renderCloud();
 };
 $('v6Cloud').onclick=e=>{
  const cv=e.currentTarget,b=cv.getBoundingClientRect(),x=(e.clientX-b.left)*cv.width/b.width/Math.min(2,devicePixelRatio||1),
   y=(e.clientY-b.top)*cv.height/b.height/Math.min(2,devicePixelRatio||1);
  const pts=cv.__points||[],p=pts.map(v=>({...v,dist:Math.hypot(v.x-x,v.y-y)})).sort((a,b)=>a.dist-b.dist)[0];
  if(p&&p.dist<14){selected={tensor:p.tensor,index:p.index};renderDetail();renderCloud()}
 };
 $('v6Rotate').onclick=()=>{angle+=Math.PI/7;renderCloud()};
 function fingerprintFrom(cap,profile,probes,updates=[]){return N.create(cap,{profile,probes,updates})}
 async function captureViT(){
  if(!root.SCEPT_V3?.getSnapshot()?.ready||!root.SCEPT_MRI.state.loaded)
   throw Error('正在等待真实 ViT 以及 MRI 输入载入');
  const ctx=root.SCEPT_V3.getScientificContext(),names=ctx.probes.map(p=>p.name);
  const from='/vit/encoder/layer.5/output/Add_output_0',
        to='/vit/encoder/layer.6/output/Add_output_0',
   valid=ctx.extraProbes?.some(p=>p.name===from)&&ctx.extraProbes?.some(p=>p.name===to);
  if(valid)names.push(from,to);
  const graph=ctx.graph,description={inputs:[{name:'pixel_values',type:'float32',shape:[1,3,224,224]}],
   outputs:[{name:'logits',type:'float32',shape:[1,null]},
    ...names.map(name=>{
     const m=graph.outputMetadata[name];
     if(!m||m.type!=='float32'||!Array.isArray(m.shape))throw Error('模型没有真实导出的观测张量 '+name);
     return{name,type:m.type,shape:m.shape.map(d=>Number.isSafeInteger(d)&&d>0?d:null)};
    })]};
  const session={sha256:ctx.sha256,description,run:async feeds=>{
   const outputs=await ctx.session.run({pixel_values:new root.ort.Tensor('float32',
      feeds.pixel_values.data,feeds.pixel_values.dims)});
   const selected={logits:outputs.logits};
   for(const n of names){if(!outputs[n])throw Error('Missing actual ViT tensor '+n);selected[n]=outputs[n]}
   return selected;
  }};
  const X=root.SCEPT_MRI.makeSlice(false,2);
  const engine=C.make({observation:O,contract:R});
  const cap=await engine.captureOnce({session,
   feeds:{pixel_values:{type:'float32',dims:[1,3,224,224],data:X}},
   modelId:'public-sha-verified-ViT-Tiny-ImageNet',
   profileId:'vit-token-channel',probes:names,
   preprocessing:{id:'scept-mri-v3-native-axial',config:{input:'native-current-MRI-axial-slice',
     channelMode:'grayscale-replicated-3',inputSize:224,normalizer:'v3-mri-fixed-source-quantiles'}},
   runtime:{engine:'onnxruntime-web',version:'1.20.1',provider:'wasm'}});
  const updates=[{inputTensor:from,outputTensor:to,graph,
   coordinateSemantics:'same-residual-stream',
   declaredBy:'pinned-vit-tiny-transformer-adapter-layer5-to-layer6',
   verifiedIdentity:valid}];
  return{cap,profile:'vit-token-channel',probes:names,updates,
    note:'ViT 预训练于普通图片；MRI 输入仅用于模型内部数值观测，不是临床模型'};
 }
 async function captureCNN(){
  if(!root.SCEPT_MRI?.state.loaded)throw Error('MRI 未加载');
  const ctx=await root.SCEPT_V6_CNN.load(),X=root.SCEPT_MRI.makeCnnInput(false),
   probes=ctx.selected.length?ctx.selected:['dense_2'];
  const engine=C.make({observation:O,contract:R});
  const cap=await engine.captureOnce({session:ctx,
   feeds:{input:{type:'float32',dims:[1,256,256,3],data:X}},
   modelId:'Udit013-EfficientNetB3-brain-tumor-leakage-reported',
   probes,profileId:ctx.profile,
   preprocessing:{id:'scept-mri-v3-cnn-native-axial',config:{
      inputSize:256,channelMode:'NHWC-RGB-gray-duplicate',normalizer:'v3-mri-cnn-source-quantiles'}},
   runtime:{engine:'onnxruntime-web',version:'1.20.1',provider:'wasm'}});
  return{cap,profile:ctx.profile,probes,updates:[],
   note:'MRI CNN 作者报告数据泄漏；本次仅观测模型内部张量/输出，不是临床诊断'};
 }
 async function run(){
  if(busy)return;busy=true;$('v6Run').disabled=true;
  try{
   say('正在执行单次真实模型前向传播（无 A/B 比较）…');
   const result=mode()==='vit'?await captureViT():await captureCNN(),
    next=result.cap,fingerprint=fingerprintFrom(next,result.profile,result.probes,result.updates);
   if(capture)capture.release();capture=next;current=fingerprint;selected=null;
   history.push(fingerprint);if(history.length>32)history.shift();
   const complete={record:next.exportRecord(),naf:JSON.parse(JSON.stringify(fingerprint))};
   for(const fn of runListeners){try{fn(complete)}catch(err){console.warn('V6.5 journal subscriber failed',err)}}
   reference=null;$('v6Scale').value='raw';
   stats.runs++;
   $('v6Export').disabled=false;$('v6BuildReference').disabled=history.length<6;
   renderFingerprint();
   show('v6ModelInfo',result.note+
     (mode()==='cnn'?' · 实际隐藏层：'+result.probes.filter(p=>p!=='dense_2').length:''));
   show('v6History',history.length+' 个独立 Run Record，最近：'+current.runId+
     '；旧运行的完整张量不再缓存在内存中，仅保留摘要。');
   say('NAF 已完成：仅执行了一次真正的模型前向推理；所有颜色来自真实内部数值。');
  }catch(e){say('NAF 不可评价：'+e.message,true)}
  finally{busy=false;$('v6Run').disabled=false}
 }
 $('v6Run').onclick=run;
 for(const id of ['v6Metric','v6Scale','v6RawMax'])$(id).onchange=()=>{
  if($('v6Scale').value==='reference'&&!reference){
   $('v6Scale').value='raw';say('需要至少5个实际不同输入、同一模型和预处理的记录，先创建参考分布。');
  }
  selected=null;renderFingerprint();
 };
 $('v6Model').onchange=()=>{say('已选择模型：'+mode()+'，点击「单次运行并观察」以采集真实活动。')};
 $('v6BuildReference').onclick=()=>{
  try{
   if(!current)throw Error('先采集真实 NAF');
   const candidates=history.filter(x=>x.runId!==current.runId&&x.profile===current.profile&&
      x.provenance.model.sha256===current.provenance.model.sha256&&
      x.provenance.preprocessing.sha256===current.provenance.preprocessing.sha256);
   reference=K.buildReference(candidates,{
    label:'Browser-run MRI slices (not independent patients)',
    domain:'public-MNI-template-or-user-provided-MRI-axial-image',
    kind:'within-volume'});
   $('v6Scale').value='reference';selected=null;renderFingerprint();
   say('已生成同模型经验参考（'+reference.nSamples+' 次真实运行）；并非独立患者参考或医学异常值。');
  }catch(e){reference=null;say('参考校准不可评价：'+e.message,true)}
 };
 $('v6Export').onclick=()=>{
  if(!capture||!current)return;
  const out={record:capture.exportRecord(),naf:current,
   calibration:reference?{schema:reference.schema,kind:reference.kind,domain:reference.domain,
      label:reference.label,nSamples:reference.nSamples}:null};
  const url=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='scept-v6-single-run-naf.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),3000);
 };
 root.SCEPT_V6={getSnapshot:()=>({hasNaf:!!current,runCount:stats.runs,hasReference:!!reference,
   groups:current?.groups.map(g=>({tensor:g.tensor,status:g.status,role:g.role,units:g.units.length})),
   updates:current?.updates.map(g=>({status:g.status,units:g.units.length,reason:g.reason||null})),
   architecture:current?.profile,runId:current?.runId,recordCount:history.length}),
  getNaf:()=>current,getReference:()=>reference,getRecord:()=>capture?.exportRecord()||null,
  onRunCapture:(fn)=>{if(typeof fn!=='function')throw Error('run listener must be a function');
   runListeners.add(fn);return()=>runListeners.delete(fn)},
  runOnce:run};
 say('准备就绪：等待实际预训练模型与 MRI。基础 NAF 只需要一次运行。');
});
})(typeof window==='undefined'?null:window);
