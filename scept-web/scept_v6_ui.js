'use strict';
/* V6 NAF public observatory. One forward/run, real source tensor only.
 * This file NEVER asks for A/B to render a primary fingerprint.
 */
(function(root){if(!root?.document)return;
root.document.addEventListener('DOMContentLoaded',()=>{
 const $=id=>document.getElementById(id),O=root.SCEPT_OBSERVATION,R=root.SCEPT_RUN_RECORD,
 C=root.SCEPT_CAPTURE_ONCE,N=root.SCEPT_NAF,K=root.SCEPT_NAF_CALIBRATION;
 if(!$('v6Run')||!O||!R||!C||!N||!K)return;
 let current=null,capture=null,history=[],reference=null,selected=null,busy=false;
 let selectedFeatureIndex=0,valuePage=0,detailValues=null;
 if(!root.SCEPT_CLOUD_ORBIT?.newCamera)throw Error('Real NAF 3D camera module not loaded');
 const archiveEngine=root.SCEPT_FULL_ARCHIVE;
 let archiving=false,lastArchiveInfo=null;
 const archiveSay=(v,error=false)=>{const e=$('v6ArchiveStatus');e.textContent=v;e.dataset.error=String(error)};
 const camera=root.SCEPT_CLOUD_ORBIT.newCamera({
  redraw:()=>renderCloud(),
  onTap:pos=>pickCloud(pos)
 });
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
 function probeLabel(g){
  const t=g.tensor||'unknown';
  const layer=t.match(/\/layer\.(\d+)\//);
  const kind=/\/value\//i.test(t)?'Attention Value':
    /\/output\/Add_output/i.test(t)?'Residual Stream':
    g.role==='channel'?'CNN Conv':'原始张量';
  return (layer?'第 '+Number(layer[1])+' 层 · ':'')+kind+' · '+t.split('/').slice(-2).join('/').slice(0,48);
 }
 function populateInspector(g,u){
  const layer=$('v6LayerSelect'),unit=$('v6UnitSelect');
  const gs=groups().filter(x=>x.status==='measured'&&x.units.length);
  layer.replaceChildren();
  if(!gs.length){layer.add(new Option('当前视图无可用观测张量',''));layer.disabled=true;
   unit.replaceChildren();unit.add(new Option('无可用位置',''));unit.disabled=true;return}
  layer.disabled=false;
  for(const candidate of gs)layer.add(new Option(probeLabel(candidate),candidate.tensor));
  layer.value=g?.tensor||gs[0].tensor;
  unit.replaceChildren();
  if(!g?.units.length){unit.add(new Option('无可用位置',''));unit.disabled=true;return}
  unit.disabled=false;
  for(const item of g.units){
   const desc=g.role==='token'?'Token':g.role==='channel'?'通道':'位置';
   unit.add(new Option(desc+' '+item.index,item.index.toString()));
  }
  unit.value=String(u?.index??g.units[0].index);
 }
 const fmtExact=x=>Number.isFinite(x)?Number(x).toPrecision(9):'—';
 const PAGE_SIZE=32;
 function selectUnit(tensor,index){
  const g=groups().find(g=>g.tensor===tensor&&g.status==='measured');
  if(!g||!g.units.some(u=>u.index===index))return;
  const changed=selected?.tensor!==tensor||selected?.index!==index;
  selected={tensor,index};
  if(changed){selectedFeatureIndex=0;valuePage=0}
  renderDetail();renderCloud();
 }
 function renderValueTable(){
  const host=$('v6ChannelRows');host.replaceChildren();
  const vals=detailValues;
  if(!vals?.length){
   const tr=document.createElement('tr'),cell=document.createElement('td');
   cell.colSpan=2;cell.textContent='原始数值不可评价或未被保留。';tr.append(cell);host.append(tr);
   $('v6ValuePage').textContent='—';
   $('v6ValuePrev').disabled=$('v6ValueNext').disabled=true;return;
  }
  const pages=Math.ceil(vals.length/PAGE_SIZE);
  valuePage=Math.max(0,Math.min(pages-1,valuePage));
  const begin=valuePage*PAGE_SIZE,end=Math.min(vals.length,begin+PAGE_SIZE);
  const chunk=document.createDocumentFragment();
  for(let i=begin;i<end;i++){
   const tr=document.createElement('tr');
   if(i===selectedFeatureIndex)tr.className='v6-value-selected';
   const idx=document.createElement('td'),v=document.createElement('td');
   idx.textContent=String(i);v.textContent=fmtExact(vals[i]);
   tr.append(idx,v);
   tr.addEventListener('click',()=>{selectedFeatureIndex=i;updateExact(true)});
   chunk.append(tr);
  }
  host.append(chunk);
  $('v6ValuePage').textContent='第 '+(valuePage+1)+' / '+pages+' 页 · 共 '+vals.length+' 个实测数值';
  $('v6ValuePrev').disabled=valuePage===0;
  $('v6ValueNext').disabled=valuePage===pages-1;
 }
 function updateExact(renderTable=true){
  const vals=detailValues,control=$('v6ChannelIndex');
  if(!vals?.length){control.disabled=true;control.max='0';
   $('v6ChannelExact').textContent='当前观测没有可读取的逐项原始数值。';
   if(renderTable)renderValueTable();return}
  selectedFeatureIndex=Math.max(0,Math.min(vals.length-1,Math.round(selectedFeatureIndex)));
  control.disabled=false;control.min='0';control.max=String(vals.length-1);
  control.value=String(selectedFeatureIndex);
  const p=currentGroup(),space=p?.role==='token'?'特征通道':'通道内空间位置 / 特征值';
  $('v6ChannelExact').textContent=space+' '+selectedFeatureIndex+
    ' = '+fmtExact(vals[selectedFeatureIndex])+' （实际记录的浮点值）';
  if(renderTable)renderValueTable();
  const cv=$('v6Channels'),layout=cv.__patternLayout;
  if(layout&&cv.getContext){
   // Redraw the plotted values with a precise highlighted index.
   drawPattern(vals,false);
  }
 }
 function renderDetail(){
  if(!current){
   show('v6Selected','尚未运行模型');
   show('v6InspectStats','未观测');detailValues=null;
   populateInspector(null,null);drawPattern(null);updateExact();return;
  }
  const gs=groups().filter(x=>x.status==='measured'&&x.units.length);
  if(!gs.some(g=>g.tensor===selected?.tensor&&g.units.some(u=>u.index===selected.index))){
   selected=gs.length?{tensor:gs[0].tensor,index:gs[0].units[0].index}:null;
   selectedFeatureIndex=0;valuePage=0;
  }
  const g=currentGroup(),u=g?.units.find(x=>x.index===selected?.index);
  populateInspector(g,u);
  if(!u){
   show('v6Selected','当前观测模式没有可评价的实际层 / Token。');
   show('v6InspectStats','未观测或不可评价');
   show('v6PatternNote','当前模型未提供此模式所需的真实中间张量。');
   detailValues=null;drawPattern(null);updateExact();return;
  }
  const actualMean=Number.isFinite(u.mean)?fmtExact(u.mean):'—';
  show('v6Selected',g.tensor+' / '+g.role+' '+selected.index+
    ' · RMS '+fmtExact(u.rms)+' · Mean '+actualMean+'。这不是解释重要性。');
  const head='RMS: '+fmtExact(u.rms)+' · Mean: '+actualMean+
    (Number.isFinite(u.count)?' · 实测数值数: '+u.count:'');
  show('v6InspectStats',head);
  if(g.view==='update'){
   show('v6InspectStats','单次残差更新 RMS: '+fmtExact(u.rms)+
    ' · 有符号平均更新: '+actualMean+' · 实际计算位置: '+selected.index);
   show('v6PatternNote','仅显示有真实前后模块张量支持的更新汇总。当前更新模式尚不提供逐通道差异表，不能用状态数值冒充更新数值。');
   detailValues=null;drawPattern(null);updateExact();return;
  }
  if(!capture){
   show('v6PatternNote','当前运行的原始张量已释放，精确数值不可评价；请重新执行真实模型。');
   detailValues=null;drawPattern(null);updateExact();return;
  }
  try{
   const p=N.getPattern(capture,g.tensor,selected.index,current.profile,{maxValues:65536});
   detailValues=p.values;
   selectedFeatureIndex=Math.max(0,Math.min(p.values.length-1,selectedFeatureIndex));
   const low=p.values.reduce((v,x)=>Math.min(v,x),Infinity),
     high=p.values.reduce((v,x)=>Math.max(v,x),-Infinity);
   show('v6InspectStats',head+' · Min: '+fmtExact(low)+' · Max: '+fmtExact(high));
   const unitMeaning=g.role==='token'?'特征通道':'通道内实际空间特征值';
   show('v6PatternNote',p.values.length+' 个真实测量的'+unitMeaning+
    ' · '+JSON.stringify(p.coordinates)+'。同样的 RMS 可能对应不同的通道模式。');
   drawPattern(p.values);updateExact();
  }catch(e){
   detailValues=null;
   show('v6PatternNote','真实通道不可评价：'+e.message);
   drawPattern(null);updateExact();
  }
 }
 function drawPattern(vals){
  const cv=$('v6Channels'),ctx=cv.getContext('2d');
  const dpr=Math.min(2,devicePixelRatio||1),w=Math.max(460,Math.round(cv.getBoundingClientRect().width||720)),h=178;
  cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle='#071723';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#43636b';ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  cv.__patternLayout=null;
  if(!vals||!vals.length)return;
  let max=1e-10;for(const v of vals)max=Math.max(max,Math.abs(v));
  const bars=Math.min(vals.length,640),step=Math.ceil(vals.length/bars),size=w/bars;
  cv.__patternLayout={bars,step,count:vals.length,width:w,height:h};
  for(let i=0;i<bars;i++){
   const v=vals[i*step],height=Math.abs(v)/max*(h/2-12);
   ctx.fillStyle=v>=0?'#5cd2bb':'#eca769';
   ctx.fillRect(i*size,v>=0?h/2-height:h/2,Math.max(1,size*.76),Math.max(.5,height));
  }
  const mark=Math.floor(selectedFeatureIndex/step);
  ctx.strokeStyle='#ffefa4';ctx.lineWidth=2;
  ctx.strokeRect(mark*size,h/2-83,Math.max(2,size),166);
  ctx.font='10px system-ui';ctx.fillStyle='#a1cbd0';ctx.fillText(
   vals.length>bars?'仅下采样显示实测值（每 '+step+' 项抽样）：共 '+vals.length:
   '实测值 · 正负数有含义 · 黄色框为当前选中索引',8,h-8);
 }
 $('v6LayerSelect').onchange=e=>{
  const g=groups().find(g=>g.tensor===e.target.value&&g.status==='measured');
  if(g?.units.length)selectUnit(g.tensor,g.units[0].index);
 };
 $('v6UnitSelect').onchange=e=>{
  const g=groups().find(g=>g.tensor===$('v6LayerSelect').value);
  const n=Number(e.target.value);
  if(g?.units.some(u=>u.index===n))selectUnit(g.tensor,n);
 };
 $('v6ChannelIndex').onchange=e=>{
  if(!detailValues?.length)return;
  const n=Number(e.target.value);
  if(Number.isSafeInteger(n)&&n>=0&&n<detailValues.length){
   selectedFeatureIndex=n;valuePage=Math.floor(n/PAGE_SIZE);updateExact();
  }else updateExact();
 };
 $('v6Channels').onclick=e=>{
  const layout=e.currentTarget.__patternLayout;
  if(!layout||!detailValues?.length)return;
  const rect=e.currentTarget.getBoundingClientRect();
  const x=(e.clientX-rect.left)/rect.width*layout.width;
  const index=Math.floor(Math.max(0,Math.min(layout.bars-1,
   Math.floor(x/(layout.width/layout.bars))))*layout.step);
  selectedFeatureIndex=Math.min(detailValues.length-1,index);
  valuePage=Math.floor(selectedFeatureIndex/PAGE_SIZE);updateExact();
 };
 $('v6ValuePrev').onclick=()=>{valuePage--;renderValueTable()};
 $('v6ValueNext').onclick=()=>{valuePage++;renderValueTable()};
 function renderCloud(){
  const cv=$('v6Cloud'),ctx=cv.getContext('2d'),dpr=Math.min(2,devicePixelRatio||1);
  const w=Math.max(440,Math.round(cv.getBoundingClientRect().width||670)),h=300;
  cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle='#061923';ctx.fillRect(0,0,w,h);
  const gs=groups().filter(g=>g.status==='measured'&&g.units.length);
  if(!gs.length){cv.__points=[];return}
  const points=[];
  gs.forEach((g,layer)=>{
   // Measured model units; their spatial arrangement here is a 3D VIEW ONLY.
   // No model topology, anatomical coordinates or learned edges are invented.
   const shown=Math.min(g.units.length,400),columns=20,rows=Math.ceil(shown/columns);
   for(let unit=0;unit<shown;unit++){
    const u=g.units[unit],
      x0=(unit%columns-(columns-1)/2)*14,
      y0=(Math.floor(unit/columns)-(rows-1)/2)*14,
      z0=(layer-(gs.length-1)/2)*44,
      projected=camera.project(x0,y0,z0,w,h),
      v=value(g,u),isSelected=selected?.tensor===g.tensor&&selected?.index===u.index;
    points.push({...projected,r:isSelected?Math.min(8,Math.max(3.4,4.4*projected.scale)):
      Math.min(5.5,Math.max(.85,1.85*projected.scale)),v,isSelected,
      tensor:g.tensor,index:u.index});
   }
  });
  // Painter's order: distant units first; click targets prefer visible ones.
  points.sort((a,b)=>b.depth-a.depth);
  for(const p of points){
   ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
   ctx.fillStyle=p.isSelected?'#fff1a6':color(p.v);ctx.fill();
   if(p.isSelected){ctx.strokeStyle='#fff1a6';ctx.lineWidth=1.5;ctx.stroke()}
  }
  ctx.font='11px system-ui';ctx.fillStyle='#a9cbd2';
  ctx.fillText('拖动旋转 · 滚轮缩放 · Shift/右键拖动平移 · 仅展示真实观测单位',8,h-10);
  cv.__points=points;
 }
 function pickCloud(pos){
  const cv=$('v6Cloud'),pts=cv.__points||[];
  // Click after a true pointer tap only, never after orbit/pan/zoom gestures.
  // Front-most points win overlapping screen-space hit tests.
  const nearest=pts.filter(p=>Math.hypot(p.x-pos.x,p.y-pos.y)<Math.max(8,p.r+5))
    .sort((a,b)=>a.depth-b.depth)[0];
  if(nearest){
   selected={tensor:nearest.tensor,index:nearest.index};
   renderDetail();renderCloud();
  }
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
 // Pointer captures are coordinated in a single camera; drag cannot select.
 camera.bind($('v6Cloud'));
 $('v6Rotate').onclick=()=>camera.nudgeYaw(Math.PI/7);
 $('v6ZoomIn').onclick=()=>camera.zoomBy(1.25);
 $('v6ZoomOut').onclick=()=>camera.zoomBy(.8);
 $('v6Reset').onclick=()=>camera.reset();
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
   retainInputs:$('v6RetainInput').checked,
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
   retainInputs:$('v6RetainInput').checked,
   probes,profileId:ctx.profile,
   preprocessing:{id:'scept-mri-v3-cnn-native-axial',config:{
      inputSize:256,channelMode:'NHWC-RGB-gray-duplicate',normalizer:'v3-mri-cnn-source-quantiles'}},
   runtime:{engine:'onnxruntime-web',version:'1.20.1',provider:'wasm'}});
  return{cap,profile:ctx.profile,probes,updates:[],
   note:'MRI CNN 作者报告数据泄漏；本次仅观测模型内部张量/输出，不是临床诊断'};
 }
 async function run(){
  if(busy||archiving)return;busy=true;$('v6Run').disabled=true;
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
   $('v6Export').disabled=false;$('v6ExportFull').disabled=false;
   $('v6BuildReference').disabled=history.length<6;
   lastArchiveInfo=null;
   archiveSay('可下载本次全部已捕获张量（'+next.listStored().length+' 个）。'+
    (next.listStoredInputs().length?'已保留模型输入':'模型输入未保留')+
    '；开始下一次运行之前请保存 ZIP。');
   renderFingerprint();
   show('v6ModelInfo',result.note+
     (mode()==='cnn'?' · 实际隐藏层：'+result.probes.filter(p=>p!=='dense_2').length:''));
   show('v6History',history.length+' 个独立 Run Record，最近：'+current.runId+
     '；旧运行的完整张量不再缓存在内存中，仅保留摘要。');
   say('NAF 已完成：仅执行了一次真正的模型前向推理；所有颜色来自真实内部数值。');
   if($('v6AutoArchive').checked)await exportFull(true);
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
 async function exportFull(auto=false){
  if(archiving)return;
  if(!capture||!current){archiveSay('请先运行一次真实观测。',true);return}
  const wasRunDisabled=$('v6Run').disabled;
  archiving=true;$('v6Run').disabled=true;$('v6ExportFull').disabled=true;
  try{
   archiveSay('校验真实张量并创建 ZIP 中…');
   const result=await archiveEngine.build({
    capture,naf:current,reference,
    includeInputs:$('v6RetainInput').checked,tensorSha:C.tensorSha
   });
   const url=URL.createObjectURL(result.blob),a=document.createElement('a');
   a.href=url;a.download=result.suggestedFilename;document.body.appendChild(a);a.click();a.remove();
   setTimeout(()=>URL.revokeObjectURL(url),10000);
   lastArchiveInfo={runId:current.runId,bytes:result.size,outputs:result.outputTensorCount,
    inputs:result.includesInputs,files:result.fileCount};
   archiveSay((auto?'自动下载已开始：':'下载已开始：')+
    result.outputTensorCount+' 个实际测量输出张量；大小 '+
    (result.size/1048576).toFixed(2)+' MiB；'+
    (result.includesInputs?'包含实际模型输入。':'不包含模型输入。'));
  }catch(e){archiveSay('完整归档失败：'+e.message,true)}
  finally{archiving=false;$('v6ExportFull').disabled=false;$('v6Run').disabled=wasRunDisabled}
 }
 $('v6ExportFull').onclick=()=>exportFull(false);
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
  getSelectedPattern:()=>{const g=currentGroup();return capture&&selected&&g?.view==='state'?
   N.getPattern(capture,selected.tensor,selected.index,current.profile,{maxValues:65536}):null},
  getCloudView:()=>camera.snapshot(),
  getArchiveStatus:()=>lastArchiveInfo,
  buildFullArchive:()=>archiveEngine.build({capture,naf:current,reference,
   includeInputs:$('v6RetainInput').checked,tensorSha:C.tensorSha}),
  onRunCapture:(fn)=>{if(typeof fn!=='function')throw Error('run listener must be a function');
   runListeners.add(fn);return()=>runListeners.delete(fn)},
  runOnce:run};
 say('准备就绪：等待实际预训练模型与 MRI。基础 NAF 只需要一次运行。');
});
})(typeof window==='undefined'?null:window);
