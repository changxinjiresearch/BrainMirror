'use strict';
/* SCEPT U8 universal visual research studio.
 * Canvas perspective is only a visual graph layout, NOT literal AI neurons.
 * All colored values come from genuinely observed ONNX outputs / paired feeds.
 */
(function(){
 const api=window.SCEPT_VISUAL_DATA;
 const $=id=>document.getElementById(id);
 if(!api||!$('u8Graph'))return;
 const state={session:null,source:'u5',mode:'compare',scene:null,feed:null,
  captures:new Map(),audit:null,u6:null,u7:null,graphBase:null,
  yaw:-.3,pitch:.24,zoom:1,selection:null,selectionTask:null,lastError:null};
 const num=x=>Number.isFinite(x)?x.toPrecision(4):'—';
 const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
 function notice(s,type='pending'){$('u8Status').textContent=s;$('u8Status').dataset.state=type}
 function clear(){state.scene=null;state.feed=null;state.captures.clear();state.selection=null;
  state.audit=null;state.u6=null;state.u7=null;state.selectionTask=null;
  $('u8Task').replaceChildren();$('u8Tensor').replaceChildren();
  $('u8Task').disabled=true;$('u8Tensor').disabled=true;
  state.graphBase=state.session?api.createScene:null;
  paint();
 }
 function attach(session){
  state.session=session;clear();
  if(session){
   state.graphBase=api.graphView(session.description,[]);
   notice('模型已加载：计算图可导航；真实激活颜色将在实验执行后出现。','success');
  }else{state.graphBase=null;notice('请先加载模型。未生成任何演示/模拟模型激活。')}
  paint();
 }
 function layout(){return $('u8InputLayout').value}
 function inputName(){return $('u8InputName').value.trim()||state.session?.description.inputs?.[0]?.name||''}
 function capture(source,payload){
  if(!state.session||payload.modelSha256!==state.session.sha256)return;
  try{
   const l=layout(),name=inputName();
   if(l==='off')return;
   const volume=api.captureInput({baseline:payload.baseline,edited:payload.edited,
    inputName:name,layout:l,source:'actual-model-feed-'+source,taskId:payload.taskId,maxSide:72});
   const key=source+':'+payload.taskId;
   state.captures.delete(key);state.captures.set(key,volume);
   while(state.captures.size>5)state.captures.delete(state.captures.keys().next().value);
   if(state.source===source&&(state.selectionTask===payload.taskId||!state.selectionTask)){
    state.feed=volume;paintVolume();
   }
  }catch(e){state.lastError=e.message;paintVolume()}
 }
 function onAB(experiment){
  if(!state.session)return;
  capture('u5',{...experiment,taskId:'T001',modelSha256:state.session.sha256});
 }
 function snapshots({audit=null,u6=null,u7=null}={}){
  state.audit=audit;state.u6=u6;state.u7=u7;
  refresh();
 }
 function getTasks(){
  if(state.source==='u5')return state.audit?.outputs?[{id:'T001',status:'completed'}]:[];
  if(state.source==='u6')return state.u6?.tasks||[];
  return state.u7?.tasks||[];
 }
 function rebuildTaskOptions(){
  const select=$('u8Task'),tasks=getTasks(),current=select.value;
  select.replaceChildren();
  for(const t of tasks){
   const option=document.createElement('option');option.value=t.id;option.textContent=t.id+' · '+t.status;
   select.append(option);
  }
  const chosen=tasks.find(t=>t.id===current&&t.status==='completed')||
    tasks.find(t=>t.status==='completed')||tasks[0];
  if(chosen)select.value=chosen.id;
  select.disabled=!tasks.length;
  return chosen;
 }
 function refresh(){
  if(!state.session){paint();return}
  const t=rebuildTaskOptions();
  state.selectionTask=t?.id||null;
  state.scene=null;
  if(t?.status==='completed')try{
   const base={description:state.session.description,modelSha256:state.session.sha256,taskId:t.id};
   if(state.source==='u5')state.scene=api.fromAudit({...base,audit:state.audit});
   else if(state.source==='u6')state.scene=api.fromQueue({...base,snapshot:state.u6});
   else if(state.u7)state.scene=api.fromEvidence({...base,evidence:state.u7});
   state.lastError=null;
  }catch(e){state.lastError=e.message}
  const opts=$('u8Tensor'),previous=opts.value;
  opts.replaceChildren();
  for(const v of state.scene?.views||[]){
   const opt=document.createElement('option');opt.value=v.tensor;
   opt.textContent=v.tensor+' ['+v.shape.join('×')+']';
   opts.append(opt);
  }
  if([...opts.options].some(o=>o.value===previous))opts.value=previous;
  opts.disabled=!opts.options.length;
  state.feed=state.captures.get(state.source+':'+state.selectionTask)||null;
  const scene=state.scene;
  $('u8Facts').textContent=scene?
   '真实模型 SHA-256 '+scene.modelSha256+' · 来源 '+state.source+
   ' · 观测 '+scene.observedCount+' · 真实中间张量 '+scene.observedIntermediateCount+
   ' · 算子图 '+(scene.graph.originalCount||0)+' 个算子 / 显示 '+(scene.graph.renderedNodes||0):
   state.session?'已验证计算图；暂无该来源下成功的实际 A/B 结果。':'—';
  const hasScene=!!scene;
  notice(hasScene?'实际实验数据已同步，灰色算子表示尚无对应测量。':
   t?.status==='failed'?'该任务执行失败；不显示伪造的张量变化。':
   '等待来自 U5/U6/U7 的真实 ONNX 实验输出。',hasScene?'success':'pending');
  paint();
 }
 function drawable(canvas){
  const ctx=canvas.getContext('2d');if(!ctx)return null;
  const rect=canvas.getBoundingClientRect(),width=Math.max(300,Math.round(rect.width||600)),height=Math.max(180,Math.round(rect.height||320));
  const dpr=Math.min(2,window.devicePixelRatio||1);
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){
   canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  }
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#061924';ctx.fillRect(0,0,width,height);
  return{ctx,width,height};
 }
 function label(ctx,text,x,y,color='#9dbbc2',font='11px sans-serif'){
  ctx.font=font;ctx.fillStyle=color;ctx.fillText(String(text),x,y);
 }
 function project(x,y,z,w,h,panX=0,span=1){
  const yaw=state.yaw,tilt=state.pitch;
  const u=x*Math.cos(yaw)+z*Math.sin(yaw),v=z*Math.cos(yaw)-x*Math.sin(yaw);
  const hh=y*Math.cos(tilt)-v*Math.sin(tilt),depth=y*Math.sin(tilt)+v*Math.cos(tilt);
  const perspective=state.zoom*2.7/(3.8-Math.max(-2.5,Math.min(2.5,depth)));
  const scale=Math.min(w/5.4,h/3.3)*perspective*span;
  return{x:panX+w/2+u*scale,y:h/2-hh*scale,depth,scale};
 }
 function value(v,mode){return mode==='before'?v.A:mode==='after'?v.B:v.delta}
 function paintMetric(value,scale,mode){
  if(!Number.isFinite(value))return '#526b73';
  if(mode==='delta'){
   if(value<=1e-9)return '#415963';
   const z=clamp(value/Math.max(scale,1e-12),0,1);
   return 'rgb('+Math.round(90+160*z)+','+Math.round(165+45*z)+','+Math.round(181-95*z)+')';
  }
  const z=clamp(value/Math.max(scale,1e-12),0,1);
  return 'rgb('+Math.round(60+55*z)+','+Math.round(120+112*z)+','+Math.round(150+66*z)+')';
 }
 function graphPaint(){
  const c=drawable($('u8Graph'));if(!c)return;
  const {ctx,width:w,height:h}=c,g=state.scene?.graph||state.graphBase;
  if(!g||g.status!=='available'||!g.nodes.length){
   label(ctx,'该模型未暴露可验证的 ONNX 算子依赖结构',20,35);return
  }
  const modes=state.mode==='compare'?['before','after','delta']:[state.mode];
  const pane=w/modes.length,ids=new Map(g.nodes.map((n,i)=>[n.id,i]));
  const projections=g.nodes.map((n,i)=>{
   const k=i/Math.max(1,g.nodes.length-1),theta=i*2.399963;
   const x=-1.9+3.8*k,y=Math.cos(theta)*(.45+.48*((i%31)/31)),z=Math.sin(theta)*(.45+.48*((i%31)/31));
   return{x,y,z,node:n}
  });
  const maxDelta=Math.max(1e-12,...g.nodes.map(n=>n.observed?.delta||0));
  const maxAB=Math.max(1e-12,...g.nodes.flatMap(n=>[n.observed?.A||0,n.observed?.B||0]));
  const rendered=[];
  for(let panel=0;panel<modes.length;panel++){
   const mode=modes[panel],x0=panel*pane;
   ctx.save();ctx.beginPath();ctx.rect(x0,0,pane,h);ctx.clip();
   const positions=projections.map(p=>project(p.x,p.y,p.z,pane,h,x0));
   for(const e of g.edges){
    const i=ids.get(e.from),j=ids.get(e.to);if(i===undefined||j===undefined)continue;
    ctx.strokeStyle='#55798545';ctx.lineWidth=.65;ctx.beginPath();
    ctx.moveTo(positions[i].x,positions[i].y);ctx.lineTo(positions[j].x,positions[j].y);ctx.stroke();
   }
   for(let i=0;i<projections.length;i++){
    const n=projections[i].node,p=positions[i],m=n.observed;
    const val=m?value(m,mode):null,scale=mode==='delta'?maxDelta:maxAB;
    const color=m?paintMetric(val,scale,mode):'#47636d';
    ctx.fillStyle=color;ctx.beginPath();ctx.arc(p.x,p.y,m?3.8:2,0,Math.PI*2);ctx.fill();
    if(m){ctx.strokeStyle='#b6f4dc';ctx.lineWidth=.7;ctx.stroke()}
    if(panel===0)rendered.push({id:n.id,op:n.op,measured:!!m,tensor:m?.tensor||null,
     A:m?.A??null,B:m?.B??null,delta:m?.delta??null});
   }
   label(ctx,mode==='before'?'A · 实测':mode==='after'?'B · 实测':'Δ · 实测',x0+12,22);
   ctx.restore();
  }
  $('u8GraphInfo').textContent='真实 ONNX '+g.originalCount+' 个算子 / '+g.originalEdges+
   ' 条依赖 · 当前绘制 '+g.renderedNodes+' 个真实算子 / '+g.renderedEdges+
   ' 条真实边 · 发亮的仅为被观测张量的生产算子 · 空间位置只是导航布局';
  state.renderedNodes=rendered;
 }
 function unitsPaint(){
  const c=drawable($('u8Units'));if(!c)return;
  const {ctx,width:w,height:h}=c,scene=state.scene;
  const tensor=$('u8Tensor').value,view=scene?.views.find(v=>v.tensor===tensor);
  if(!view||!view.units.length){
   label(ctx,view?'当前张量未声明可安全解释的 Token/Channel/Feature 轴。':'先运行真正的实验并选择包含真实轴统计的张量。',18,35);return
  }
  const modes=state.mode==='compare'?['before','after','delta']:[state.mode],pane=w/modes.length;
  const maxSignal=Math.max(1e-12,...view.units.flatMap(v=>[v.A,v.B]));
  const maxDelta=Math.max(1e-12,...view.units.map(v=>v.delta));
  const step=Math.max(1,Math.ceil(view.units.length/800));
  for(let k=0;k<modes.length;k++){
   const mode=modes[k],ox=k*pane;
   ctx.save();ctx.beginPath();ctx.rect(ox,0,pane,h);ctx.clip();
   // Indexed helix spreads genuine units in 3D. Coordinates do not imply
   // medical patch positions or a physical neuron layout.
   const dots=[];
   for(let i=0;i<view.units.length;i+=step){
    const u=view.units[i],t=i/Math.max(1,view.units.length-1),theta=i*2.399963;
    const pos=project(-1.65+3.3*t,Math.cos(theta)*.84,Math.sin(theta)*.84,pane,h,ox);
    dots.push({...pos,u});
   }
   dots.sort((a,b)=>a.depth-b.depth);
   for(const d of dots){
    ctx.fillStyle=paintMetric(value(d.u,mode),mode==='delta'?maxDelta:maxSignal,mode);
    ctx.beginPath();ctx.arc(d.x,d.y,clamp(1.8+d.scale*.6,1.4,4.4),0,Math.PI*2);ctx.fill();
   }
   label(ctx,(view.role||'unit')+' / '+mode+(step>1?' (1/'+step+' sampled indices)':''),ox+10,22);
   ctx.restore();
  }
  $('u8UnitsInfo').textContent=view.units.length+' 个实际 '+view.role+' 索引，'+view.shape.join('×')+
   ' · 每点 A/B/Δ 来自 U4 真实数值轴聚合，不再限制 197 Token 或 192 Channel · 未证明 Token 对应 MRI 区域';
 }
 function planeMode(value,mode){return mode==='before'?value.A:mode==='after'?value.B:value.delta}
 function drawPlane(ctx,plane,mode,x,y,availableW,availableH,signalMin,signalMax,deltaMax){
  const data=planeMode(plane,mode),width=plane.cols,height=plane.rows;
  const tmp=document.createElement('canvas');tmp.width=width;tmp.height=height;
  const imgctx=tmp.getContext('2d');if(!imgctx)return;
  const frame=imgctx.createImageData(width,height);
  const rng=Math.max(1e-12,signalMax-signalMin),dr=Math.max(1e-12,deltaMax);
  for(let i=0;i<data.length;i++){
   let r,g,b;
   if(mode==='delta'){
    const d=data[i],ratio=clamp(Math.abs(d)/dr,0,1);
    if(Math.abs(d)<=1e-7){r=19;g=35;b=44}
    else if(d>0){r=Math.round(69+185*ratio);g=Math.round(51+75*ratio);b=Math.round(63+45*ratio)}
    else {r=Math.round(39+34*ratio);g=Math.round(92+107*ratio);b=Math.round(117+119*ratio)}
   }else{
    const t=clamp((data[i]-signalMin)/rng,0,1),c=Math.round(t*255);
    r=c;g=c;b=c;
   }
   const p=i*4;frame.data[p]=r;frame.data[p+1]=g;frame.data[p+2]=b;frame.data[p+3]=255;
  }
  imgctx.putImageData(frame,0,0);
  const scale=Math.min(availableW/width,availableH/height),dw=width*scale,dh=height*scale;
  ctx.imageSmoothingEnabled=false;ctx.drawImage(tmp,x+(availableW-dw)/2,y+(availableH-dh)/2,dw,dh);
 }
 function paintVolume(){
  const c=drawable($('u8Volume'));if(!c)return;
  const {ctx,width:w,height:h}=c,v=state.feed;
  if(!v){
   label(ctx,'暂无匹配的真实 A/B 输入切片。请选择布局并运行实验。',18,35);
   $('u8VolumeInfo').textContent=state.lastError?'该输入不能可视化：'+state.lastError:
    '仅识别真实模型输入，MRI 身份、空间方向和解剖配准均不自动推断。';
   return;
  }
  const orientation=$('u8SliceAxis').value,mode=state.mode;
  let p;
  try{p=api.plane(v,orientation,Number($('u8Slice').value))}catch(e){$('u8VolumeInfo').textContent=e.message;return}
  if(!p){
   label(ctx,'2D 输入没有该正交方向；请选择第一视图。',18,35);return
  }
  const min=Math.min(...v.A,...v.B),max=Math.max(...v.A,...v.B);
  const modes=mode==='compare'?['before','after','delta']:[mode],pane=w/modes.length;
  for(let i=0;i<modes.length;i++){
   const x=i*pane;ctx.save();ctx.beginPath();ctx.rect(x,0,pane,h);ctx.clip();
   drawPlane(ctx,p,modes[i],x+5,30,pane-10,h-44,min,max,v.maxAbsDelta);
   label(ctx,modes[i]==='before'?'A 原始输入':modes[i]==='after'?'B 干预输入':'Δ 输入差分',x+12,20);
   ctx.restore();
  }
  $('u8VolumeInfo').textContent=v.kind+' · ['+v.shape.join('×')+'] · '+v.layout+
   ' · 实际输入采样 ['+v.sampled.join('×')+']'+(v.resampled?' (等距降采样)':'')+
   ' · 变化采样点 '+v.changedSampled+' · '+p.orientation+' 轴切片 '+p.index+
   ' · 显式张量轴，尚未核验 MRI 世界坐标/病灶';
 }
 function updateSlice(){
  const v=state.feed,axis=$('u8SliceAxis').value;
  $('u8SliceAxis').disabled=!v||v.spatialDims===2;
  if(!v){$('u8Slice').disabled=true;$('u8Slice').value=0;return}
  if(v.spatialDims===2&&axis!=='axial')$('u8SliceAxis').value='axial';
  const name=$('u8SliceAxis').value,k=name==='axial'?0:name==='coronal'?1:2;
  const length=v.spatialDims===3?v.sampled[k]:1;
  $('u8Slice').disabled=length<=1;$('u8Slice').max=String(Math.max(0,length-1));
  $('u8Slice').value=String(Math.floor((length-1)/2));
 }
 function paint(){graphPaint();unitsPaint();updateSlice();paintVolume()}
 function getSnapshot(){
  return{version:api.VERSION,hasScene:!!state.scene,source:state.source,taskId:state.selectionTask,
   modelSha256:state.session?.sha256||null,
   graph:state.scene?.graph||state.graphBase||null,views:state.scene?.views||[],
   hasModelInput:!!state.feed,
   modelInput:state.feed?{layout:state.feed.layout,shape:state.feed.shape,sampled:state.feed.sampled,
    changed:state.feed.changedSampled,source:state.feed.source,anatomicalStatus:state.feed.anatomicalStatus}:null,
   mode:state.mode,renderedNodes:state.renderedNodes||[],lastError:state.lastError};
 }
 $('u8Source').onchange=()=>{state.source=$('u8Source').value;refresh()};
 $('u8Task').onchange=()=>{state.selectionTask=$('u8Task').value;refresh()};
 $('u8Tensor').onchange=unitsPaint;
 $('u8Mode').onchange=()=>{state.mode=$('u8Mode').value;paint()};
 $('u8InputLayout').onchange=()=>{state.feed=null;state.captures.clear();paint();
  notice('输入布局已更改。请重新运行实验以从真实模型输入重新抽样。')};
 $('u8InputName').onchange=$('u8InputLayout').onchange;
 $('u8SliceAxis').onchange=()=>{updateSlice();paintVolume()};
 $('u8Slice').oninput=paintVolume;
 const graph=$('u8Graph'),token=$('u8Units');
 for(const canvas of [graph,token]){
  let down=null;
  canvas.addEventListener('pointerdown',e=>{down={x:e.clientX,y:e.clientY,yaw:state.yaw,pitch:state.pitch};canvas.setPointerCapture(e.pointerId)});
  canvas.addEventListener('pointermove',e=>{
   if(!down)return;state.yaw=down.yaw+(e.clientX-down.x)/130;
   state.pitch=clamp(down.pitch+(e.clientY-down.y)/180,-1.35,1.35);
   graphPaint();unitsPaint();
  });
  canvas.addEventListener('pointerup',()=>down=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();state.zoom=clamp(state.zoom*(e.deltaY>0?.92:1.08),.4,3);
   graphPaint();unitsPaint()},{passive:false});
 }
 window.addEventListener('resize',paint);
 window.SCEPT_VISUAL_STUDIO=Object.freeze({attach,snapshots,capture,onAB,refresh,getSnapshot,
  plane:api.plane});
 notice('统一工作台已就绪。请选择真实模型并执行实验。');
 paint();
})();