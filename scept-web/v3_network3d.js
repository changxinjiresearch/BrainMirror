'use strict';
/* SCEPT v3 true 3D ONNX dependency navigator.
   Geometry positions are only a layout. Edges are exact producer-to-consumer
   tensor dependencies from the real model, never invented layer links.
   Neutral op colors indicate operator types, NOT inferred activations.
   Only exposed ONNX probe-output producer nodes carry measured Δ overlays.
   All transforms use genuine 3D camera rotation/perspective, WebGL2 if available.
*/
(function(){
let canvas=document.getElementById('networkCanvas');if(!canvas)return;
const $=id=>document.getElementById(id);
const state={graph:null,mode:'overview',layer:6,selected:null,nodes:[],edges:[],allLayerCounts:[],
 yaw:-.3,pitch:.16,distance:15,backend:null,gl:null,ctx:null,observed:new Map(),probeNames:new Map(),
 drag:null,down:null,lastTap:0,layoutSerial:0,graphNodes:0,graphEdges:0,layeringKind:'transformer',visualMode:'compare',deltaScale:null,signalScale:null,observedSource:null,observationVersion:0,measuredCount:0};
const dec=(n,places=3)=>Number.isFinite(n)?n.toFixed(places):'—';
const light={MatMul:[.37,.84,.76],Gemm:[.46,.88,.73],Conv:[.43,.78,.96],Add:[.44,.69,.91],LayerNormalization:[.75,.61,.97],Softmax:[.97,.78,.49],Reshape:[.43,.65,.70],Transpose:[.42,.65,.70],Gather:[.48,.74,.77]};
const decodeLayer=(name)=>{
 const m=String(name||'').match(/(?:^|\/)vit\/encoder\/layer\.(\d+)(?:\/|$)/);
 return m?Number(m[1]):/vit\/embeddings\//.test(name)?-1:/vit\/layernorm|classifier/.test(name)?12:-2;
};
function initialize(){
 try{
  const probeCanvas=document.createElement('canvas');
  const preflight=probeCanvas.getContext('webgl2',{alpha:false});
  if(!preflight)throw Error('WebGL2 unavailable (software perspective fallback)');
  const gl=canvas.getContext('webgl2',{alpha:false,antialias:true,preserveDrawingBuffer:true});
  if(!gl)throw Error('WebGL2 unavailable on primary viewport');
  function shader(type,code){const s=gl.createShader(type);gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error('Shader '+gl.getShaderInfoLog(s));return s}
  const vertex=`#version 300 es
   precision highp float;
   in vec3 aPosition; in vec3 aColor; in float aSize;
   uniform float uYaw,uPitch,uDistance,uScale,uAspect,uPoint;
   out vec3 vColor;
   void main(){
    float ca=cos(uYaw),sa=sin(uYaw),cp=cos(uPitch),sp=sin(uPitch);
    vec3 p=vec3(aPosition.x*ca+aPosition.z*sa,aPosition.y,-aPosition.x*sa+aPosition.z*ca);
    p=vec3(p.x,p.y*cp-p.z*sp,p.y*sp+p.z*cp);
    float d=max(1.0,uDistance-p.z);
    gl_Position=vec4(p.x*uScale/uAspect,p.y*uScale,0.0,d);
    gl_PointSize=uPoint>0.5?min(32.0,max(3.0,aSize*13.0/d)):1.0;
    vColor=aColor;
   }`;
  const frag=`#version 300 es
   precision mediump float; in vec3 vColor; uniform float uPoint;
   out vec4 outColor;
   void main(){
    if(uPoint>0.5 && distance(gl_PointCoord,vec2(.5))>.48)discard;
    outColor=vec4(vColor,1.0);
   }`;
  const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,frag));gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
  const stride=7*4,buffer=gl.createBuffer();
  gl.useProgram(program);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  for(const [name,size,offset]of [['aPosition',3,0],['aColor',3,3*4],['aSize',1,6*4]]){
   const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,stride,offset)
  }
  state.gl={gl,program,buffer,uniform:Object.fromEntries(['uYaw','uPitch','uDistance','uScale','uAspect','uPoint'].map(n=>[n,gl.getUniformLocation(program,n)]))};
  state.backend='webgl2';
 }catch(e){
  // WebGL and Canvas2D are mutually exclusive on the same canvas once chosen.
  // If WebGL context/shader creation fails, replace the *unbound* canvas
  // before initializing the fallback, then attach interaction handlers.
  state.rendererError=String(e.message||e);
  let ctx=null;
  try{ctx=canvas.getContext('2d')}catch{}
  if(!ctx){
   const replacement=canvas.cloneNode(false);
   canvas.parentNode.replaceChild(replacement,canvas);
   canvas=replacement;
   try{ctx=canvas.getContext('2d')}catch{}
  }
  state.ctx=ctx;
  state.backend=ctx?'perspective-fallback':'unavailable';
 }
}
function layerOf(node,id){return state.layeringKind==='transformer'?decodeLayer(node.name):Math.min(11,Math.floor(id*12/Math.max(1,state.graphNodes)))}
function listLayers(g){
 const counts=Array.from({length:12},()=>0);
 g.nodes.forEach((n,id)=>{const l=layerOf(n,id);if(l>=0&&l<12)counts[l]++});return counts;
}
// Fixed shared per-metric scales are initialized on the first real run and
// retained across re-runs. The explicit recalibration button changes scales.
const clamp=x=>Math.max(0,Math.min(1,x));
// Smooth scale keeps large changes visually distinguishable without\n// silently re-normalizing between experiments: t=(value/scale)/(1+value/scale).\nconst strength=(value,scale)=>!Number.isFinite(value)||!scale?0:Math.max(0,value)/(Math.max(0,value)+scale);
const fmtValue=x=>Number.isFinite(x)?(Math.abs(x)>0&&Math.abs(x)<.0001?x.toExponential(3):x.toFixed(5)):'—';
function observedFor(node){const layer=state.probeNames.get(node.id);return layer===undefined?null:state.observed.get(layer)||null}
function metricValue(o,mode){
 if(!o)return null;
 return mode==='before'?o.beforeMean:mode==='after'?o.afterMean:o.deltaMean;
}
function nodeVisual(node,mode){
 const observed=observedFor(node);
 if(!observed)return {color:[.35,.48,.53],size:node.size,measured:false,metric:null};
 const value=metricValue(observed,mode);
 // A true zero-delta tensor must look neutral, not artificially active.
 if(mode==='delta'&&value<=1e-7)return{color:[.35,.48,.53],size:8,measured:true,metric:value,intensity:0,
  delta:observed.deltaMean,before:observed.beforeMean,after:observed.afterMean,layer:observed.layer};
 const scale=mode==='delta'?state.deltaScale:state.signalScale;
 const t=strength(value,scale);
 // Quantitative magnitude is represented by a continuous color and radius.
 // B uses the exact same activation scale as A for meaningful comparison.
 const color=mode==='delta'
 ? [.31+.65*t,.65-.11*t,.62-.40*t]
 : [.29+.69*t,.62+.23*t,.64-.19*t];
 return{color,size:15+18*t,measured:true,metric:value,intensity:t,delta:observed.deltaMean,
         before:observed.beforeMean,after:observed.afterMean,layer:observed.layer};
}
function calibrate(){
 const observed=[...state.observed.values()];
 if(!observed.length)return;
 const maxDelta=Math.max(...observed.map(x=>x.deltaMean));
 const maxSignal=Math.max(...observed.flatMap(x=>[x.beforeMean,x.afterMean]));
 state.deltaScale=Math.max(1e-12,maxDelta);
 state.signalScale=Math.max(1e-12,maxSignal);
 updateUI();paint();
}
function setVisualMode(mode){
 if(!['compare','before','after','delta'].includes(mode))return false;
 state.visualMode=mode;updateUI();paint();return true;
}
function nodePosition(id,part,total,layer,focus){
 const ordinal=part+1;
 if(!focus){
  const radius=.22+((id*17)%11)*.036;
  const phi=ordinal*2.399963+layer*.4;
  return[-5.2+layer*.94,Math.sin(phi)*radius*3.7,Math.cos(phi)*radius*3.7];
 }
 const span=Math.max(1,total-1),u=part/span,angle=ordinal*2.399963;
 const radius=.45+.5*Math.sqrt((part%53)/53);
 return[-3.7+7.4*u,Math.cos(angle)*radius*1.65,Math.sin(angle)*radius*1.55];
}
function makeView(){
 const g=state.graph;if(!g){state.nodes=[];state.edges=[];paint();return}
 const ids=[],layerBuckets=Array.from({length:12},()=>[]);
 g.nodes.forEach((n,id)=>{const l=layerOf(n,id);if(l>=0&&l<12)layerBuckets[l].push(id)});
 const focus=state.mode==='layer';
 if(focus){ids.push(...layerBuckets[state.layer]);}
 else for(let l=0;l<12;l++){
  const bucket=layerBuckets[l],cap=44;
  if(bucket.length<=cap)ids.push(...bucket);
  else {
   const used=new Set();
   for(let j=0;j<cap;j++){const id=bucket[Math.floor((j+.5)*bucket.length/cap)];used.add(id)}
   // Observed probe producing nodes must remain visible even if not sampled.
   for(const id of bucket)if(state.probeNames.has(id))used.add(id);
   ids.push(...used);
  }
 }
 const order=new Map(ids.map((id,i)=>[id,i])),positions=[];
 const counters=new Map();
 for(const id of ids){
  const n=g.nodes[id],layer=layerOf(n,id),bucket=layerBuckets[layer]||[],part=bucket.indexOf(id);
  const p=nodePosition(id,part,bucket.length,layer,focus);
  positions.push({id,layer,op:n.op,name:n.name,inputs:n.inputs,outputs:n.outputs,p,size:state.probeNames.has(id)?8:5});
 }
 state.nodes=positions;state.edges=g.edges.filter(e=>order.has(e.from)&&order.has(e.to)).map(e=>({a:order.get(e.from),b:order.get(e.to),from:e.from,to:e.to,tensor:e.tensor}));
 state.layoutSerial++;paint();updateUI();
}
function setGraph(g){
 if(!g||!Array.isArray(g.nodes)||!Array.isArray(g.edges))return;
 if(g!==state.graph){state.graph=g;state.graphNodes=g.nodes.length;state.graphEdges=g.edges.length;
  state.layeringKind=g.nodes.some(n=>decodeLayer(n.name)>=0&&decodeLayer(n.name)<12)?'transformer':'sequence';
  state.probeNames.clear();
  g.nodes.forEach((n,i)=>n.outputs.forEach(name=>{
   const m=name.match(/\/vit\/encoder\/layer\.(\d+)\/attention\/attention\/value\/MatMul_output_0$/);
   if(m)state.probeNames.set(i,Number(m[1]));
  }));
  state.observed.clear();state.observedSource=null;state.measuredCount=0;state.deltaScale=null;state.signalScale=null;
  state.allLayerCounts=listLayers(g);state.mode='overview';state.selected=null;state.yaw=-.3;state.pitch=.16;state.distance=15;makeView();
 }
}
function setObservations(results){
 // Preserve data-driven colors when the user only changes the selected layer.
 if(state.observedSource===results)return;
 state.observedSource=results;state.observed.clear();
 for(const l of results||[]){
  if(!l||!Number.isInteger(l.layer)||!Array.isArray(l.before)||!Array.isArray(l.after)||
    l.before.length!==l.after.length||l.before.length===0)continue;
  let a=0,b=0,d=0;
  for(let i=0;i<l.before.length;i++){
    if(!Number.isFinite(l.before[i])||!Number.isFinite(l.after[i]))throw Error('Invalid measured ONNX tensor values');
    a+=Math.abs(l.before[i]);b+=Math.abs(l.after[i]);d+=Math.abs(l.after[i]-l.before[i]);
  }
  state.observed.set(l.layer,{layer:l.layer,probe:l.probe,
   beforeMean:a/l.before.length,afterMean:b/l.before.length,deltaMean:d/l.before.length,
   size:l.before.length});
 }
 state.measuredCount=state.observed.size;state.observationVersion++;
 // Calibrate the initial display only once; do not silently rescale on re-run.
 if(state.deltaScale===null&&state.measuredCount)calibrate();
 paint();updateUI();
}
function setLayer(layer){const n=Number(layer);if(!Number.isInteger(n)||n<0||n>11)return;
 state.layer=n;updateUI();paint();
}
function enterLayer(layer=state.layer){
 if(!state.graph)return false;
 if(!Number.isInteger(layer)||layer<0||layer>11||!state.allLayerCounts[layer])return false;
 state.mode='layer';state.layer=layer;state.selected=null;state.yaw=-.3;state.pitch=.12;state.distance=11;
 makeView();return true;
}
function overview(){if(!state.graph)return;state.mode='overview';state.selected=null;state.distance=15;state.yaw=-.3;state.pitch=.16;makeView()}
function zoom(dir){state.distance=Math.max(4,Math.min(37,state.distance+dir));paint();updateUI()}
function uniforms(point,viewWidth,viewHeight){
 const {gl,uniform}=state.gl;
 gl.uniform1f(uniform.uYaw,state.yaw);
 gl.uniform1f(uniform.uPitch,state.pitch);
 gl.uniform1f(uniform.uDistance,state.distance);
 gl.uniform1f(uniform.uScale,3.6);
 gl.uniform1f(uniform.uAspect,viewWidth/viewHeight);
 gl.uniform1f(uniform.uPoint,point?1:0);
}
function pack(arr){
 const data=new Float32Array(arr.length*7);
 for(let i=0;i<arr.length;i++)data.set([...arr[i].p,...arr[i].color,arr[i].size],i*7);
 return data;
}
function paneModes(){return state.visualMode==='compare'?['before','after']:[state.visualMode]}
function paint(){
 const nodes=state.nodes,modes=paneModes();
 if(state.backend==='webgl2'){
  const {gl,buffer}=state.gl,ratio=window.devicePixelRatio||1;
  const w=Math.min(1500,Math.max(2,Math.round(canvas.clientWidth*ratio))),h=Math.min(850,Math.max(2,Math.round(canvas.clientHeight*ratio)));
  if(w!==canvas.width||h!==canvas.height){canvas.width=w;canvas.height=h}
  gl.viewport(0,0,w,h);gl.clearColor(.025,.085,.12,1);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(state.gl.program);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  const allLines=[];
  for(const e of state.edges){const a=nodes[e.a],b=nodes[e.b];allLines.push({p:a.p,color:[.19,.31,.36],size:1},{p:b.p,color:[.19,.31,.36],size:1})}
  for(let i=0;i<modes.length;i++){
   const x0=Math.round(w*i/modes.length),x1=Math.round(w*(i+1)/modes.length),paneWidth=x1-x0;
   gl.viewport(x0,0,paneWidth,h);
   if(allLines.length){gl.bufferData(gl.ARRAY_BUFFER,pack(allLines),gl.DYNAMIC_DRAW);uniforms(false,paneWidth,h);gl.drawArrays(gl.LINES,0,allLines.length)}
   if(nodes.length){
    const pts=nodes.map(n=>({...n,...nodeVisual(n,modes[i])}));
    for(const pt of pts)if(state.selected===pt.id){pt.color=[1,.94,.63];pt.size+=5}
    gl.bufferData(gl.ARRAY_BUFFER,pack(pts),gl.DYNAMIC_DRAW);
    uniforms(true,paneWidth,h);gl.drawArrays(gl.POINTS,0,pts.length);
   }
  }
 }else if(state.ctx){
  const ctx=state.ctx,w=canvas.clientWidth||canvas.width,h=canvas.clientHeight||canvas.height;
  ctx.setTransform(canvas.width/w,0,0,canvas.height/h,0,0);
  ctx.fillStyle='#061923';ctx.fillRect(0,0,w,h);
  for(let i=0;i<modes.length;i++){
   const kind=modes[i],left=i*w/modes.length,right=(i+1)*w/modes.length;
   ctx.save();ctx.beginPath();ctx.rect(left,0,right-left,h);ctx.clip();
   for(const e of state.edges){
    const a=screenPoint(nodes[e.a].p,kind),b=screenPoint(nodes[e.b].p,kind);
    ctx.strokeStyle='#2c4d58';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
   }
   const pts=nodes.map(n=>({...n,q:screenPoint(n.p,kind)})).sort((a,b)=>b.q.depth-a.q.depth);
   for(const n of pts){
    const vis=nodeVisual(n,kind),rgb=(state.selected===n.id?[1,.94,.63]:vis.color).map(v=>Math.round(v*255));
    const radius=vis.measured?Math.max(6,vis.size*.37):3;
    if(vis.measured){ctx.fillStyle='rgba(255,166,65,.13)';ctx.beginPath();ctx.arc(n.q.x,n.q.y,radius+5,0,Math.PI*2);ctx.fill()}
    ctx.fillStyle='rgb('+rgb.join(',')+')';ctx.beginPath();ctx.arc(n.q.x,n.q.y,radius,0,Math.PI*2);ctx.fill();
    if(state.selected===n.id){ctx.strokeStyle='#fff7c8';ctx.lineWidth=2;ctx.stroke()}
   }
   ctx.restore();
  }
  if(modes.length===2){ctx.strokeStyle='#3a7780';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke()}
  ctx.setTransform(1,0,0,1,0,0);
 }
}
function screenPoint(p,which){
 const ca=Math.cos(state.yaw),sa=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
 const x=p[0]*ca+p[2]*sa,z=-p[0]*sa+p[2]*ca,y=p[1]*cp-z*sp,dz=p[1]*sp+z*cp,depth=Math.max(1,state.distance-dz);
 const modes=paneModes(),view=which|| (modes.length===2?'after':modes[0]);
 const paneIndex=modes.length===2&&view==='after'?1:0;
 const width=(canvas.clientWidth||canvas.width)/modes.length,height=canvas.clientHeight||canvas.height;
 const aspect=width/height;
 return{x:(x*3.6/(aspect*depth)+1)*width/2+paneIndex*width,
         y:(1-y*3.6/depth)*height/2,depth};
}
function nearest(x,y,max=21){
 let best=null,d=max*max;
 const view=state.visualMode==='compare'?(x<(canvas.clientWidth||canvas.width)/2?'before':'after'):state.visualMode;
 for(const n of state.nodes){const p=screenPoint(n.p,view),dd=(p.x-x)**2+(p.y-y)**2;if(dd<d){d=dd;best=n}}
 return best;
}
function selectNode(id){
 const n=state.nodes.find(n=>n.id===id);if(!n)return false;
 state.selected=id;if(n.layer>=0&&n.layer<=11)state.layer=n.layer;
 const g=state.graph,incoming=g.edges.filter(e=>e.to===id),outgoing=g.edges.filter(e=>e.from===id);
 $('networkNodeDetails').textContent='算子 #'+id+' · '+(n.op||'Unknown')+' · '+(n.name||'(无名称)')+
   '\n输入张量：'+(n.inputs.join('、')||'无')+'\n输出张量：'+(n.outputs.join('、')||'无')+
   '\n真实数据依赖：'+incoming.length+' 条输入、'+outgoing.length+' 条输出；'+
   '示例：'+[...incoming.slice(0,2),...outgoing.slice(0,2)].map(e=>'#'+e.from+' → #'+e.to+' ('+e.tensor+')').join('；');
 const measured=observedFor(n);
 if(measured){
   $('networkNodeDetails').textContent+='\n\n此节点已开放真实输出：'+measured.probe+
     '\n原始 A · 平均绝对激活：'+fmtValue(measured.beforeMean)+
     '\n修改 B · 平均绝对激活：'+fmtValue(measured.afterMean)+
     '\n真实平均 |B − A|：'+fmtValue(measured.deltaMean)+
     '\n观测元素数：'+measured.size+'；颜色对应此张量，不代表未观测的相邻节点。';
 }else{
   $('networkNodeDetails').textContent+='\n该算子目前未开放可测张量，不能判断它的具体激活变化。';
 }
 if(state.mode==='overview')$('networkEnter').disabled=false;
 if(typeof window.SCEPT_NETWORK3D_LAYER_SELECTED==='function')window.SCEPT_NETWORK3D_LAYER_SELECTED(state.layer);
 paint();updateUI();return true;
}
function updateUI(){
 const labels={compare:'双视图 A/B',before:'原始输入 A',after:'修改后 B',delta:'真实差异 Δ'};
 const ids={compare:'responseCompare',before:'responseBefore',after:'responseAfter',delta:'responseDelta'};
 for(const [name,id] of Object.entries(ids)){const b=$(id);b.classList.toggle('active',name===state.visualMode);b.setAttribute('aria-pressed',String(name===state.visualMode))}
 $('networkPairLabels').hidden=state.visualMode!=='compare';
 const ds=state.deltaScale,ss=state.signalScale;
 const detail=state.measuredCount?
 '已观测 '+state.measuredCount+' 个真正的张量输出生产节点；A/B 共用参考刻度 '+fmtValue(ss)+
 '，Δ 参考刻度 '+fmtValue(ds)+'（首次实验后锁定，平滑色映射）':'等待真实隐藏层数据。未开放观测点均呈中性颜色。';
 $('responseScaleLabel').textContent=state.measuredCount?'固定参考刻度：A/B '+fmtValue(ss)+' · Δ '+fmtValue(ds):'等待首次真实模型计算';
 $('responseLegendDetail').textContent=detail+'；映射函数 t=数值/(数值+刻度)，不会饱和成同一种颜色或静默重新标定。';
 $('responseCalibrate').disabled=!state.measuredCount;

 $('networkOverview').disabled=state.mode==='overview';
 $('networkEnter').disabled=!state.graph||state.mode==='layer'||!state.allLayerCounts[state.layer];
 $('networkDepth').textContent=state.mode==='layer'?(state.layeringKind==='transformer'?'第 '+state.layer+' 层':'顺序段 '+state.layer)+' · 算子详细视图':(state.layeringKind==='transformer'?'Transformer':'ONNX 网络')+' · 三维结构总览';
 $('networkStats').textContent=state.graph?
 (state.mode==='layer'?(state.layeringKind==='transformer'?'第 ':'顺序段 ')+state.layer+' ，真实算子 '+state.allLayerCounts[state.layer]+' 个 · ':'实际 ONNX 结构 · ')+
 '显示 '+state.nodes.length+'/'+state.graphNodes+' 算子 · '+state.edges.length+'/'+state.graphEdges+' 条直接数据依赖 · '+(state.backend==='webgl2'?'WebGL 3D':'透视降级模式')
 :'等待获取真实 ONNX 计算结构';
 if(state.mode==='layer'&&state.selected===null)$('networkNodeDetails').textContent='当前显示第 '+state.layer+' 层全部已解析算子。单击节点查看真实张量输入输出；连线仅代表 ONNX 定义的直接依赖。';
}
function bind(){
 for(const [id,mode] of [['responseCompare','compare'],['responseDelta','delta'],['responseBefore','before'],['responseAfter','after']]){
  $(id).onclick=()=>setVisualMode(mode);
 }
 $('responseCalibrate').onclick=calibrate;

 $('networkOverview').onclick=overview;
 $('networkEnter').onclick=()=>enterLayer(state.layer);
 $('networkZoomIn').onclick=()=>zoom(-1.3);
 $('networkZoomOut').onclick=()=>zoom(1.3);
 $('networkReset').onclick=()=>{state.yaw=-.3;state.pitch=.16;state.distance=state.mode==='layer'?11:15;paint();updateUI()};
 canvas.addEventListener('pointerdown',e=>{state.down={x:e.clientX,y:e.clientY};state.drag={x:e.clientX,y:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId)});
 canvas.addEventListener('pointermove',e=>{
  if(!state.drag)return;
  const dx=e.clientX-state.drag.x,dy=e.clientY-state.drag.y;
  if(Math.abs(e.clientX-state.down.x)+Math.abs(e.clientY-state.down.y)>4)state.drag.moved=true;
  if(state.drag.moved){state.yaw+=dx*.008;state.pitch=Math.max(-1.28,Math.min(1.28,state.pitch+dy*.008));paint()}
  state.drag.x=e.clientX;state.drag.y=e.clientY;
 });
 canvas.addEventListener('pointerup',e=>{
  const moved=state.drag?.moved;
  state.drag=null;if(moved)return;
  const r=canvas.getBoundingClientRect(),n=nearest(e.clientX-r.left,e.clientY-r.top);
  if(n){selectNode(n.id)}
 });
 canvas.addEventListener('dblclick',e=>{if(state.selected!==null)enterLayer(state.layer)});
 canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY>0?1.1:-1.1)},{passive:false});
 canvas.addEventListener('keydown',e=>{
  if(e.key==='+'||e.key==='='){e.preventDefault();zoom(-1)}
  if(e.key==='-'){e.preventDefault();zoom(1)}
  if(e.key==='Escape'){e.preventDefault();overview()}
  if(e.key==='Enter'){e.preventDefault();enterLayer(state.layer)}
 });
 window.addEventListener('resize',paint);
}
initialize();bind();paint();updateUI();
window.SCEPT_NETWORK3D={
 setGraph,setObservations,highlightLayer:setLayer,enterLayer,overview,selectNode,setVisualMode,calibrate,
 getSnapshot:()=>({backend:state.backend,rendererError:state.rendererError||null,visualMode:state.visualMode,deltaScale:state.deltaScale,signalScale:state.signalScale,measuredCount:state.measuredCount,observationVersion:state.observationVersion,measuredNodes:state.nodes.filter(n=>observedFor(n)).map(n=>({nodeId:n.id,layer:n.layer,meanA:observedFor(n).beforeMean,meanB:observedFor(n).afterMean,meanDelta:observedFor(n).deltaMean,deltaColor:nodeVisual(n,'delta').color,deltaSize:nodeVisual(n,'delta').size})),mode:state.mode,layer:state.layer,yaw:state.yaw,pitch:state.pitch,distance:state.distance,displayNodes:state.nodes.length,displayEdges:state.edges.length,totalNodes:state.graphNodes,totalEdges:state.graphEdges,edgesVerified:state.edges.every(e=>state.graph?.edges.some(real=>real.from===e.from&&real.to===e.to&&real.tensor===e.tensor)),observedProducers:state.probeNames.size,selected:state.selected,layoutSerial:state.layoutSerial,pickTarget:state.nodes.length?{id:state.nodes[Math.floor(state.nodes.length/2)].id,...screenPoint(state.nodes[Math.floor(state.nodes.length/2)].p)}:null})
};
})();
