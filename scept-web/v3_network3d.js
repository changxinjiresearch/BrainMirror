'use strict';
/* SCEPT v3 true 3D ONNX dependency navigator.
   Geometry positions are only a layout. Edges are exact producer-to-consumer
   tensor dependencies from the real model, never invented layer links.
   Neutral op colors indicate operator types, NOT inferred activations.
   Only exposed ONNX probe-output producer nodes carry measured Δ overlays.
   All transforms use genuine 3D camera rotation/perspective, WebGL2 if available.
*/
(function(){
const canvas=document.getElementById('networkCanvas');if(!canvas)return;
const $=id=>document.getElementById(id);
const state={graph:null,mode:'overview',layer:6,selected:null,nodes:[],edges:[],allLayerCounts:[],
 yaw:-.3,pitch:.16,distance:15,backend:null,gl:null,ctx:null,observed:new Map(),probeNames:new Map(),
 drag:null,down:null,lastTap:0,layoutSerial:0,graphNodes:0,graphEdges:0};
const dec=(n,places=3)=>Number.isFinite(n)?n.toFixed(places):'—';
const light={MatMul:[.37,.84,.76],Gemm:[.46,.88,.73],Conv:[.43,.78,.96],Add:[.44,.69,.91],LayerNormalization:[.75,.61,.97],Softmax:[.97,.78,.49],Reshape:[.43,.65,.70],Transpose:[.42,.65,.70],Gather:[.48,.74,.77]};
const decodeLayer=(name)=>{
 const m=String(name||'').match(/(?:^|\/)vit\/encoder\/layer\.(\d+)(?:\/|$)/);
 return m?Number(m[1]):/vit\/embeddings\//.test(name)?-1:/vit\/layernorm|classifier/.test(name)?12:-2;
};
function initialize(){
 try{
  const gl=canvas.getContext('webgl2',{alpha:false,antialias:true,preserveDrawingBuffer:true});
  if(!gl)throw Error('WebGL2 unavailable');
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
  // Software projection fallback keeps real 3D coordinates, depth and camera;
  // no spurious scientific values are introduced when WebGL is unavailable.
  try{state.ctx=canvas.getContext('2d');state.backend='perspective-fallback'}catch{}
  if(!state.ctx){state.backend='unavailable';$('networkStats').textContent='当前浏览器无法创建三维渲染上下文';}
 }
}
function listLayers(g){
 const counts=Array.from({length:12},()=>0);
 g.nodes.forEach(n=>{const l=decodeLayer(n.name);if(l>=0&&l<12)counts[l]++});return counts;
}
function opColor(node){
 const c=light[node.op]||[.39,.61,.67];
 const active=state.probeNames.get(node.id),value=active!==undefined?state.observed.get(active):undefined;
 // A single exact tensor's producer node may be marked; do NOT map layer
 // averages to non-probed operators, which would fabricate response.
 if(value&&value.mean>1e-7)return [.93,.72,.31];
 if(active!==undefined)return [.67,.83,.61];
 return c;
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
 g.nodes.forEach((n,id)=>{const l=decodeLayer(n.name);if(l>=0&&l<12)layerBuckets[l].push(id)});
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
  const n=g.nodes[id],layer=decodeLayer(n.name),bucket=layerBuckets[layer]||[],part=bucket.indexOf(id);
  const p=nodePosition(id,part,bucket.length,layer,focus);
  positions.push({id,layer,op:n.op,name:n.name,inputs:n.inputs,outputs:n.outputs,p,size:state.probeNames.has(id)?8:5});
 }
 state.nodes=positions;state.edges=g.edges.filter(e=>order.has(e.from)&&order.has(e.to)).map(e=>({a:order.get(e.from),b:order.get(e.to),from:e.from,to:e.to,tensor:e.tensor}));
 state.layoutSerial++;paint();updateUI();
}
function setGraph(g){
 if(!g||!Array.isArray(g.nodes)||!Array.isArray(g.edges))return;
 if(g!==state.graph){state.graph=g;state.graphNodes=g.nodes.length;state.graphEdges=g.edges.length;
  state.probeNames.clear();
  g.nodes.forEach((n,i)=>n.outputs.forEach(name=>{
   const m=name.match(/\/vit\/encoder\/layer\.(\d+)\/attention\/attention\/value\/MatMul_output_0$/);
   if(m)state.probeNames.set(i,Number(m[1]));
  }));
  state.allLayerCounts=listLayers(g);state.mode='overview';state.selected=null;state.yaw=-.3;state.pitch=.16;state.distance=15;makeView();
 }
}
function setObservations(results){
 state.observed.clear();
 for(const l of results||[]){if(l&&Number.isInteger(l.layer)&&Number.isFinite(l.summary?.mean))state.observed.set(l.layer,l.summary)}
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
function uniforms(point){
 const {gl,uniform}=state.gl,w=canvas.width,h=canvas.height;
 gl.uniform1f(uniform.uYaw,state.yaw);gl.uniform1f(uniform.uPitch,state.pitch);gl.uniform1f(uniform.uDistance,state.distance);
 gl.uniform1f(uniform.uScale,3.6);gl.uniform1f(uniform.uAspect,w/h);gl.uniform1f(uniform.uPoint,point?1:0);
}
function pack(arr){const data=new Float32Array(arr.length*7);for(let i=0;i<arr.length;i++){const n=arr[i],idx=i*7;data.set([...n.p,...n.color,n.size],idx)}return data}
function paint(){
 const nodes=state.nodes;
 if(state.backend==='webgl2'){
  const {gl,buffer}=state.gl,ratio=window.devicePixelRatio||1;
  const w=Math.min(1500,Math.max(2,Math.round(canvas.clientWidth*ratio))),h=Math.min(850,Math.max(2,Math.round(canvas.clientHeight*ratio)));
  if(w!==canvas.width||h!==canvas.height){canvas.width=w;canvas.height=h}
  gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.025,.085,.12,1);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(state.gl.program);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  if(state.edges.length){
   const lines=[];for(const e of state.edges){const a=nodes[e.a],b=nodes[e.b];lines.push({p:a.p,color:[.20,.36,.42],size:1},{p:b.p,color:[.20,.36,.42],size:1})}
   gl.bufferData(gl.ARRAY_BUFFER,pack(lines),gl.DYNAMIC_DRAW);uniforms(false);gl.drawArrays(gl.LINES,0,lines.length);
  }
  if(nodes.length){
   const points=nodes.map(n=>({...n,color:state.selected===n.id?[1,.92,.43]:opColor(n),size:state.selected===n.id?11:n.size}));
   gl.bufferData(gl.ARRAY_BUFFER,pack(points),gl.DYNAMIC_DRAW);uniforms(true);gl.drawArrays(gl.POINTS,0,points.length);
  }
 }else if(state.ctx){
  const ctx=state.ctx;ctx.fillStyle='#061923';ctx.fillRect(0,0,canvas.width,canvas.height);
  for(const e of state.edges){const a=screenPoint(nodes[e.a].p),b=screenPoint(nodes[e.b].p);ctx.strokeStyle='#325662';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
  const pts=nodes.map(n=>({...n,q:screenPoint(n.p)})).sort((a,b)=>a.q.depth-b.q.depth);
  for(const n of pts){const c=state.selected===n.id?[1,.9,.36]:opColor(n);ctx.fillStyle='rgb('+c.map(x=>Math.round(x*255)).join(',')+')';ctx.beginPath();ctx.arc(n.q.x,n.q.y,state.selected===n.id?6.5:4,0,Math.PI*2);ctx.fill()}
 }
}
function screenPoint(p){
 const ca=Math.cos(state.yaw),sa=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
 const x=p[0]*ca+p[2]*sa,z=-p[0]*sa+p[2]*ca,y=p[1]*cp-z*sp,dz=p[1]*sp+z*cp,depth=Math.max(1,state.distance-dz);
 const aspect=canvas.width/canvas.height;
 return{x:(x*3.6/(aspect*depth)+1)*canvas.clientWidth/2,y:(1-y*3.6/depth)*canvas.clientHeight/2,depth};
}
function nearest(x,y,max=17){
 let best=null,d=max*max;
 for(const n of state.nodes){const p=screenPoint(n.p),dd=(p.x-x)**2+(p.y-y)**2;if(dd<d){d=dd;best=n}}
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
 if(state.mode==='overview')$('networkEnter').disabled=false;
 if(typeof window.SCEPT_NETWORK3D_LAYER_SELECTED==='function')window.SCEPT_NETWORK3D_LAYER_SELECTED(state.layer);
 paint();updateUI();return true;
}
function updateUI(){
 $('networkOverview').disabled=state.mode==='overview';
 $('networkEnter').disabled=!state.graph||state.mode==='layer'||!state.allLayerCounts[state.layer];
 $('networkDepth').textContent=state.mode==='layer'?'第 '+state.layer+' 层 · 算子详细视图':'完整 Transformer · 三维结构总览';
 $('networkStats').textContent=state.graph?
 (state.mode==='layer'?'第 '+state.layer+' 层，真实算子 '+state.allLayerCounts[state.layer]+' 个 · ':'实际 ONNX 结构 · ')+
 '显示 '+state.nodes.length+'/'+state.graphNodes+' 算子 · '+state.edges.length+'/'+state.graphEdges+' 条直接数据依赖 · '+(state.backend==='webgl2'?'WebGL 3D':'透视降级模式')
 :'等待获取真实 ONNX 计算结构';
 if(state.mode==='layer'&&state.selected===null)$('networkNodeDetails').textContent='当前显示第 '+state.layer+' 层全部已解析算子。单击节点查看真实张量输入输出；连线仅代表 ONNX 定义的直接依赖。';
}
function bind(){
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
 setGraph,setObservations,highlightLayer:setLayer,enterLayer,overview,selectNode,
 getSnapshot:()=>({backend:state.backend,mode:state.mode,layer:state.layer,yaw:state.yaw,pitch:state.pitch,distance:state.distance,displayNodes:state.nodes.length,displayEdges:state.edges.length,totalNodes:state.graphNodes,totalEdges:state.graphEdges,selected:state.selected,layoutSerial:state.layoutSerial,pickTarget:state.nodes.length?{id:state.nodes[Math.floor(state.nodes.length/2)].id,...screenPoint(state.nodes[Math.floor(state.nodes.length/2)].p)}:null})
};
})();
