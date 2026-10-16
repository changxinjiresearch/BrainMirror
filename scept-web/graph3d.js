'use strict';
/* SCEPT visual topology: all edges are model-defined tensor dependencies.
   Coordinates are a 3D visualization layout, not anatomical or causal evidence. */
(function(){
let graph=null,observed=null,nodes=[],links=[],yaw=.25,pitch=.24,points=[],moving=null,hover=null;
const canvas=document.getElementById('graph3dCanvas');if(!canvas)return;
const ctx=canvas.getContext('2d');
function layout(g){
 const N=Math.min(g.nodes.length,160),depth=new Array(N).fill(0),adj=Array.from({length:N},()=>[]);
 const parents=Array.from({length:N},()=>[]);
 for(const e of g.edges)if(e.from<N&&e.to<N){adj[e.from].push(e.to);parents[e.to].push(e.from)}
 for(let i=0;i<N;i++){
  if(parents[i].length)depth[i]=Math.min(120,Math.max(...parents[i].map(k=>depth[k]+1)));
 }
 const maxDepth=Math.max(1,...depth);
 const levels=new Map();
 for(let i=0;i<N;i++){const k=depth[i];if(!levels.has(k))levels.set(k,[]);levels.get(k).push(i)}
 nodes=[];
 for(let i=0;i<N;i++){
  const peers=levels.get(depth[i]),j=peers.indexOf(i),angle=2*Math.PI*j/Math.max(1,peers.length);
  const radial=.48+.16*(i%4);
  nodes.push({id:i,x:-1.4+2.8*depth[i]/maxDepth,y:radial*Math.cos(angle),z:radial*Math.sin(angle),name:g.nodes[i].name,op:g.nodes[i].op,outputs:g.nodes[i].outputs});
 }
 links=g.edges.filter(e=>e.from<N&&e.to<N).slice(0,2000);
}
function project(n){
 let [x,y,z]=[n.x,n.y,n.z];
 const a=Math.cos(yaw),b=Math.sin(yaw),c=Math.cos(pitch),d=Math.sin(pitch);
 const rx=x*a-z*b,rz=x*b+z*a,ry=y*c-rz*d,depth=y*d+rz*c;
 const scale=300/(3.8+depth);
 return{x:canvas.width/2+rx*scale,y:canvas.height/2+ry*scale,depth};
}
function delta(node){
 if(!observed)return null;
 let result=null;
 for(const name of node.outputs){if(!(name in observed))continue;const o=observed[name];if(!o?.before||!o?.after)continue;
  let sum=0,N=Math.min(o.before.length,10000);for(let i=0;i<N;i++)sum+=Math.abs(o.after[i]-o.before[i]);
  result=Math.max(result||0,sum/Math.max(1,N));
 }
 return result;
}
function paint(){
 ctx.fillStyle='#071a24';ctx.fillRect(0,0,canvas.width,canvas.height);
 const grd=ctx.createLinearGradient(0,0,canvas.width,canvas.height);grd.addColorStop(0,'rgba(48,110,121,.08)');grd.addColorStop(1,'rgba(48,110,121,.0)');
 ctx.fillStyle=grd;ctx.fillRect(0,0,canvas.width,canvas.height);
 if(!graph){ctx.fillStyle='#8da9ae';ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillText('Import an ONNX model to inspect the real operator graph',canvas.width/2,canvas.height/2);return}
 points=nodes.map(n=>({...project(n),node:n}));
 ctx.lineWidth=1;
 for(const edge of links){const a=points[edge.from],b=points[edge.to];if(!a||!b)continue;ctx.strokeStyle='rgba(98,166,167,.20)';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
 const ordered=points.slice().sort((a,b)=>b.depth-a.depth);
 for(const p of ordered){
  const d=delta(p.node),selected=hover===p.node.id;
  ctx.beginPath();ctx.arc(p.x,p.y,selected?7:4.3,0,2*Math.PI);
  ctx.fillStyle=d===null?'#739aa8':d>1e-7?'#5ae7c5':'#b1bfc1';ctx.fill();
  if(selected){ctx.strokeStyle='#fff1b2';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#e0f9f2';ctx.font='12px system-ui';ctx.textAlign='left';ctx.fillText((p.node.name||p.node.op||'node').slice(0,42),p.x+10,p.y-7)}
 }
 ctx.textAlign='left';ctx.fillStyle='#88adb1';ctx.font='11px system-ui';
 ctx.fillText('3D LAYOUT · OPERATOR NODES · REAL TENSOR DEPENDENCIES',18,24);
 ctx.fillText('Green: measured output changed · Grey-blue: no output probe · Drag to rotate',18,canvas.height-16);
}
function inspectPoint(x,y){
 let best=null,d=17**2;for(const p of points){const d2=(p.x-x)**2+(p.y-y)**2;if(d2<d){d=d2;best=p.node}}
 hover=best?.id??null;
 if(best){
  const n=best.node,ident='Node '+n.id+' '+n.op+' · '+(n.name||'(unnamed)')+
   ' · Input tensors: '+graph.nodes[n.id].inputs.join(', ')+
   ' · Output tensors: '+n.outputs.join(', ')+
   ' · Delta: '+(delta(n)===null?'not exposed as ONNX output':delta(n).toExponential(4));
  canvas.title=ident;
  const details=document.getElementById('graphDetails');if(details)details.textContent=ident;
 }else canvas.title='Drag to rotate real ONNX operator graph';
 paint();
}
canvas.addEventListener('pointerdown',e=>{moving={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(moving){yaw+=(e.clientX-moving.x)*.009;pitch=Math.max(-1.3,Math.min(1.3,pitch+(e.clientY-moving.y)*.009));moving={x:e.clientX,y:e.clientY};paint()}else{const r=canvas.getBoundingClientRect();inspectPoint((e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height)}});
canvas.addEventListener('pointerup',()=>{moving=null});
window.SCEPT_GRAPH3D={render(g,o){graph=g;observed=o||null;if(g)layout(g);paint()}};
paint();
})();