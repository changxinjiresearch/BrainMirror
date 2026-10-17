'use strict';
/* SCEPT visual topology: all edges are model-defined tensor dependencies.
   Coordinates are a 3D visualization layout, not anatomical or causal evidence. */
(function(){
let graph=null,observed=null,nodes=[],links=[],yaw=.25,pitch=.24,points=[],moving=null,hover=null;
const canvas=document.getElementById('graph3dCanvas');if(!canvas)return;
const ctx=canvas.getContext('2d');
function layout(g){
 const total=g.nodes.length,chosen=new Set();
 if(total<=180){for(let i=0;i<total;i++)chosen.add(i)}
 else{
  for(let i=0;i<16;i++){chosen.add(i);chosen.add(total-1-i)}
  const groups=9,perGroup=13;
  for(let k=0;k<groups;k++){
   const center=Math.round((k+.5)*total/groups),left=Math.max(0,Math.min(total-perGroup,center-Math.floor(perGroup/2)));
   for(let i=left;i<left+perGroup;i++)chosen.add(i);
  }
  const probeNames=new Set(observed?Object.keys(observed):[]);
  for(let i=0;i<total;i++)if(g.nodes[i].outputs.some(n=>probeNames.has(n))){
    for(let j=Math.max(0,i-5);j<=Math.min(total-1,i+5);j++)chosen.add(j);
    for(const edge of g.edges)if(edge.to===i||edge.from===i){chosen.add(edge.from);chosen.add(edge.to)}
  }
 }
 const ids=[...chosen].sort((a,b)=>a-b),idToView=new Map(ids.map((id,i)=>[id,i]));
 const bands=new Map();
 for(const id of ids){const band=Math.floor(id/Math.max(1,total/11));if(!bands.has(band))bands.set(band,[]);bands.get(band).push(id)}
 nodes=ids.map(id=>{
  const peers=bands.get(Math.floor(id/Math.max(1,total/11))),p=peers.indexOf(id);
  const angle=2*Math.PI*p/Math.max(1,peers.length),radial=.44+.17*(id%4);
  const n=g.nodes[id];
  return{id,x:-1.42+2.84*id/Math.max(1,total-1),y:radial*Math.cos(angle),z:radial*Math.sin(angle),name:n.name,op:n.op,outputs:n.outputs};
 });
 links=g.edges.filter(e=>idToView.has(e.from)&&idToView.has(e.to)).slice(0,2000).map(e=>({from:idToView.get(e.from),to:idToView.get(e.to)}));
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
 if(!graph){ctx.fillStyle='#8da9ae';ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillText('正在准备真实模型计算图…',canvas.width/2,canvas.height/2);return}
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
 ctx.fillText('真实 ONNX 算子采样 · 三维布局仅作展示',18,24);
 ctx.fillText('绿色：已观测到差异 · 灰蓝：未开放观测点 · 拖动旋转',18,canvas.height-16);
}
function inspectPoint(x,y){
 let best=null,d=17**2;for(const p of points){const d2=(p.x-x)**2+(p.y-y)**2;if(d2<d){d=d2;best=p.node}}
 hover=best?.id??null;
 if(best){
  const n=best.node,ident='计算节点 #'+n.id+' · '+n.op+' · '+(n.name||'(无名称)')+
   ' · 输入张量：'+graph.nodes[n.id].inputs.join(', ')+
   ' · 输出张量：'+n.outputs.join(', ')+
   ' · 数值差异：'+(delta(n)===null?'尚未开放为可测观测点':delta(n).toExponential(4));
  canvas.title=ident;
  const details=document.getElementById('graphDetails');if(details)details.textContent=ident;
 }else canvas.title='拖动可旋转真实 ONNX 计算图';
 paint();
}
canvas.addEventListener('pointerdown',e=>{moving={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(moving){yaw+=(e.clientX-moving.x)*.009;pitch=Math.max(-1.3,Math.min(1.3,pitch+(e.clientY-moving.y)*.009));moving={x:e.clientX,y:e.clientY};paint()}else{const r=canvas.getBoundingClientRect();inspectPoint((e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height)}});
canvas.addEventListener('pointerup',()=>{moving=null});
window.SCEPT_GRAPH3D={render(g,o){graph=g;observed=o||null;if(g)layout(g);paint()}};
paint();
})();