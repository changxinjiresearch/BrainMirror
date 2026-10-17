'use strict';
/* Read-only real token-level activation viewer. 3D coordinates are layout,
 not neurons or inferred attention connections. */
(function(){
const $=id=>document.getElementById(id),cv=$('token3dCanvas'),ctx=cv.getContext('2d');
const s={raw:null,data:null,kind:'operators',scope:'all',layer:6,selected:null,patch:null,patchMap:null,
 yaw:-.44,pitch:.25,dist:14,scaleA:null,scaleD:null,runRevision:null,version:0,mouse:null,channels:null};
const clamp=(n,a,b)=>Math.max(a,Math.min(n,b)),fmt=n=>Math.abs(n)<.0001&&n!==0?n.toExponential(3):n.toFixed(5);
const mode=()=>SCEPT_NETWORK3D.getSnapshot().visualMode;
function nodes(){
 if(!s.data)return[];
 const nodes=[];
 for(const l of s.data.layers){
  if(s.scope==='layer'&&l.layer!==s.layer)continue;
  for(const t of l.tokens)nodes.push({layer:l.layer,token:t.token,info:t,
    pos:t.isCls?[(s.scope==='all'?l.layer-6:0)*1.25,2.8,0]:
    [(s.scope==='all'?l.layer-6:0)*1.25,(t.row-6.5)*.38,(t.col-6.5)*.38]});
 }
 return nodes;
}
function project(p,pane,count){
 const a=Math.cos(s.yaw),b=Math.sin(s.yaw),c=Math.cos(s.pitch),d=Math.sin(s.pitch);
 const x=p[0]*a+p[2]*b,z=-p[0]*b+p[2]*a,y=p[1]*c-z*d,dep=p[1]*d+z*c;
 const ww=(cv.clientWidth||960)/count,hh=cv.clientHeight||410,k=4.5/Math.max(1,s.dist-dep);
 return{x:pane*ww+ww*.5+x*k*ww*.47,y:hh*.5-y*k*hh*.47,depth:dep};
}
function displayValue(t,m){return m==='before'?t.before:m==='after'?t.after:t.rms}
function style(t,m){
 const v=displayValue(t,m),scale=m==='delta'?s.scaleD:s.scaleA;
 const z=v/Math.max(1e-12,v+(scale||1)),zero=m==='delta'&&v<=1e-9;
 return {r:zero?2.8:3.5+8*z,
  fill:zero?'#57737b':m==='delta'?'rgb('+Math.round(60+194*z)+','+Math.round(154+57*z)+','+Math.round(172-91*z)+')':
  'rgb('+Math.round(60+120*z)+','+Math.round(125+112*z)+','+Math.round(151-63*z)+')'};
}
function render(){
 if(s.kind!=='tokens')return;
 const w=cv.clientWidth||960,h=cv.clientHeight||410,ratio=Math.min(2,devicePixelRatio||1);
 if(cv.width!==Math.round(w*ratio)||cv.height!==Math.round(h*ratio)){cv.width=Math.round(w*ratio);cv.height=Math.round(h*ratio)}
 ctx.setTransform(ratio,0,0,ratio,0,0);ctx.fillStyle='#061923';ctx.fillRect(0,0,w,h);
 const modes=mode()==='compare'?['before','after']:[mode()],arr=nodes();
 for(let pane=0;pane<modes.length;pane++){
  const m=modes[pane],width=w/modes.length;ctx.save();ctx.beginPath();ctx.rect(pane*width,0,width,h);ctx.clip();
  const drawn=arr.map(n=>({...n,p:project(n.pos,pane,modes.length)})).sort((a,b)=>a.p.depth-b.p.depth);
  for(const n of drawn){
   const vis=style(n.info,m),active=s.selected?.layer===n.layer&&s.selected?.token===n.token,
     patchSelected=s.patch!==null&&n.token===s.patch+1;
   if(patchSelected){ctx.strokeStyle='#87c4ff';ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(n.p.x,n.p.y,vis.r+5,0,Math.PI*2);ctx.stroke()}
   if(active){ctx.strokeStyle='#ffb862';ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(n.p.x,n.p.y,vis.r+3,0,Math.PI*2);ctx.stroke()}
   ctx.fillStyle=vis.fill;ctx.beginPath();ctx.arc(n.p.x,n.p.y,vis.r,0,Math.PI*2);ctx.fill();
   if(n.info.isCls){ctx.strokeStyle='#d9bbf8';ctx.lineWidth=1;ctx.stroke()}
  }
  ctx.fillStyle='#97c3bd';ctx.font='12px sans-serif';
  ctx.fillText(m==='before'?'A · 原始真实激活':m==='after'?'B · 修改后真实激活':'真实逐 Token RMS 差异 Δ',pane*width+12,20);
  ctx.restore();
 }
 if(modes.length===2){ctx.strokeStyle='#2b656d';ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke()}
 ctx.setTransform(1,0,0,1,0,0);
 const stale=s.runRevision!==null&&SCEPT_MRI.state.revision!==s.runRevision;
 $('tokenDepth').textContent=(s.scope==='all'?'五层概览':'第 '+s.layer+' 层')+' · '+arr.length+' 个真实 Token';
 $('token3dNote').textContent=(stale?'输入已修改，显示的是上一次计算结果；请重新计算。':'当前显示最近一次推理的真实张量数据。')+
 ' 蓝色轮廓：选中输入 Patch；黄色轮廓：选中 Token；没有绘制未经测量的注意力连接。';
}
function choose(layer,token){
 if(!s.data?.layers.some(l=>l.layer===layer)||!Number.isInteger(token)||token<0||token>=197)return false;
 s.selected={layer,token};s.layer=layer;
 const t=s.data.layers.find(l=>l.layer===layer).tokens[token];
 $('tokenDetails').hidden=false;
 $('tokenTitle').textContent='第 '+layer+' 层 · '+(t.isCls?'CLS Token（无空间 Patch）':'Token '+token+' · Patch '+(token-1)+' · 行 '+t.row+' / 列 '+t.col);
 $('tokenStats').textContent='RMS Δ '+fmt(t.rms)+' · MAE Δ '+fmt(t.mae)+' · 变化通道 '+t.changed+'/192';
 s.channels=SCEPT_TENSOR_ADAPTER.channelAt(s.raw,layer,token);renderChannel();render();return true;
}
function renderChannel(){
 if(!s.channels)return;
 const c=$('tokenChannelCanvas'),g=c.getContext('2d'),v=s.channels,ww=c.width,hh=c.height,
 max=Math.max(1e-12,...v.map(x=>x.absDelta)),step=(ww-38)/192,mid=hh/2+9;
 g.fillStyle='#081b25';g.fillRect(0,0,ww,hh);
 g.strokeStyle='#76969d';g.beginPath();g.moveTo(16,mid);g.lineTo(ww-10,mid);g.stroke();
 for(let i=0;i<192;i++){const x=v[i],h=x.absDelta/max*98;g.fillStyle=x.delta>=0?'#5edcc1':'#f3a569';
  g.fillRect(20+i*step,x.delta>=0?mid-h:mid,Math.max(1,step-1),Math.max(1,h))}
 const pick=Number($('tokenChannelSelect').value),a=v[pick];
 g.strokeStyle='#f8cf72';g.beginPath();g.moveTo(20+pick*step,32);g.lineTo(20+pick*step,hh-8);g.stroke();
 g.font='12px sans-serif';g.fillStyle='#b1d7d2';g.fillText('真实 B−A：正值向上，负值向下；192 个通道',16,17);
 $('tokenChannelValue').textContent=String(pick);
 $('tokenChannelInfo').textContent='通道 '+pick+' · 原始 A '+fmt(a.before)+' · 修改后 B '+fmt(a.after)+' · B−A '+fmt(a.delta);
}
function focusPatch(patch){
 s.patch=Number.isInteger(patch)?patch:null;
 if(s.patch!==null&&s.kind==='tokens'&&s.data){choose(s.layer,s.patch+1)}else render();
}
function show(k){
 if(k==='tokens'&&!s.data)return false;
 s.kind=k;
 $('operator3dShell').hidden=k!=='operators';$('token3dShell').hidden=k!=='tokens';
 $('networkOverview').closest('.network-toolbar').hidden=k!=='operators';
 $('tokenDetails').hidden=k!=='tokens'||!s.selected;
 for(const [id,val]of [['viewOperators','operators'],['viewTokens','tokens']]){
  $(id).classList.toggle('active',k===val);$(id).setAttribute('aria-pressed',String(k===val))}
 if(k==='tokens'){SCEPT_NETWORK3D.setVisualMode('delta');render()}
 return true;
}
function update(raw,mriRevision){
 s.raw=raw;s.data=SCEPT_TENSOR_ADAPTER.build(raw);s.runRevision=mriRevision;s.version++;
 if(s.scaleA===null)recalibrate();
 $('viewTokens').disabled=false;
 if(s.selected)choose(s.selected.layer,s.selected.token);
 render();
 return s.data;
}
function recalibrate(){
 if(!s.data)return;
 s.scaleA=Math.max(1e-12,...s.data.layers.flatMap(l=>l.tokens.flatMap(t=>[t.before,t.after])));
 s.scaleD=Math.max(1e-12,...s.data.layers.flatMap(l=>l.tokens.map(t=>t.rms)));
 render();
}
function clear(){s.data=null;s.raw=null;s.selected=null;s.scaleA=null;s.scaleD=null;s.kind='operators';$('viewTokens').disabled=true;show('operators')}
const get=()=>({kind:s.kind,scope:s.scope,mode:mode(),version:s.version,layer:s.layer,selected:s.selected,
 yaw:s.yaw,pitch:s.pitch,distance:s.dist,count:nodes().length,runRevision:s.runRevision,scaleA:s.scaleA,scaleD:s.scaleD,
 layers:s.data?.layers.map(l=>({layer:l.layer,tokens:l.tokens.map(t=>({token:t.token,before:t.before,after:t.after,mae:t.mae,rms:t.rms,changed:t.changed}))}))||[]});
$('viewTokens').disabled=true;
$('viewTokens').onclick=()=>show('tokens');$('viewOperators').onclick=()=>show('operators');
for(const id of ['responseCompare','responseDelta','responseBefore','responseAfter'])$(id).addEventListener('click',render);
$('tokenOverview').onclick=()=>{s.scope='all';s.dist=14;render()};
$('tokenEnter').onclick=()=>{s.scope='layer';s.dist=10;render()};
$('tokenZoomIn').onclick=()=>{s.dist=clamp(s.dist-1.1,4,36);render()};
$('tokenZoomOut').onclick=()=>{s.dist=clamp(s.dist+1.1,4,36);render()};
$('tokenReset').onclick=()=>{s.yaw=-.44;s.pitch=.25;s.dist=14;render()};
$('tokenChannelSelect').oninput=renderChannel;
let mouse=null;
cv.addEventListener('pointerdown',e=>{mouse={x:e.clientX,y:e.clientY,originX:e.clientX,originY:e.clientY,moved:false};cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointermove',e=>{if(!mouse)return;
 const dx=e.clientX-mouse.x,dy=e.clientY-mouse.y;if(Math.abs(e.clientX-mouse.originX)+Math.abs(e.clientY-mouse.originY)>4)mouse.moved=true;
 if(mouse.moved){s.yaw+=dx*.009;s.pitch=clamp(s.pitch+dy*.009,-1.25,1.25);render()}
 mouse.x=e.clientX;mouse.y=e.clientY;
});
cv.addEventListener('pointerup',e=>{if(!mouse)return;const moved=mouse.moved;mouse=null;if(moved)return;
 const rect=cv.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
 const n=mode()==='compare'?2:1,p=x>rect.width/2&&n===2?1:0;
 let nearest=null,dist=350;
 for(const node of nodes()){const pos=project(node.pos,p,n),d=(x-pos.x)**2+(y-pos.y)**2;if(d<dist){dist=d;nearest=node}}
 if(nearest)choose(nearest.layer,nearest.token);
});
cv.addEventListener('wheel',e=>{e.preventDefault();s.dist=clamp(s.dist+(e.deltaY>0?1.2:-1.2),4,36);render()},{passive:false});
cv.addEventListener('dblclick',()=>{s.scope=s.scope==='all'?'layer':'all';render()});
cv.addEventListener('keydown',e=>{if(e.key==='Escape'){s.scope='all';render()}if(e.key==='Enter'){s.scope='layer';render()}});
window.addEventListener('resize',render);
window.SCEPT_ACTIVITY3D={update,clear,show,select:choose,focusPatch,
 syncPatchMap(map,selected){s.patchMap=map;if(selected!==undefined)s.patch=selected;render()},
 highlightLayer(layer){s.layer=Number(layer);render()},
 render,recalibrate,getSnapshot:get};
})();
