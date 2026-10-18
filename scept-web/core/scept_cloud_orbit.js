'use strict';
/* SCEPT point-cloud CAMERA ONLY: user-controlled 3D orbit/zoom/pan.
 * The cloud geometry is a layout of measured units, not model topology.
 * Camera motion never alters an underlying NAF, unit, or inference output.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_CLOUD_ORBIT=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='6.5.1',DEFAULT=Object.freeze({yaw:0.48,pitch:0.28,zoom:1.35,panX:0,panY:0});
 const MIN_ZOOM=.35,MAX_ZOOM=5,MAX_PITCH=Math.PI/2-.08;
 const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 function newCamera({redraw=()=>{},onTap=()=>{}}={}){
  if(typeof redraw!=='function'||typeof onTap!=='function')throw Error('Camera callbacks must be functions');
  let state={...DEFAULT},canvas=null,pointers=new Map(),pinch=null,dragged=false,scheduled=false;
  function notify(){
   if(scheduled)return;scheduled=true;
   if(typeof requestAnimationFrame==='function')
    requestAnimationFrame(()=>{scheduled=false;redraw()});
   else{scheduled=false;redraw()}
  }
  function snapshot(){return Object.freeze({...state})}
  function reset(){state={...DEFAULT};notify()}
  function rotate(dx,dy){
   if(!finite(dx)||!finite(dy))return;
   state.yaw=(state.yaw+dx*0.008)%(2*Math.PI);
   state.pitch=clamp(state.pitch+dy*0.008,-MAX_PITCH,MAX_PITCH);
   notify();
  }
  function pan(dx,dy){
   if(!finite(dx)||!finite(dy))return;
   state.panX=clamp(state.panX+dx,-5000,5000);
   state.panY=clamp(state.panY+dy,-5000,5000);
   notify();
  }
  function setZoom(next,point){
   if(!finite(next))return;
   const old=state.zoom,z=clamp(next,MIN_ZOOM,MAX_ZOOM);
   if(point&&finite(point.x)&&finite(point.y)&&finite(point.width)&&finite(point.height)){
    const mx=point.x-point.width/2,my=point.y-point.height/2;
    state.panX=clamp(mx-(mx-state.panX)*z/old,-5000,5000);
    state.panY=clamp(my-(my-state.panY)*z/old,-5000,5000);
   }
   state.zoom=z;notify();
  }
  function zoomBy(factor){if(finite(factor)&&factor>0)setZoom(state.zoom*factor)}
  function nudgeYaw(delta){if(finite(delta)){state.yaw=(state.yaw+delta)%(Math.PI*2);notify()}}
  // Standard Y-up world, positive z further from camera. Native coordinate
  // system comes from the display layout; underlying tensor role is unchanged.
  function project(x,y,z,w,h){
   if(![x,y,z,w,h].every(finite)||w<=0||h<=0)throw Error('Invalid camera point or viewport');
   const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),
    cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
   const rx=x*cy+z*sy,rz=z*cy-x*sy;
   const ry=y*cp-rz*sp,depth=rz*cp+y*sp;
   const perspective=clamp(620/(620+depth),.3,3);
   return{x:w/2+state.panX+rx*state.zoom*perspective,
    y:h/2+state.panY+ry*state.zoom*perspective,
    depth,scale:state.zoom*perspective};
  }
  function bounds(){const b=canvas.getBoundingClientRect();
   return {rect:b,width:canvas.width/(typeof devicePixelRatio==='number'?
    Math.min(2,devicePixelRatio||1):1),height:canvas.height/(typeof devicePixelRatio==='number'?
    Math.min(2,devicePixelRatio||1):1)}}
  function local(ev){
   const b=bounds();
   return{x:(ev.clientX-b.rect.left)*b.width/b.rect.width,
    y:(ev.clientY-b.rect.top)*b.height/b.rect.height,
    width:b.width,height:b.height};
  }
  function metric(){
   if(pointers.size!==2)return null;
   const a=[...pointers.values()],cx=(a[0].x+a[1].x)/2,cy=(a[0].y+a[1].y)/2;
   return {cx,cy,distance:Math.hypot(a[0].x-a[1].x,a[0].y-a[1].y)};
  }
  function down(ev){
   if(ev.pointerType==='mouse'&&ev.button!==0&&ev.button!==2)return;
   if(!canvas)return;
   const pos=local(ev);
   pointers.set(ev.pointerId,{x:pos.x,y:pos.y,startX:pos.x,startY:pos.y,
     moved:false,pan:ev.button===2||ev.shiftKey});
   if(pointers.size>1){dragged=true;pinch=metric()}
   else{dragged=false;pinch=null}
   try{canvas.setPointerCapture(ev.pointerId)}catch(_){}
   if(ev.pointerType!=='mouse'||ev.button===2)ev.preventDefault();
  }
  function move(ev){
   if(!pointers.has(ev.pointerId)||!canvas)return;
   const p=pointers.get(ev.pointerId),q=local(ev),
    dx=q.x-p.x,dy=q.y-p.y;
   p.x=q.x;p.y=q.y;
   if(Math.hypot(q.x-p.startX,q.y-p.startY)>4){p.moved=true;dragged=true}
   if(pointers.size>=2){
    const current=metric();
    if(current&&pinch&&pinch.distance>2&&current.distance>2){
     pan(current.cx-pinch.cx,current.cy-pinch.cy);
     setZoom(state.zoom*current.distance/pinch.distance,{
      x:current.cx,y:current.cy,width:q.width,height:q.height});
    }
    pinch=current;ev.preventDefault();return;
   }
   if(p.moved&&(dx!==0||dy!==0)){
    if(p.pan||ev.shiftKey)pan(dx,dy);
    else rotate(dx,dy);
    ev.preventDefault();
   }
  }
  function end(ev,cancelled){
   const p=pointers.get(ev.pointerId);
   if(!p)return;
   const pos=canvas&&local(ev);
   pointers.delete(ev.pointerId);
   try{if(canvas?.hasPointerCapture?.(ev.pointerId))canvas.releasePointerCapture(ev.pointerId)}catch(_){}
   if(!cancelled&&!p.moved&&!dragged&&pointers.size===0&&pos)onTap(pos);
   if(pointers.size===1){const left=pointers.values().next().value;
    left.startX=left.x;left.startY=left.y;left.moved=true}
   pinch=metric();
   if(!pointers.size)dragged=false;
  }
  function wheel(ev){
   if(!canvas)return;
   ev.preventDefault();
   const q=local(ev),delta=ev.deltaMode===1?ev.deltaY*16:ev.deltaMode===2?ev.deltaY*100:ev.deltaY;
   setZoom(state.zoom*Math.exp(-clamp(delta,-250,250)*.0018),q);
  }
  function key(ev){
   if(ev.key==='+'||ev.key==='='){zoomBy(1.18);ev.preventDefault()}
   else if(ev.key==='-'||ev.key==='_'){zoomBy(1/1.18);ev.preventDefault()}
   else if(ev.key==='ArrowLeft'){rotate(-19,0);ev.preventDefault()}
   else if(ev.key==='ArrowRight'){rotate(19,0);ev.preventDefault()}
   else if(ev.key==='ArrowUp'){rotate(0,-19);ev.preventDefault()}
   else if(ev.key==='ArrowDown'){rotate(0,19);ev.preventDefault()}
   else if(ev.key==='Home'){reset();ev.preventDefault()}
  }
  function bind(element){
   if(!element||typeof element.addEventListener!=='function'||canvas)
    throw Error('Bind exactly one valid orbit canvas');
   canvas=element;
   canvas.addEventListener('pointerdown',down);
   canvas.addEventListener('pointermove',move);
   canvas.addEventListener('pointerup',ev=>end(ev,false));
   canvas.addEventListener('pointercancel',ev=>end(ev,true));
   canvas.addEventListener('lostpointercapture',ev=>end(ev,true));
   canvas.addEventListener('wheel',wheel,{passive:false});
   canvas.addEventListener('contextmenu',ev=>ev.preventDefault());
   canvas.addEventListener('keydown',key);
   return snapshot();
  }
  return Object.freeze({VERSION,bind,snapshot,reset,rotate,pan,zoomBy,setZoom,nudgeYaw,project});
 }
 return Object.freeze({VERSION,DEFAULT,MIN_ZOOM,MAX_ZOOM,newCamera});
});
