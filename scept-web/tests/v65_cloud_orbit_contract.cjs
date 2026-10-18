'use strict';
/* Camera SOFTWARE gesture acceptance only; model and NAF values are never mocked here.
 * Genuine ONNX run, preserving Run Record, is tested separately in Chrome.
 */
const assert=require('node:assert/strict'),Orbit=require('../core/scept_cloud_orbit.js');
const dispatch=new Map(),captures=new Set(),clicks=[];
let frames=0;
const fake={
 width:800,height:300,getBoundingClientRect:()=>({left:40,top:20,width:800,height:300}),
 addEventListener:(event,fn)=>dispatch.set(event,fn),
 setPointerCapture:id=>captures.add(id),
 hasPointerCapture:id=>captures.has(id),
 releasePointerCapture:id=>captures.delete(id)
};
const controller=Orbit.newCamera({redraw:()=>frames++,onTap:p=>clicks.push(p)});
controller.bind(fake);
function event(type,{pointerId=1,clientX=440,clientY=170,button=0,pointerType='mouse',
 shiftKey=false,deltaY=0,deltaMode=0,key=''}={}){
 const e={pointerId,clientX,clientY,button,pointerType,shiftKey,deltaY,deltaMode,key,
  prevented:false,preventDefault(){this.prevented=true}};
 const f=dispatch.get(type);assert(f,'Missing '+type+' handler');f(e);return e;
}
const initial=controller.snapshot();
assert(Math.abs(initial.yaw-.48)<1e-9);
assert(Math.abs(initial.pitch-.28)<1e-9);
const p=controller.project(60,20,40,800,300);
assert(Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.scale>0);
event('pointerdown',{pointerId:1});
event('pointermove',{pointerId:1,clientX:510,clientY:210});
event('pointerup',{pointerId:1,clientX:510,clientY:210});
const afterOrbit=controller.snapshot();
assert(afterOrbit.yaw!==initial.yaw&&afterOrbit.pitch!==initial.pitch,
 'Dragging must rotate horizontally AND vertically');
assert.equal(clicks.length,0,'Drag must not select any NAF unit');
const panBefore=controller.snapshot();
event('pointerdown',{pointerId:2,clientX:400,clientY:200,shiftKey:true});
event('pointermove',{pointerId:2,clientX:475,clientY:235,shiftKey:true});
event('pointerup',{pointerId:2,clientX:475,clientY:235,shiftKey:true});
const panned=controller.snapshot();
assert(panned.panX>panBefore.panX&&panned.panY>panBefore.panY,'Shift+drag pans canvas');
assert.equal(panned.yaw,panBefore.yaw);
event('pointerdown',{pointerId:4,clientX:450,clientY:200,button:2});
event('pointermove',{pointerId:4,clientX:490,clientY:200,button:2});
event('pointerup',{pointerId:4,clientX:490,clientY:200,button:2});
assert(controller.snapshot().panX>panned.panX,'Right button pans');
const w=event('wheel',{deltaY:-250,clientX:700,clientY:210});
assert(w.prevented,'Wheel must prevent page scroll only while cursor is over canvas');
assert(controller.snapshot().zoom>initial.zoom,'Wheel zoom in');
const zoomBeforePinch=controller.snapshot().zoom;
event('pointerdown',{pointerId:10,clientX:400,clientY:180,pointerType:'touch'});
event('pointerdown',{pointerId:11,clientX:500,clientY:180,pointerType:'touch'});
event('pointermove',{pointerId:11,clientX:550,clientY:180,pointerType:'touch'});
event('pointerup',{pointerId:11,clientX:550,clientY:180,pointerType:'touch'});
event('pointerup',{pointerId:10,clientX:400,clientY:180,pointerType:'touch'});
assert(controller.snapshot().zoom>zoomBeforePinch,'Two-finger pinch must zoom');
assert.equal(clicks.length,0,'Pinch must not spuriously click');
event('pointerdown',{pointerId:22,clientX:460,clientY:180});
event('pointerup',{pointerId:22,clientX:460,clientY:180});
assert.equal(clicks.length,1,'Short stationary tap must select the actual target point');
controller.setZoom(999);assert.equal(controller.snapshot().zoom,Orbit.MAX_ZOOM);
controller.setZoom(1e-7);assert.equal(controller.snapshot().zoom,Orbit.MIN_ZOOM);
event('keydown',{key:'Home'});
const home=controller.snapshot();
assert.deepEqual(home,initial,'Home must exactly reset all view parameters');
event('keydown',{key:'+'});assert(controller.snapshot().zoom>home.zoom);
event('keydown',{key:'ArrowRight'});assert(controller.snapshot().yaw!==home.yaw);
assert(frames>=9,'Camera should redraw when the viewport changes');
console.log('PASS SCEPT NAF camera: two-axis free drag orbit, wheel/keyboard/pinch zoom, shift/right pan, tap-no-drag disambiguation, bounds and full reset');
