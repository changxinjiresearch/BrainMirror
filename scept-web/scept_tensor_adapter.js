'use strict';
/* SCEPT real model tensor adapter. Strict layout checks; no inferred values.
   Every token/channel metric is recomputed from the original ONNX data. */
(function(){
const N=197,C=192,PATCH_GRID=14;
function parseLayer(layer){
 const dims=layer.shape,expected=N*C;
 if(!Array.isArray(dims)||dims.length!==3||dims[0]!==1||dims[1]!==N||dims[2]!==C)
   throw Error('观测张量布局不匹配，预期 [1,197,192]，实际 '+JSON.stringify(dims)+'. 已停止细粒度映射，绝不猜测 Token 顺序。');
 if(!Array.isArray(layer.before)||!Array.isArray(layer.after)||layer.before.length!==expected||layer.after.length!==expected)
   throw Error('当前第 '+layer.layer+' 层的实际张量元素数不符合 197 × 192');
 const tokens=new Array(N),channelMax=0;
 for(let t=0;t<N;t++){
  let sumA=0,sumB=0,sumAbs=0,sumSq=0,maxDelta=0,signed=0,changed=0;
  for(let c=0;c<C;c++){
   const i=t*C+c,a=layer.before[i],b=layer.after[i],d=b-a,ad=Math.abs(d);
   if(!Number.isFinite(a)||!Number.isFinite(b))throw Error('观测张量存在非有限值');
   sumA+=Math.abs(a);sumB+=Math.abs(b);sumAbs+=ad;sumSq+=d*d;signed+=d;
   maxDelta=Math.max(maxDelta,ad);if(ad>1e-7)changed++;
  }
  tokens[t]={layer:layer.layer,token:t,isCls:t===0,
    patch:t===0?null:t-1,row:t===0?null:Math.floor((t-1)/PATCH_GRID),
    col:t===0?null:(t-1)%PATCH_GRID,
    before:sumA/C,after:sumB/C,mae:sumAbs/C,rms:Math.sqrt(sumSq/C),
    maxDelta,meanSigned:signed/C,changed,total:C};
 }
 return{layer:layer.layer,probe:layer.probe,shape:dims.slice(),tokenCount:N,channelCount:C,tokens};
}
function build(layers){
 if(!Array.isArray(layers)||!layers.length)throw Error('无实际观测张量');
 const result=layers.map(parseLayer);
 const layersSeen=new Set(result.map(x=>x.layer));
 if(layersSeen.size!==result.length)throw Error('存在重复层级记录');
 return {schema:'scept.activity.token-channel.v1',tokenCount:N,channelCount:C,patchGrid:PATCH_GRID,layers:result};
}
function channelAt(layers,layer,token){
 if(!Number.isInteger(token)||token<0||token>=N)throw Error('Token index out of bounds');
 const raw=layers.find(x=>x.layer===layer);if(!raw)throw Error('层级未开放观测');
 parseLayer(raw); // validate shape and numbers against observed model first
 const arr=[];for(let c=0;c<C;c++){
  const i=token*C+c,a=raw.before[i],b=raw.after[i];
  arr.push({channel:c,before:a,after:b,delta:b-a,absDelta:Math.abs(b-a)});
 }
 return arr;
}
function mapInput(base,edited){
 const SIDE=224,PIXELS=SIDE*SIDE;
 if(!base||!edited||base.length!==3*PIXELS||edited.length!==3*PIXELS)throw Error('输入维度必须为三个 224×224 通道');
 let changed=0,maxDelta=0,sumDelta=0;
 const patch=Array.from({length:196},(_,i)=>({patch:i,token:i+1,row:Math.floor(i/14),col:i%14,changed:0,meanAbs:0,maxAbs:0}));
 for(let i=0;i<PIXELS;i++){
  const x=i%SIDE,y=Math.floor(i/SIDE),p=patch[Math.floor(y/16)*14+Math.floor(x/16)];
  let local=0;
  for(let c=0;c<3;c++){
   const d=Math.abs(edited[c*PIXELS+i]-base[c*PIXELS+i]);if(!Number.isFinite(d))throw Error('MRI 输入出现非有限值');
   local=Math.max(local,d);
   p.meanAbs+=d/(3*256);p.maxAbs=Math.max(p.maxAbs,d);sumDelta+=d/(3*PIXELS);
  }
  if(local>1e-8){changed++;p.changed++}
 }
 const affected=patch.filter(p=>p.changed>0).map(p=>p.patch);
 return{shape:[1,3,224,224],changedPixels:changed,changedPatches:affected.length,
  patches:patch,affectedPatchIds:affected,meanAbs:sumDelta,
  empty:changed===0,orientation:'Model input is voxel x and y increasing; axial display y-axis is reversed. CLS token has no spatial patch.'};
}
window.SCEPT_TENSOR_ADAPTER={build,channelAt,mapInput,N,C};
})();
