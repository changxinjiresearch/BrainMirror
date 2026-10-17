'use strict';
/* U9 on-demand performance budget: no invented benchmark numbers. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_PERFORMANCE=api;})(typeof window!=='undefined'?window:null,function(){
 const fail=(code,s)=>{const e=Error('SCEPT Performance: '+s);e.code=code;throw e};
 const now=()=>typeof performance!=='undefined'&&typeof performance.now==='function'?performance.now():Date.now();
 function estimateFeeds(feeds){
  if(!feeds||typeof feeds!=='object')fail('INVALID_FEEDS','Typed inputs required');
  let bytes=0;
  for(const t of Object.values(feeds)){
   if(!ArrayBuffer.isView(t?.data))fail('INVALID_FEEDS','Typed model inputs only');
   bytes+=t.data.byteLength;
  }
  return bytes;
 }
 function quantile(arr,p){
  const values=arr.slice().sort((a,b)=>a-b),x=(values.length-1)*p,i=Math.floor(x);
  return values[i]+(values[Math.min(i+1,values.length-1)]-values[i])*(x-i);
 }
 async function benchmark({session,feeds,runs=3,warmup=1,maxMeanMs=null,maxInputMiB=128}){
  if(!session?.run||!session?.sha256||!session?.description)fail('INVALID_SESSION','Verified real model required');
  if(!Number.isInteger(runs)||runs<1||runs>15||!Number.isInteger(warmup)||warmup<0||warmup>3)
   fail('RESOURCE_LIMIT','Runs must be [1,15] and warmup [0,3]');
  const bytes=estimateFeeds(feeds),miB=bytes/1048576;
  if(!Number.isFinite(maxInputMiB)||maxInputMiB<=0||miB>maxInputMiB)
   fail('RESOURCE_LIMIT','Model input memory estimate exceeds explicit budget');
  if(maxMeanMs!==null&&(!Number.isFinite(maxMeanMs)||maxMeanMs<=0))fail('INVALID_BUDGET','Budget positive ms required');
  // Use actual runtime; do not claim GPU memory or full peak browser RAM.
  for(let j=0;j<warmup;j++)await session.run(feeds);
  const times=[];
  for(let i=0;i<runs;i++){
   const start=now();
   const result=await session.run(feeds);
   if(!result||!Object.keys(result).length)fail('INFERENCE_FAILED','No real output returned');
   times.push(now()-start);
  }
  const mean=times.reduce((a,b)=>a+b,0)/times.length;
  return{schema:'scept.u9.performance.v1',modelSha256:session.sha256,
   runtime:session.description.runtime||'unspecified',runs,warmup,timesMs:times,
   medianMs:quantile(times,.5),p95Ms:quantile(times,.95),meanMs:mean,
   inputMiB:miB,maxMeanMs,meetsBudget:maxMeanMs===null?null:mean<=maxMeanMs,
   status:'measured',note:'Browser-local on-demand wall-clock measurements. No GPU/heap peak claimed; compare only same runtime/hardware.'};
 }
 return Object.freeze({VERSION:'1.0.0',benchmark,estimateFeeds});
});