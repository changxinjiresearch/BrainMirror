'use strict';
/* SCEPT V6.0 / 6.1: single-run observation contract (no A/B requirement).
 * An activity observation is NOT saliency, evidence dependence or causality.
 * Browser/CommonJS. No network, paid API, simulated activation or storage.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_RUN_RECORD=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='6.0.0-contract.1',SCHEMA='scept.run-record.v1';
 const STATUSES=Object.freeze(['measured','not_observed','unsupported','failed']);
 const TYPES=Object.freeze(['float32','float64','int32','int64','uint8','bool']);
 function fail(code,message){const e=Error('SCEPT Run Record: '+message);e.code=code;throw e}
 const string=(x,name)=>{if(typeof x!=='string'||!x.trim())fail('INVALID_RECORD',name+' is required');return x};
 const sha=(x,name)=>{if(!/^[0-9a-f]{64}$/i.test(x||''))fail('INVALID_HASH',name+' must be verified SHA-256');return x.toLowerCase()};
 function signature(meta,name){
  if(!meta||typeof meta!=='object')fail('INVALID_SIGNATURE','Missing signature '+name);
  const type=string(meta.type,name+'.type');
  if(!TYPES.includes(type))fail('UNSUPPORTED_DTYPE',type);
  if(!Array.isArray(meta.shape)||!meta.shape.length||meta.shape.length>8||
    !meta.shape.every(n=>Number.isSafeInteger(n)&&n>0||n===null||typeof n==='string'&&n.trim()))
    fail('INVALID_SIGNATURE',name+' shape must be declared (allow named dynamic axes)');
  return {name:string(meta.name,name+'.name'),type,shape:meta.shape.slice()};
 }
 function signatureSet(list,name){
  if(!Array.isArray(list)||!list.length||list.length>64)fail('INVALID_SIGNATURE',name+' is required with 1..64 entries');
  const out=list.map((v,i)=>signature(v,name+'['+i+']'));
  if(new Set(out.map(x=>x.name)).size!==out.length)fail('DUPLICATE_SIGNATURE',name+' names must be unique');
  return out;
 }
 function validateTensor(t,meta,maxElements=8_000_000){
  if(!t||!TYPES.includes(t.type)||!Array.isArray(t.dims)||t.dims.length!==meta.shape.length||
     !t.dims.every((d,i)=>Number.isSafeInteger(d)&&d>0&&
       (typeof meta.shape[i]!=='number'||d===meta.shape[i])))
    fail('TENSOR_SIGNATURE_MISMATCH','actual runtime tensor contradicts '+meta.name);
  const n=t.dims.reduce((a,b)=>a*b,1);
  if(!Number.isSafeInteger(n)||n>maxElements)fail('RESOURCE_LIMIT','Actual tensor exceeds allowed element limit '+meta.name);
  if(!t.data||!ArrayBuffer.isView(t.data)||
     t.data.length!==n||t.type!==meta.type||!ArrayBuffer.isView(t.data)||t.data instanceof DataView)
    fail('TENSOR_SIGNATURE_MISMATCH','actual tensor dtype/elements invalid '+meta.name);
  for(let i=0;i<t.data.length;i++){
   const v=t.data[i];
   if(t.type==='int64' ? typeof v!=='bigint' : typeof v!=='number'||!Number.isFinite(v))
    fail('NONFINITE_ACTIVATION','invalid actual value in '+meta.name);
   if(t.type==='bool'&&v!==0&&v!==1)fail('TENSOR_SIGNATURE_MISMATCH','bad boolean encoding');
  }
  return n;
 }
 function checkProvenance(p){
  if(!p||typeof p!=='object')fail('INVALID_PROVENANCE','Missing verifiable provenance');
  const model={id:string(p.model?.id,'model.id'),sha256:sha(p.model?.sha256,'model.sha256')};
  const feedNames=Object.keys(p.inputs||{}).sort();
  if(!feedNames.length)fail('INVALID_PROVENANCE','Named input SHA-256s required');
  const inputs={};for(const key of feedNames)inputs[string(key,'input.name')]=sha(p.inputs[key],'input.'+key);
  const preprocess={id:string(p.preprocessing?.id,'preprocessing.id'),
    sha256:sha(p.preprocessing?.sha256,'preprocessing.sha256')};
  return{model,inputs,preprocessing:preprocess,
    runtime:{engine:string(p.runtime?.engine,'runtime.engine'),
      version:string(p.runtime?.version,'runtime.version'),
      provider:string(p.runtime?.provider,'runtime.provider')},
    source:'actual-frozen-model-forward'};
 }
 function record({runId,createdAt,provenance,description,outputs,observations,availability,
     limits,claim='internal-activity-only'}){
  const inputSig=signatureSet(description?.inputs,'inputs'),outSig=signatureSet(description?.outputs,'outputs');
  if(!Array.isArray(outputs)||outputs.length!==outSig.length||new Set(outputs.map(x=>x.name)).size!==outputs.length)
    fail('OUTPUT_MISMATCH','Every declared output must be accounted for in one forward');
  for(const o of outputs){
   const meta=outSig.find(m=>m.name===o.name);
   if(!meta||o.status!=='measured'||!o.sha256)fail('OUTPUT_MISMATCH','No authentic measured output: '+o.name);
   sha(o.sha256,'output.sha256');
  }
  if(!Array.isArray(observations)||observations.length>64)fail('INVALID_OBSERVATION','Invalid probe list');
  if(!Array.isArray(availability)||availability.length!==observations.length)
    fail('INVALID_OBSERVATION','Every requested probe must have a status');
  const names=observations.map(x=>x.tensor);
  if(new Set(names).size!==names.length)fail('DUPLICATE_OBSERVATION','Duplicate probe');
  if(availability.some((v,i)=>v.tensor!==observations[i].tensor||!STATUSES.includes(v.status)))
    fail('INVALID_OBSERVATION','Status does not match probes');
  for(let i=0;i<observations.length;i++){
   const o=observations[i],v=availability[i];
   if(v.status==='measured'){
    if(o.status!=='measured'||!outSig.some(m=>m.name===o.tensor)||!o.activity||
      o.activity.source!=='actual-runtime-output')fail('UNTRUSTED_OBSERVATION','Missing actual probe tensor');
   }else if(o.status!=='not_measured'||o.activity!==null)
    fail('FABRICATED_OBSERVATION','Unmeasured probe may not carry activity values');
  }
  if(claim!=='internal-activity-only')fail('OVERCLAIM','Single-run does not establish explanation or causality');
  if(!Number.isSafeInteger(limits?.maxTensorElements)||limits.maxTensorElements<=0||
   !Number.isSafeInteger(limits?.maxCaptureBytes)||limits.maxCaptureBytes<=0)
   fail('INVALID_LIMIT','Resource limits required');
  const out={schema:SCHEMA,version:VERSION,runId:string(runId,'runId'),
   createdAt:string(createdAt,'createdAt'),provenance:checkProvenance(provenance),
   signature:{inputs:inputSig,outputs:outSig},outputs,observations,availability,limits,
   science:{status:'observation_only',causal:'NOT_ESTABLISHED',
    output_importance:'NOT_ESTABLISHED',stability:'NOT_EVALUATED',clinical:'NOT_ESTABLISHED',
    claim,comparison_required:false}};
  // Prevent serializing TypedArrays/raw patient input in record itself.
  JSON.stringify(out,(key,value)=>{
   if(ArrayBuffer.isView(value)||value instanceof ArrayBuffer)
    fail('RAW_DATA_LEAK','Record contains raw tensor buffer: '+key);
   return value;
  });
  return out;
 }
 return Object.freeze({VERSION,SCHEMA,STATUSES,TYPES,signature,signatureSet,
   validateTensor,checkProvenance,record});
});
