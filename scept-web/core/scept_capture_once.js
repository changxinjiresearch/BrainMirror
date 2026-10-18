'use strict';
/* SCEPT V6.1 one genuine forward -> independently reusable run record and NAF
 * observations. No paired input, target class, lesion, explanation or A/B required.
 * Full tensor values are private, bounded and *never* in the default JSON export.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_CAPTURE_ONCE=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='6.1.0-core.1';
 function fail(code,msg){const e=Error('SCEPT captureOnce: '+msg);e.code=code;throw e}
 const utf8=new TextEncoder();
 function stable(value){
  if(value===null||typeof value==='string'||typeof value==='boolean')return JSON.stringify(value);
  if(typeof value==='number'&&Number.isFinite(value))return JSON.stringify(value);
  if(Array.isArray(value))return '['+value.map(stable).join(',')+']';
  if(value&&Object.getPrototypeOf(value)===Object.prototype){
   const keys=Object.keys(value).sort();
   return '{'+keys.map(k=>JSON.stringify(k)+':'+stable(value[k])).join(',')+'}';
  }
  fail('INVALID_PREPROCESSING','Preprocessing must be explicit serializable finite configuration');
 }
 async function digest(bytes){
  if(!globalThis.crypto?.subtle)fail('CRYPTO_UNAVAILABLE','SHA-256 is required for every actual tensor');
  const hash=await globalThis.crypto.subtle.digest('SHA-256',bytes);
  return [...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('');
 }
 async function tensorSha(t){
  const head=utf8.encode(stable({type:t.type,shape:t.dims}));
  const content=new Uint8Array(t.data.buffer,t.data.byteOffset,t.data.byteLength);
  const bytes=new Uint8Array(head.length+1+content.length);
  bytes.set(head);bytes[head.length]=0;bytes.set(content,head.length+1);
  return digest(bytes);
 }
 function sameInputSignature(description,feeds,contract,limit){
  const declared=contract.signatureSet(description.inputs,'inputs');
  const actual=Object.keys(feeds||{}).sort(),expected=declared.map(x=>x.name).sort();
  if(JSON.stringify(actual)!==JSON.stringify(expected))
   fail('INPUT_MISMATCH','Exactly declared named feeds required');
  let bytes=0;
  for(const m of declared){
   const t=feeds[m.name];
   contract.validateTensor(t,m,limit.maxTensorElements);
   bytes+=t.data.byteLength;
   if(bytes>limit.maxCaptureBytes)fail('RESOURCE_LIMIT','Input exceeds capture byte budget');
  }
  return declared;
 }
 function make({observation,contract,clock=()=>new Date().toISOString()}={}){
  if(typeof observation?.observeOnce!=='function'||typeof contract?.record!=='function')
   fail('INVALID_DEPENDENCY','Real U4 observeOnce and V6 Run Record contract required');
  async function captureOnce({session,feeds,modelId,probes,profileId='generic',
     preprocessing,runtime,limits={maxTensorElements:8_000_000,maxCaptureBytes:32_000_000,
       maxAxisUnits:2048},retainTensors=true,retainInputs=false}={}){
   if(typeof session?.run!=='function'||!/^[a-f0-9]{64}$/i.test(session?.sha256||'')||
     !session.description?.inputs||!session.description?.outputs)
    fail('INVALID_SESSION','Verified model hash, real execution session and complete ONNX signatures required');
   if(!Array.isArray(probes)||!probes.length||probes.length>64||new Set(probes).size!==probes.length)
    fail('INVALID_PROBES','Declare 1..64 unique actual probe names');
   if(typeof profileId!=='string'||typeof preprocessing?.id!=='string'||
     !preprocessing.id.trim()||!preprocessing.config||!runtime?.engine||!runtime?.provider||!runtime?.version)
    fail('INVALID_PROTOCOL','Explicit observation profile, preprocessing and runtime required');
   if(!Number.isSafeInteger(limits.maxTensorElements)||limits.maxTensorElements<1||
     !Number.isSafeInteger(limits.maxCaptureBytes)||limits.maxCaptureBytes<1||
     !Number.isSafeInteger(limits.maxAxisUnits)||limits.maxAxisUnits<0||limits.maxAxisUnits>16384)
    fail('INVALID_LIMIT','Positive bounded tensor, byte and axis limits required');
   const desc=session.description;
   sameInputSignature(desc,feeds,contract,limits);
   const storedInputs=new Map();
   if(retainInputs)for(const m of desc.inputs){const t=feeds[m.name];
    storedInputs.set(m.name,{type:t.type,dims:t.dims.slice(),data:new t.data.constructor(t.data)});
   }
   const inputHashes={};
   for(const m of desc.inputs)inputHashes[m.name]=await tensorSha(feeds[m.name]);
   const preprocessHash=await digest(utf8.encode(stable(preprocessing)));
   const outputMetas=contract.signatureSet(desc.outputs,'outputs');
   // The session is called ONCE, with the unchanged caller inputs.
   const actual=await session.run(feeds);
   for(const m of desc.inputs)
    if(await tensorSha(feeds[m.name])!==inputHashes[m.name])
     fail('INPUT_MUTATION','Model runtime modified the original input: '+m.name);
   const outNames=Object.keys(actual||{}).sort(),expected=outputMetas.map(o=>o.name).sort();
   if(JSON.stringify(outNames)!==JSON.stringify(expected))
    fail('OUTPUT_MISMATCH','Every declared actual graph output is required; no extras allowed');
   let used=0;
   const outputRecords=[],raw=new Map();
   for(const meta of outputMetas){
    const t=actual[meta.name];
    contract.validateTensor(t,meta,limits.maxTensorElements);
    used+=t.data.byteLength;
    if(used>limits.maxCaptureBytes)fail('RESOURCE_LIMIT','Outputs exceed bounded retention / hashing budget');
    outputRecords.push({name:meta.name,status:'measured',type:t.type,shape:t.dims.slice(),
      sha256:await tensorSha(t),elements:t.data.length});
    if(retainTensors)raw.set(meta.name,{type:t.type,dims:t.dims.slice(),
      data:new t.data.constructor(t.data)});
   }
   const measurements=observation.observeOnce({description:desc,outputs:actual,
     probes,profileId,maxAxisUnits:limits.maxAxisUnits});
   const observations=measurements.map(v=>({tensor:v.tensor,status:v.status,
     activity:v.activity,reason:v.reason||null}));
   const availability=observations.map(v=>({tensor:v.tensor,status:v.status==='measured'?'measured':'not_observed',
     reason:v.reason||null}));
   const idBytes=new Uint8Array(16);
   if(!globalThis.crypto?.getRandomValues)fail('CRYPTO_UNAVAILABLE','Secure run nonce required');
   globalThis.crypto.getRandomValues(idBytes);
   const record=contract.record({runId:'run-'+[...idBytes].map(v=>v.toString(16).padStart(2,'0')).join(''),
     createdAt:clock(),
     provenance:{model:{id:modelId,sha256:session.sha256},inputs:inputHashes,
       preprocessing:{id:preprocessing.id,sha256:preprocessHash},runtime},
     description:desc,outputs:outputRecords,observations,availability,
     limits:{...limits},claim:'internal-activity-only'});
   // Immutable export snapshot: never attach raw MRI, tensor buffers or mutable ORT objects.
   const json=JSON.stringify(record);
   let released=false;
   return Object.freeze({
    record:JSON.parse(json),
    exportRecord:()=>JSON.parse(json),
    listStored:()=>released?[]:[...raw.keys()],
    listStoredInputs:()=>released?[]:[...storedInputs.keys()],
    getInputTensor:(name)=>{
     if(released)fail('RELEASED','Retained tensors have been cleared');
     const v=storedInputs.get(name);if(!v)return null;
     return {type:v.type,dims:v.dims.slice(),data:new v.data.constructor(v.data)};
    },
    getTensor:(name)=>{
     if(released)fail('RELEASED','Retained tensors have been cleared');
     const v=raw.get(name);
     if(!v)return null;
     return {type:v.type,dims:v.dims.slice(),data:new v.data.constructor(v.data)};
    },
    release:()=>{raw.clear();storedInputs.clear();released=true},
    isReleased:()=>released
   });
  }
  return Object.freeze({captureOnce});
 }
 return Object.freeze({VERSION,make,tensorSha,stable});
});
