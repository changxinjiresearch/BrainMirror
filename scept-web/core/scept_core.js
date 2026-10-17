'use strict';
/**
 * SCEPT Universal Core — U1 contract and capability registry.
 * Neither a trained model nor an explanation is synthesized here.
 * Each adapter must explicitly declare supported operations; unknown support
 * is a hard failure rather than a fabricated activation or decision.
 * Runs in Browser or CommonJS without third-party dependencies.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_CORE=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0-u2';
 const KINDS=Object.freeze(['model','data','target','observation','intervention','explanation','evaluation']);
 const CAPABILITIES=Object.freeze([
  'inference','graph.read','observations.output','observations.intermediate',
  'gradients','hidden.intervention','data.preprocess','target.classification',
  'target.segmentation','target.embedding','target.loss','input.intervention',
  'method.explain','evaluation.score'
 ]);
 class SceptError extends Error {
  constructor(code,message,details){super(message);this.name='SceptError';this.code=code;this.details=details||null;}
 }
 const fail=(code,message,details)=>{throw new SceptError(code,message,details)};
 const isRecord=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const assertString=(v,name)=>{if(typeof v!=='string'||!v.trim())fail('INVALID_CONTRACT',name+' must be a nonempty string')};
 const numericType=t=>['float32','float64','int32','int64','uint8','bool'].includes(t);
 const positiveDims=(shape,allowDynamic=false)=>Array.isArray(shape)&&shape.length>=1&&shape.length<=8&&shape.every(d=>
   Number.isSafeInteger(d)&&d>0 || allowDynamic&&(typeof d==='string'&&d.length>0||d===null));
 function normalizeInput(meta){
  if(!isRecord(meta))fail('INVALID_CONTRACT','Tensor metadata must be an object');
  assertString(meta.name,'Tensor name');
  if(!numericType(meta.type))fail('UNSUPPORTED_DTYPE','Tensor data type not implemented: '+meta.type);
  if(!positiveDims(meta.shape,true))fail('INVALID_SHAPE','Tensor metadata shape is not a valid dimensional array');
  return {name:meta.name,type:meta.type,shape:meta.shape.slice()};
 }
 function validateTensor(tensor,meta){
  if(!isRecord(tensor)||!numericType(tensor.type)||!positiveDims(tensor.dims))
   fail('INVALID_TENSOR','Tensor must provide supported type, concrete positive dims and data');
  const count=tensor.dims.reduce((a,b)=>a*b,1);
  if(!Number.isSafeInteger(count)||count>1e8)fail('RESOURCE_LIMIT','Tensor exceeds configured element limit');
  if(!tensor.data||typeof tensor.data.length!=='number'||tensor.data.length!==count)
   fail('INVALID_TENSOR','Tensor element count does not match dimensions');
  if(meta){
   if(tensor.type!==meta.type)fail('TYPE_MISMATCH','Input tensor '+meta.name+' must be '+meta.type);
   if(tensor.dims.length!==meta.shape.length||!tensor.dims.every((x,i)=>typeof meta.shape[i]!=='number'||x===meta.shape[i]))
    fail('SHAPE_MISMATCH','Tensor does not match model metadata for '+meta.name);
  }
  if(tensor.type!=='int64')for(const v of tensor.data)if(!Number.isFinite(v))
   fail('INVALID_TENSOR','Tensor contains NaN or Infinity');
  return {type:tensor.type,dims:tensor.dims.slice(),elements:count};
 }
 function validateManifest(manifest){
  if(!isRecord(manifest))fail('INVALID_CONTRACT','Model manifest must be an object');
  assertString(manifest.modelId,'modelId');
  assertString(manifest.adapterId,'adapterId');
  if(manifest.sha256!==undefined&&!/^[0-9a-f]{64}$/i.test(manifest.sha256))
   fail('INVALID_CONTRACT','Model sha256 must be a 64-digit hexadecimal hash');
  return Object.freeze({...manifest});
 }
 function createRegistry(){
  const all=new Map(KINDS.map(k=>[k,new Map()]));
  function register(kind,adapter){
   if(!all.has(kind))fail('UNKNOWN_ADAPTER_KIND','Unrecognized adapter type '+kind);
   if(!isRecord(adapter))fail('INVALID_ADAPTER','Adapter object required');
   assertString(adapter.id,'adapter.id');
   if(adapter.kind!==kind)fail('INVALID_ADAPTER','Adapter kind must be '+kind);
   if(!Array.isArray(adapter.capabilities)||adapter.capabilities.some(c=>!CAPABILITIES.includes(c))||
      new Set(adapter.capabilities).size!==adapter.capabilities.length)
      fail('INVALID_CAPABILITIES','Adapter must explicitly declare supported capabilities');
   if(kind==='model')for(const method of ['inspect','load'])if(typeof adapter[method]!=='function')
      fail('INVALID_ADAPTER','Model adapter needs '+method+'()');
   if(all.get(kind).has(adapter.id))fail('DUPLICATE_ADAPTER','Adapter ID already registered: '+adapter.id);
   all.get(kind).set(adapter.id,Object.freeze(adapter));
   return adapter;
  }
  function find(kind,id){
   if(!all.has(kind))fail('UNKNOWN_ADAPTER_KIND','Unknown kind '+kind);
   return all.get(kind).get(id)||null;
  }
  function requireAdapter(kind,id){const a=find(kind,id);if(!a)fail('ADAPTER_NOT_FOUND','No '+kind+' adapter: '+id);return a}
  function list(kind){if(!all.has(kind))fail('UNKNOWN_ADAPTER_KIND','Unknown kind '+kind);return [...all.get(kind).values()].map(a=>({id:a.id,kind:a.kind,capabilities:a.capabilities.slice()}))}
  function requireCapability(adapter,cap){
   if(!CAPABILITIES.includes(cap)||!adapter.capabilities.includes(cap))
     fail('UNSUPPORTED_CAPABILITY',adapter.id+' does not provide '+cap);
  }
  return {register,find,requireAdapter,list,requireCapability};
 }
 async function sha256(buffer){
  const bytes=buffer instanceof ArrayBuffer?buffer:ArrayBuffer.isView(buffer)?
    buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength):null;
  if(!bytes)fail('INVALID_MODEL','Model bytes must be ArrayBuffer or TypedArray');
  if(typeof crypto!=='undefined'&&crypto.subtle){
   return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(v=>v.toString(16).padStart(2,'0')).join('');
  }
  fail('CRYPTO_UNAVAILABLE','SHA-256 not available; refuse unverified model provenance');
 }
 async function openModel(registry,manifest,modelBytes,options={}){
  const m=validateManifest(manifest),adapter=registry.requireAdapter('model',m.adapterId);
  registry.requireCapability(adapter,'inference');
  const actualSha=await sha256(modelBytes);
  if(m.sha256&&actualSha!==m.sha256.toLowerCase())fail('MODEL_HASH_MISMATCH','Model SHA-256 mismatch');
  const description=await adapter.inspect(modelBytes,{manifest:m,options});
  if(!Array.isArray(description.inputs)||!Array.isArray(description.outputs))
   fail('INVALID_ADAPTER','Model inspect must return input and output metadata');
  const inspected={
   ...description,
   inputs:description.inputs.map(normalizeInput),
   outputs:description.outputs.map(normalizeInput),
   capabilities:adapter.capabilities.slice()
  };
  const session=await adapter.load(modelBytes,{manifest:m,description:inspected,options});
  if(!session||typeof session.run!=='function'||typeof session.dispose!=='function')
   fail('INVALID_ADAPTER','load() must return an object with run() and dispose()');
  let closed=false;
  return {
   manifest:m,sha256:actualSha,description:inspected,
   async run(feeds,{outputs=null}={}){
    if(closed)fail('SESSION_DISPOSED','Cannot run a closed session');
    if(!isRecord(feeds))fail('INVALID_TENSOR','Named input tensor mapping required');
    const expected=new Set(inspected.inputs.map(x=>x.name));
    if(Object.keys(feeds).some(x=>!expected.has(x))||Object.keys(feeds).length!==expected.size)
      fail('INPUT_MISMATCH','Input names must exactly match model signature');
    for(const t of inspected.inputs)validateTensor(feeds[t.name],t);
    if(outputs!==null){
     if(!Array.isArray(outputs)||outputs.some(n=>!inspected.outputs.some(x=>x.name===n)))
      fail('OUTPUT_UNAVAILABLE','Requested output is not exposed; do not invent a probe');
    }
    const result=await session.run(feeds,{outputs});
    if(!isRecord(result)||!Object.keys(result).every(k=>inspected.outputs.some(x=>x.name===k)))
      fail('INVALID_ADAPTER','Adapter returned an undeclared output');
    for(const [name,tensor] of Object.entries(result))validateTensor(tensor,inspected.outputs.find(x=>x.name===name));
    return result;
   },
   async dispose(){if(!closed){closed=true;await session.dispose()}},
   isClosed:()=>closed
  };
 }
 return Object.freeze({VERSION,KINDS,CAPABILITIES,SceptError,createRegistry,sha256,validateTensor,
   validateManifest,normalizeInput,openModel});
});
