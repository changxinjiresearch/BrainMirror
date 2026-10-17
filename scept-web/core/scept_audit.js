'use strict';
/* SCEPT U9 integrity audit and deterministic, opt-in re-execution.
 * SHA-256 is a content checksum, not a digital signature or proof of origin.
 * Exports exclude raw image tensors and redact potentially sensitive ROI indices.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_AUDIT=api;})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0',SCHEMA='scept.u9.audit.v1',MAX_BYTES=6000000;
 const fail=(code,s)=>{const e=Error('SCEPT Audit: '+s);e.code=code;throw e};
 const record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 function normalize(value,{redact=false}={}){
  const seen=new Set();
  function visit(x,depth,key=''){
   if(depth>35)fail('RESOURCE_LIMIT','Nesting >35');
   if(x===null||typeof x==='string'||typeof x==='boolean')return x;
   if(typeof x==='number'){if(!Number.isFinite(x))fail('INVALID_JSON','Nonfinite metric');return x}
   if(typeof x!=='object')fail('INVALID_JSON','Non-JSON data');
   if(seen.has(x))fail('INVALID_JSON','Circular data');
   seen.add(x);let out;
   if(Array.isArray(x)){
    if(x.length>200000)fail('RESOURCE_LIMIT','Oversized evidence list');
    if(redact&&['indices','pixels','voxels','rawSamples'].includes(key))
     out={redacted:true,count:x.length};
    else out=x.map(v=>visit(v,depth+1));
   }else{
    if(Object.getPrototypeOf(x)!==Object.prototype&&Object.getPrototypeOf(x)!==null)
     fail('INVALID_JSON','Class instances and typed arrays must not enter audit');
    out={};
    for(const k of Object.keys(x).sort()){
     if(['__proto__','constructor','prototype'].includes(k))
      fail('UNSAFE_KEY','Prototype mutation keys forbidden');
     if(redact&&['data','modelBytes','weights','rawInput','patientId','patientName',
        'patientAddress','medicalRecord','phi'].includes(k)){
      out[k]={redacted:true,reason:'private raw data/identifier cannot be exported'};
      continue;
     }
     out[k]=visit(x[k],depth+1,k);
    }
   }
   seen.delete(x);return out;
  }
  const out=visit(value,0);
  const size=JSON.stringify(out).length;
  if(size>MAX_BYTES)fail('RESOURCE_LIMIT','Audit JSON exceeds 6 MB bound');
  return out;
 }
 function canonical(value){return JSON.stringify(normalize(value))}
 async function digest(core,data){
  if(typeof core?.sha256!=='function')fail('CRYPTO_UNAVAILABLE','Trusted SHA-256 adapter required');
  return core.sha256(new TextEncoder().encode(canonical(data)));
 }
 function expectedHashes(kind,record){
  const obj=kind==='u7'?record.provenance:record;
  if(!obj||!recordMap(obj.inputHashes))fail('INVALID_RECORD','Complete U6/U7 input provenance required');
  return{modelSha256:obj.modelSha256,inputHashes:obj.inputHashes,referenceHashes:obj.referenceHashes||null};
 }
 function recordMap(x){return record(x)&&Object.keys(x).length>0&&Object.values(x).every(y=>
  record(y)&&/^[a-f0-9]{64}$/i.test(y.sha256||'')&&Array.isArray(y.dims)&&
  typeof y.type==='string')}
 function kindFrom(record){
  if(record?.schema==='scept.u7.evidence.v1')return'u7';
  if(record?.schema==='scept.u6.status.v1')return'u6';
  fail('INVALID_RECORD','Accepts only native U6 snapshot or U7 evidence');
 }
 function fullCompleted(kind,rec){
  if(!Array.isArray(rec.tasks)||!rec.tasks.length||rec.tasks.length>128||
     rec.tasks.some(t=>t.status!=='completed'))fail('INCOMPLETE_EXPERIMENT','Only fully completed U6/U7 experiments can be replay certified');
  if(kind==='u6'&&rec.tasks.some(t=>!t.result?.genuineInference))
   fail('INCOMPLETE_EXPERIMENT','All U6 jobs must come from true model inference');
  if(kind==='u7'&&rec.tasks.some(t=>!t.observations?.length))
   fail('INCOMPLETE_EXPERIMENT','U7 task has no actual ONNX observations');
 }
 async function create({core,record:raw,definition,requireComplete=true}){
  const kind=kindFrom(raw),source=normalize(raw),def=normalize(definition);
  if(!record(def))fail('INVALID_DEFINITION','Original plan/manifest required');
  if(requireComplete)fullCompleted(kind,source);
  const hashes=expectedHashes(kind,source);
  if(!/^[a-f0-9]{64}$/i.test(hashes.modelSha256||''))
   fail('INVALID_MODEL','Model SHA256 must be a 64-character hex string');
  if(kind==='u7'&&def.schema!=='scept.u7.experiment.v1')
   fail('INVALID_DEFINITION','Expected exact U7 manifest');
  if(kind==='u7'&&canonical(def)!==canonical(source.experiment?.manifest))
   fail('DEFINITION_MISMATCH','Manifest used for export must exactly match completed experiment');
  if(kind==='u6'&&!(def.mode==='ab'||def.mode==='sweep'||def.mode==='batch'))
   fail('INVALID_DEFINITION','U6 requires original complete plan, not only status');
  const definitionSha256=await digest(core,def);
  const report=normalize(source,{redact:true});
  const content={schema:SCHEMA,version:VERSION,kind,modelSha256:hashes.modelSha256,
   inputHashes:hashes.inputHashes,referenceHashes:hashes.referenceHashes,
   definitionSha256,record:report,
   promises:{model:'SHA-256 verified by session',input:'SHA-256 verified against caller originals on replay',
    privacy:'Redacted numeric/identifier fields; export still may contain sensitive derived or plugin metadata',
    scientific:'No medical/causal validity inferred',signature:'NONE; content checksum is not authentication'}};
  return{...content,contentSha256:await digest(core,content)};
 }
 async function verify({core,bundle,modelSha256,feeds,referenceFeeds=null,definition}){
  if(!record(bundle)||bundle.schema!==SCHEMA||!['u6','u7'].includes(bundle.kind))
   fail('INVALID_BUNDLE','Unknown U9 audit schema');
  if(!/^[a-f0-9]{64}$/i.test(bundle.contentSha256||''))
   fail('INVALID_BUNDLE','Missing content checksum');
  const data={...bundle};delete data.contentSha256;
  if(await digest(core,data)!==bundle.contentSha256)fail('AUDIT_HASH_MISMATCH','Bundle altered since checksum');
  if(!/^[a-f0-9]{64}$/i.test(modelSha256||'')||
     bundle.modelSha256.toLowerCase()!==modelSha256.toLowerCase())
   fail('MODEL_MISMATCH','Exact original model weight SHA-256 required');
  if(!definition||await digest(core,definition)!==bundle.definitionSha256)
   fail('DEFINITION_MISMATCH','Supply original unredacted manifest or plan');
  const required=async(map,originals)=>{
   if(map===null){if(originals!==null)fail('REFERENCE_MISMATCH','Unexpected reference input');return}
   if(!record(map)||!record(originals)||Object.keys(map).length!==Object.keys(originals).length)
    fail('INPUT_HASH_MISMATCH','Exact input names required for replay');
   for(const name of Object.keys(map)){
    const t=originals[name],e=map[name];
    if(!t||t.type!==e.type||JSON.stringify(t.dims)!==JSON.stringify(e.dims))
     fail('INPUT_HASH_MISMATCH','Replay type/dims mismatch for '+name);
    if(!ArrayBuffer.isView(t.data)||await core.sha256(t.data)!==e.sha256)
     fail('INPUT_HASH_MISMATCH','Input checksum mismatch '+name);
   }
  };
  await required(bundle.inputHashes,feeds);
  await required(bundle.referenceHashes,referenceFeeds);
  return{verified:true,modelSha256,inputsVerified:Object.keys(bundle.inputHashes).length,
   contentSha256:bundle.contentSha256,definitionSha256:bundle.definitionSha256,
   status:'integrity_verified',scientificStatus:'NOT_ESTABLISHED'};
 }
 function compare({expected,observed,kind,atol=1e-5,rtol=1e-4}){
  if(typeof atol!=='number'||!Number.isFinite(atol)||atol<0||
     typeof rtol!=='number'||!Number.isFinite(rtol)||rtol<0)
   fail('INVALID_TOLERANCE','Finite nonnegative abs/rel tolerances required');
  if(!Array.isArray(expected?.tasks)||!Array.isArray(observed?.tasks)||
     expected.tasks.length!==observed.tasks.length)fail('REPLAY_DIVERGENCE','Task count differs');
  let n=0,maxAbsError=0;
  function near(a,b,label){
   if(!Number.isFinite(a)||!Number.isFinite(b)||
      Math.abs(a-b)>atol+rtol*Math.max(Math.abs(a),Math.abs(b)))
    fail('REPLAY_DIVERGENCE',label+' numerical result differs');
   n++;maxAbsError=Math.max(maxAbsError,Math.abs(a-b));
  }
  function observations(a,b,label){
   if(!Array.isArray(a)||!Array.isArray(b)||a.length!==b.length)fail('REPLAY_DIVERGENCE',label+' tensor count');
   for(let j=0;j<a.length;j++){
    const x=a[j],y=b[j];
    if(x.tensor!==y.tensor||x.evidence!==y.evidence||
      JSON.stringify(x.shape)!==JSON.stringify(y.shape)||x.changed!==y.changed)
     fail('REPLAY_DIVERGENCE',label+' tensor shape/name/changed mismatch');
    for(const k of ['mean_abs_A','mean_abs_B','mean_abs_delta','rms_delta','max_abs_delta'])
     near(x[k],y[k],label+' '+k);
   }
  }
  for(let i=0;i<expected.tasks.length;i++){
   const a=expected.tasks[i],b=observed.tasks[i];
   if(a.id!==b.id||a.seed!==b.seed||a.status!=='completed'||b.status!=='completed')
    fail('REPLAY_DIVERGENCE','Task ID/seed/status mismatch');
   if(kind==='u7'){
    observations(a.observations,b.observations,a.id);
    if(!a.target||!b.target)fail('REPLAY_DIVERGENCE','Missing target');
    for(const k of ['baseline','edited','delta'])near(a.target[k],b.target[k],a.id+' target');
    if(a.tests?.length!==b.tests?.length)fail('REPLAY_DIVERGENCE','Evaluation length differs');
    for(let k=0;k<a.tests.length;k++)if(a.tests[k].verdict!==b.tests[k].verdict||
     a.tests[k].plugin!==b.tests[k].plugin)
     fail('REPLAY_DIVERGENCE','Evaluation verdict differs');
   }else{
    observations(a.result?.outputs,b.result?.outputs,a.id);
   }
  }
  return{schema:'scept.u9.replay.verdict.v1',status:'matched',tasks:expected.tasks.length,
   comparedNumbers:n,maxAbsError,atol,rtol,
   claim:'Deterministic numerical consistency under same model/runtime; no clinical or causal validation'};
 }
 async function replay({core,bundle,session,feeds,referenceFeeds=null,definition,engine,
  atol=1e-5,rtol=1e-4}){
  await verify({core,bundle,modelSha256:session?.sha256,feeds,referenceFeeds,definition});
  fullCompleted(bundle.kind,bundle.record);
  if(!engine?.run)fail('INVALID_ENGINE','Real U6 or U7 engine required');
  if(bundle.kind==='u7'){
   if(!engine.registerExperiment||!engine.prepare||!engine.exportEvidence)
    fail('INVALID_ENGINE','Requires real U7 engine');
   engine.registerExperiment(definition);
   await engine.prepare(definition.id,{feeds,referenceFeeds});
   const output=await engine.run();
   return{verification:'verified input and audit checksums',...compare({expected:bundle.record,
    observed:output,kind:'u7',atol,rtol})};
  }
  if(!engine.create||!engine.exportAudit)fail('INVALID_ENGINE','Requires real U6 scheduler');
  await engine.create(definition,{feeds,referenceFeeds,profileId:bundle.record.profileId||'generic'});
  await engine.run();
  return{verification:'verified input and audit checksums',...compare({expected:bundle.record,
   observed:engine.exportAudit(),kind:'u6',atol,rtol})};
 }
 return Object.freeze({VERSION,SCHEMA,normalize,canonical,digest,create,verify,compare,replay});
});