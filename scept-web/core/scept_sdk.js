'use strict';
/* SCEPT Universal Core V1 / U9 - trusted extension SDK.
 * JSON manifests are data; JavaScript plugins execute with host privileges and
 * MUST be reviewed/trusted. This SDK is not a JS sandbox.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_SDK=api;})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0',SCHEMA='scept.u9.plugin.v1';
 const REQUIRES=Object.freeze({
  model:['inspect','load'],data:['prepare'],target:['capture'],
  observation:['observePair'],intervention:['apply'],
  explanation:['calculate'],evaluation:[]
 });
 const CAP=Object.freeze({
  model:'inference',data:'data.preprocess',target:'target.embedding',
  observation:'observations.output',intervention:'input.intervention',
  explanation:'method.explain',evaluation:'evaluation.score'
 });
 const fail=(code,s)=>{const e=Error('SCEPT SDK: '+s);e.code=code;throw e};
 const plain=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const goodName=x=>typeof x==='string'&&/^[a-z0-9][a-z0-9._-]{2,95}$/.test(x);
 const hash=x=>typeof x==='string'&&/^[0-9a-f]{64}$/i.test(x);
 function validateAdapter(adapter,core){
  if(!plain(adapter)||!Object.hasOwn(REQUIRES,adapter.kind)||!goodName(adapter.id))
   fail('INVALID_PLUGIN','Plugin kind/id must be from known kinds and a stable namespaced ID');
  if(!Array.isArray(adapter.capabilities)||adapter.capabilities.length===0||
     new Set(adapter.capabilities).size!==adapter.capabilities.length||
     adapter.capabilities.some(c=>!core.CAPABILITIES.includes(c)))
   fail('INVALID_CAPABILITY','Unrecognized or duplicated capability');
  if(!adapter.capabilities.includes(CAP[adapter.kind])&&
     !(adapter.kind==='intervention'&&adapter.capabilities.includes('hidden.intervention')))
   fail('MISSING_CAPABILITY','Declared plugin lacks kind-specific capability');
  for(const method of REQUIRES[adapter.kind])if(typeof adapter[method]!=='function')
   fail('INVALID_ENTRYPOINT','Missing '+method+'()');
  if(adapter.kind==='evaluation'&&typeof adapter.evaluate!=='function'&&typeof adapter.evaluateSuite!=='function')
   fail('INVALID_ENTRYPOINT','Evaluation plugin needs evaluate or evaluateSuite');
  if(adapter.kind==='model'&&(!adapter.capabilities.includes('inference')))fail('MISSING_CAPABILITY','Model must support inference');
  if(typeof adapter.version!=='undefined'&&(typeof adapter.version!=='string'||!adapter.version.trim()))
   fail('INVALID_VERSION','Version must be a nonempty string');
  return{schema:SCHEMA,id:adapter.id,kind:adapter.kind,version:adapter.version||'unspecified',
   capabilities:adapter.capabilities.slice(),entrypoints:REQUIRES[adapter.kind].slice(),
   trust:'trusted host JavaScript; NOT sandboxed'};
 }
 function install({registry,core,adapter}){
  const descriptor=validateAdapter(adapter,core);
  if(registry.find(adapter.kind,adapter.id))fail('DUPLICATE_PLUGIN','Already installed: '+adapter.id);
  registry.register(adapter.kind,adapter);
  return descriptor;
 }
 function compatibility({registry,core,xai,observationModule=null,session,manifest}){
  if(!registry?.requireAdapter||!session?.description||!hash(session.sha256))fail('INVALID_SESSION','Verified model session required');
  const results=[],missing=[];
  try{xai.validateManifest(manifest,registry);results.push({gate:'manifest',status:'pass'})}
  catch(e){missing.push({gate:'manifest',code:e.code||'INVALID_MANIFEST',reason:e.message})}
  const m=manifest||{};
  if(!hash(m.model?.sha256)||m.model.sha256.toLowerCase()!==session.sha256.toLowerCase())
   missing.push({gate:'model-sha256',code:'MODEL_MISMATCH'});
  else results.push({gate:'model-sha256',status:'pass'});
  if(m.model?.modelId&&m.model.modelId!==session.manifest?.modelId)
   missing.push({gate:'model-id',code:'MODEL_MISMATCH'});
  const available=session.description.capabilities||[];
  for(const c of m.model?.requiredCapabilities||[])
   if(!available.includes(c))missing.push({gate:'capability:'+c,code:'UNSUPPORTED_CAPABILITY'});
  const outputs=new Set(session.description.outputs?.map(x=>x.name)||[]);
  const checked=[m.target?.tensor,...(m.observation?.tensors||[])];
  for(const name of checked)if(!outputs.has(name))
   missing.push({gate:'runtime-output:'+name,code:'OUTPUT_UNAVAILABLE'});
  if(m.observation?.profileId&&observationModule){
   try{observationModule.selectProfile(m.observation.profileId)}
   catch(e){missing.push({gate:'observation-profile',code:e.code||'PROFILE_UNSUPPORTED',
    reason:e.message})}
  }
  // Compatible status is a preflight gate. It does NOT validate clinical data
  // semantics, morphology, third-party JS security, or complete ONNX execution.
  return{schema:'scept.u9.compatibility.v1',compatible:missing.length===0,
   modelSha256:session.sha256,checked:results,failures:missing,
   limitation:'Static capability/manifest inspection only. Real numeric execution and scientific controls require separate testing.'};
 }
 function template({id='research.example-xai',kind='explanation'}={}){
  if(!goodName(id)||!Object.hasOwn(REQUIRES,kind))fail('INVALID_TEMPLATE','Choose a supported kind and valid ID');
  const method=kind==='model'?'inspect, load':
   kind==='evaluation'?'evaluate':REQUIRES[kind].join(', ');
  return{schema:SCHEMA,id,kind,version:'1.0.0',
   capabilities:[CAP[kind]],
   requiredMethods:method.split(/,\s*/),
   guidance:'Implement actual methods using independently reviewed JS. Never return invented activations or assume clinical meaning.'};
 }
 function exampleManifest({modelSha256,modelId=null,inputName=null,targetName,probeName=null}={}){
  if(!hash(modelSha256)||typeof targetName!=='string'||!targetName)
   fail('INVALID_TEMPLATE','Real model SHA-256 and output tensor name are required');
  return{schema:'scept.u7.experiment.v1',id:'research.new-experiment',version:'1.0.0',
   model:{sha256:modelSha256,...(modelId?{modelId}:{})},
   target:{adapterId:'u7.target.tensor-element',tensor:targetName,index:0},
   intervention:{pluginId:'u5.noise',mode:'sweep',baseSeed:240216,
    config:{...(inputName?{inputName}:{}),sigma:0.1},parameter:'sigma',intensities:[0,0.05,0.1]},
   observation:{profileId:'generic',tensors:[probeName||targetName]},
   method:{adapterId:'u7.method.target-contrast',params:{}},
   evaluations:[{adapterId:'u7.eval.directional-shift',params:{direction:'absolute',minimum:0}},
    {adapterId:'u7.eval.dose-monotonicity',params:{tolerance:0}}],
   scientificStatus:'Engineering numerical sensitivity only; no causal or medical conclusions'};
 }
 return Object.freeze({VERSION,SCHEMA,REQUIRES,CAP,validateAdapter,install,compatibility,template,exampleManifest});
});