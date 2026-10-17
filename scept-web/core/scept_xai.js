'use strict';
/* SCEPT U7/M2: declarative experiment plugins, independent method/target/evaluation
 * adapters and real ONNX evidence. Does not certify medical or causal claims.
 * Browser/CommonJS compatible; all registered JS adapters must be trusted code.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_XAI=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0-u7',SCHEMA='scept.u7.experiment.v1';
 const record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const fail=(code,s)=>{const e=Error('SCEPT U7: '+s);e.code=code;throw e};
 const label=(s,name)=>{if(typeof s!=='string'||!s.trim()||s.length>200)fail('INVALID_MANIFEST',name+' must be a nonempty bounded string');return s};
 const own=(o,k)=>Object.prototype.hasOwnProperty.call(o,k);
 function json(value){
  const seen=new Set();let n=0;
  const check=(v,depth)=>{
   if(depth>24)fail('INVALID_MANIFEST','Excessive JSON nesting');
   if(typeof v==='number'&&!Number.isFinite(v))fail('INVALID_MANIFEST','Nonfinite number');
   if(v===undefined||typeof v==='function'||typeof v==='bigint'||typeof v==='symbol')fail('INVALID_MANIFEST','Non-JSON value');
   if(v&&typeof v==='object'){if(seen.has(v))fail('INVALID_MANIFEST','Circular structure');seen.add(v);
    if(Array.isArray(v))v.forEach(z=>check(z,depth+1));
    else if(record(v))Object.entries(v).forEach(([key,x])=>{
      if(['__proto__','constructor','prototype'].includes(key))fail('INVALID_MANIFEST','Unsafe object property');
      check(x,depth+1);
    });else fail('INVALID_MANIFEST','Non-JSON object');
    seen.delete(v);
   }
  };
  check(value,0);
  const str=JSON.stringify(value);
  if(!str||(n=str.length)>300000)fail('RESOURCE_LIMIT','Manifest/evidence too large: '+n);
  return JSON.parse(str);
 }
 function delta(a,b){return b-a}
 function num(v,labelName){
  if(typeof v!=='number'||!Number.isFinite(v))fail('INVALID_PARAMETER',labelName+' must be finite');
  return v;
 }
 function numericTensor(result,name){
  const t=result?.[name];
  if(!t||!Array.isArray(t.dims)||!t.data||!t.data.length)fail('TARGET_UNAVAILABLE','Real ONNX tensor not returned: '+name);
  if(!['float32','float64','int32','int64','uint8','bool'].includes(t.type))fail('TARGET_UNSUPPORTED','Non-numeric target dtype');
  return t;
 }
 function extract(t,selector){
  if(selector.aggregate==='mean'){
   if(t.data.length>10000000)fail('RESOURCE_LIMIT','Mean target element budget exceeded');
   let sum=0;for(const raw of t.data)sum+=num(Number(raw),'target value');
   return sum/t.data.length;
  }
  const index=selector.index;
  if(!Number.isSafeInteger(index)||index<0||index>=t.data.length)fail('TARGET_INDEX_INVALID','Explicit flattened target index outside genuine runtime tensor');
  return num(Number(t.data[index]),'target value');
 }
 function registerBuiltins(registry,core){
  const targets=[
   {id:'u7.target.tensor-element',kind:'target',capabilities:['target.embedding'],capture({selector,baseline,edited,control}){
     if(selector.aggregate==='mean')fail('TARGET_CONTRACT','Use u7.target.tensor-mean for mean target');
     const tensor=label(selector.tensor,'target tensor');
     const capture=x=>extract(numericTensor(x,tensor),selector);
     const A=capture(baseline),B=capture(edited),C=control?capture(control):null;
     return{schema:'scept.u7.target.v1',tensor,selector:{kind:'flat-index',index:selector.index},
      baseline:A,edited:B,control:C,delta:delta(A,B),controlDelta:C===null?null:delta(A,C),semantics:'explicit numeric output element'};
   }},
   {id:'u7.target.tensor-mean',kind:'target',capabilities:['target.embedding'],capture({selector,baseline,edited,control}){
     const tensor=label(selector.tensor,'target tensor'),sel={aggregate:'mean'};
     const cap=x=>extract(numericTensor(x,tensor),sel);
     const A=cap(baseline),B=cap(edited),C=control?cap(control):null;
     return{schema:'scept.u7.target.v1',tensor,selector:{kind:'mean-all-elements'},
      baseline:A,edited:B,control:C,delta:delta(A,B),controlDelta:C===null?null:delta(A,C),semantics:'explicit arithmetic mean, not inferred class confidence'};
   }}
  ];
  const methods=[
   {id:'u7.method.target-contrast',kind:'explanation',capabilities:['method.explain'],calculate({target}){
    return{schema:'scept.u7.method.v1',method:'target-contrast',score:Math.abs(target.delta),
     signedDelta:target.delta,controlScore:target.controlDelta===null?null:Math.abs(target.controlDelta),
     outputType:'target-numeric-contrast',claim:'descriptive perturbation response; no causal attribution established'};
   }},
   {id:'u7.method.activation-response',kind:'explanation',capabilities:['method.explain'],calculate({target,observations,config}){
     const n=config.topK??8;
     if(!Number.isInteger(n)||n<1||n>32)fail('INVALID_PARAMETER','topK must be integer 1..32');
     if(!Array.isArray(observations)||!observations.length)fail('OBSERVATION_UNAVAILABLE','At least one genuine tensor observation required');
     const scores=observations.map(o=>{
      if(o.evidence!=='actual-onnx-runtime-graph-outputs')fail('FAKE_OBSERVATION','Only genuinely executed ONNX tensor statistics accepted');
      return{tensor:o.tensor,source:o.source,meanAbsDelta:num(o.mean_abs_delta,'observed response'),rmsDelta:num(o.rms_delta,'rms'),
       changed:o.changed,total:o.total};
     }).sort((a,b)=>b.meanAbsDelta-a.meanAbsDelta||a.tensor.localeCompare(b.tensor)).slice(0,n);
     return{schema:'scept.u7.method.v1',method:'activation-response-ranking',
      score:Math.abs(target.delta),ranking:scores,
      outputType:'observational-activation-response',claim:'ranked real tensor changes do not establish evidential dependence'};
   }}
  ];
  const evaluations=[
   {id:'u7.eval.directional-shift',kind:'evaluation',capabilities:['evaluation.score'],evaluate({target,method,config}){
     const direction=config.direction||'absolute',threshold=num(config.minimum??0,'minimum');
     if(!['absolute','increase','decrease'].includes(direction)||threshold<0)fail('INVALID_PARAMETER','direction/minimum');
     const observed=direction==='increase'?target.delta:direction==='decrease'?-target.delta:Math.abs(target.delta);
     return{test:'directional-shift',schema:'scept.u7.test.v1',verdict:observed>=threshold?'criterion_met':'criterion_not_met',
      observed,threshold,direction,unit:'raw selected target output',scientificStatus:'numerical criterion only'};
   }},
   {id:'u7.eval.control-specificity',kind:'evaluation',capabilities:['evaluation.score'],evaluate({target,config}){
     const margin=num(config.minMargin??0,'minMargin');
     if(margin<0)fail('INVALID_PARAMETER','minMargin must be >= 0');
     if(target.controlDelta===null)return{test:'control-specificity',schema:'scept.u7.test.v1',
       verdict:'not_evaluable',reason:'No actual third inference/control output supplied',scientificStatus:'no scientific inference'};
     const main=Math.abs(target.delta),control=Math.abs(target.controlDelta),observed=main-control;
     return{test:'control-specificity',schema:'scept.u7.test.v1',verdict:observed>=margin?'criterion_met':'criterion_not_met',
      observed,main,control,threshold:margin,scientificStatus:'numeric control contrast, not confound adjustment'};
   }},
   {id:'u7.eval.dose-monotonicity',kind:'evaluation',capabilities:['evaluation.score'],scope:'suite',
    evaluateSuite({tasks,config}){
     const tolerance=num(config.tolerance??0,'tolerance');if(tolerance<0)fail('INVALID_PARAMETER','tolerance must be nonnegative');
     const successes=tasks.filter(t=>t.status==='completed'&&t.result?.analysis?.target&&Number.isFinite(t.intensity))
      .sort((a,b)=>a.intensity-b.intensity);
     if(successes.length<3)return{test:'dose-monotonicity',schema:'scept.u7.test.v1',verdict:'not_evaluable',
      reason:'At least three distinct completed strengths required'};
     if(new Set(successes.map(t=>t.intensity)).size!==successes.length)return{test:'dose-monotonicity',schema:'scept.u7.test.v1',
      verdict:'not_evaluable',reason:'Repeated strength; unique numeric intensity required'};
     const points=successes.map(t=>({strength:t.intensity,absoluteTargetDelta:Math.abs(t.result.analysis.target.delta)}));
     const violations=[];for(let i=1;i<points.length;i++)if(points[i].absoluteTargetDelta+tolerance<points[i-1].absoluteTargetDelta)
       violations.push({from:points[i-1].strength,to:points[i].strength});
     return{test:'dose-monotonicity',schema:'scept.u7.test.v1',verdict:violations.length?'criterion_not_met':'criterion_met',
      violations,points,tolerance,scientificStatus:'numerical monotonic response only; not evidence of biological causation'};
    }}
  ];
  for(const adapter of [...targets,...methods,...evaluations])registry.register(adapter.kind,adapter);
  // Optional sparse mask-based occlusion for lesion hypotheses. A self-reported
  // mask's scientific/anatomical validity is NEVER inferred from its checksum.
  registry.register('intervention',{id:'u7.masked-occlusion',kind:'intervention',capabilities:['input.intervention'],
   async apply({description,baseline,config,dataAdapter}){
    const m=config.mask;
    if(!record(m)||!Array.isArray(m.indices)||!Array.isArray(m.shape)||!m.indices.length)
     fail('MASK_REQUIRED','Explicit nonempty mask indices and actual tensor shape are required');
    label(m.sourceId,'mask sourceId');
    if(m.coordinateSpace!=='model-input-flat-indices')fail('MASK_COORDINATES','Explicit model-input-flat-indices coordinate space required');
    const name=label(m.inputName,'mask input name');
    const meta=description.inputs.find(x=>x.name===name),tensor=baseline[name];
    if(!meta||!tensor||!['float32','float64'].includes(tensor.type))fail('MASK_SIGNATURE','Mask must refer to real floating point model input');
    if(m.shape.length!==tensor.dims.length||tensor.dims.some((n,i)=>m.shape[i]!==n))
     fail('MASK_SIGNATURE','Mask shape must equal actual model input shape');
    if(m.indices.length>2000000)fail('RESOURCE_LIMIT','Mask index limit exceeded');
    const unique=new Set();
    for(const index of m.indices){
     if(!Number.isInteger(index)||index<0||index>=tensor.data.length||unique.has(index))
      fail('MASK_INVALID','Mask indices must be unique, sorted/deterministic and in range');
     unique.add(index);
    }
    const indices=[...unique].sort((a,b)=>a-b),hash=await core.sha256(Uint32Array.from(indices));
    if(m.indicesSha256!==hash)fail('MASK_HASH_MISMATCH','Mask indices checksum does not match actual region');
    const fill=num(config.fill??0,'mask fill'),edited=dataAdapter.prepare(description,baseline);
    let changed=0;
    for(const idx of indices){edited[name].data[idx]=fill;
     if(!Number.isFinite(edited[name].data[idx]))fail('NONFINITE_MASK','Edited mask value overflowed dtype');
     if(edited[name].data[idx]!==tensor.data[idx])changed++;}
    return{baseline,edited,metadata:{kind:'sparse-mask-occlusion',inputName:name,maskSourceId:m.sourceId,
      maskIndicesSha256:hash,coordinateSpace:m.coordinateSpace,selected:indices.length,changed,
      fill,anatomicalValidation:'NOT_VERIFIED',registrationValidation:'NOT_VERIFIED',
      sourceClaim:'User-declared tensor-space mask; hash ensures exact indices only'}};
   }});
 }
 function validateManifest(raw,registry){
  if(!record(raw))fail('INVALID_MANIFEST','Experiment plugin JSON object required');
  const m=json(raw);
  if(m.schema!==SCHEMA)fail('INVALID_SCHEMA','Expected '+SCHEMA);
  label(m.id,'experiment id');label(m.version,'experiment version');
  if(!record(m.model)||!/^[a-f0-9]{64}$/i.test(m.model.sha256||''))
   fail('INVALID_MANIFEST','Pinned 64-character model sha256 required');
  if(!record(m.target)||!record(m.intervention)||!record(m.observation)||!record(m.method))
   fail('INVALID_MANIFEST','Model, target, intervention, observation and method are all required');
  label(m.target.adapterId,'target adapterId');label(m.target.tensor,'target tensor');
  label(m.intervention.pluginId,'intervention pluginId');
  label(m.method.adapterId,'method adapterId');
  if(!Array.isArray(m.observation.tensors)||!m.observation.tensors.length||m.observation.tensors.length>16||
     m.observation.tensors.some(v=>typeof v!=='string'||!v))fail('INVALID_MANIFEST','Explicit observed tensor names required (<=16)');
  if(!Array.isArray(m.evaluations)||!m.evaluations.length||m.evaluations.length>16)
   fail('INVALID_MANIFEST','Nonempty registered evaluation adapters required');
  for(const [kind,id,cap] of [
   ['target',m.target.adapterId,'target.embedding'],['intervention',m.intervention.pluginId,'input.intervention'],
   ['explanation',m.method.adapterId,'method.explain']]){
    const adapter=registry.requireAdapter(kind,id);
    registry.requireCapability(adapter,kind==='intervention'&&id==='u5.internal-mask'?'hidden.intervention':cap);
  }
  for(const e of m.evaluations){
   if(!record(e))fail('INVALID_MANIFEST','Invalid evaluation');
   registry.requireCapability(registry.requireAdapter('evaluation',label(e.adapterId,'evaluation id')),'evaluation.score');
  }
  if(m.intervention.mode&&!['ab','sweep','batch'].includes(m.intervention.mode))
   fail('INVALID_MANIFEST','Schedule must be ab/sweep/batch');
  if(m.intervention.pluginId==='u7.masked-occlusion'&&
     (!m.requirements||m.requirements.lesionMask!==true))
    fail('MASK_REQUIRED','Sparse lesion experiment must explicitly declare external lesion mask precondition');
  return Object.freeze(m);
 }
 function make({registry,session,interventionEngine,observationAdapter,dataAdapter,core,schedulerFactory,onProgress=null}){
  if(!registry||!session?.run||!interventionEngine?.execute||!observationAdapter?.observePair||
     !dataAdapter?.prepare||!core?.sha256||!schedulerFactory?.make)
   fail('INVALID_ENGINE','U1/U3/U4/U5/U6 and genuine opened model session required');
  const manifests=new Map();
  let queue=null,active=null;
  function registerExperiment(manifest){
   if(queue?.snapshot().status==='running')fail('BUSY','Cannot change experiments while running');
   const m=validateManifest(manifest,registry);
   if(manifests.has(m.id))fail('DUPLICATE_EXPERIMENT','Experiment ID already registered');
   manifests.set(m.id,m);return{ id:m.id,version:m.version,schema:m.schema };
  }
  function listExperiments(){return [...manifests.values()].map(m=>({id:m.id,version:m.version,
   method:m.method.adapterId,intervention:m.intervention.pluginId}))}
  async function prepare(id,{feeds,referenceFeeds=null}={}){
   if(queue?.snapshot().status==='running')fail('BUSY','Experiment running');
   const manifest=manifests.get(id);if(!manifest)fail('EXPERIMENT_NOT_FOUND','No registered experiment '+id);
   if(manifest.model.sha256.toLowerCase()!==session.sha256.toLowerCase())
    fail('MODEL_MISMATCH','Experiment model SHA-256 differs from actual loaded model');
   if(manifest.model.modelId&&session.manifest?.modelId!==manifest.model.modelId)
    fail('MODEL_MISMATCH','Experiment model ID differs');
   if(manifest.model.requiredCapabilities?.some(c=>!session.description.capabilities.includes(c)))
    fail('UNSUPPORTED_CAPABILITY','Model lacks experiment-required capability');
   const names=new Set(session.description.outputs.map(x=>x.name));
   if(!names.has(manifest.target.tensor)||manifest.observation.tensors.some(name=>!names.has(name)))
    fail('OBSERVATION_UNAVAILABLE','Target and observed tensors must be real ONNX graph outputs');
   // Prevent phantom internal observations: a named intermediate must actually
   // have been promoted to an output during model opening.
   const declared=new Set(session.manifest?.probes||[]);
   for(const name of manifest.observation.tensors){
    if(!declared.has(name)&&!names.has(name))fail('OBSERVATION_UNAVAILABLE','Unexposed probe '+name);
   }
   if(manifest.requirements?.lesionMask){
    const mask=manifest.intervention.config?.mask;
    if(!mask?.sourceId||!mask?.indices?.length||!mask?.indicesSha256)
     fail('MASK_REQUIRED','Lesion-removal hypothesis requires an explicit mask source and verified index checksum');
   }
   const targetAdapter=registry.requireAdapter('target',manifest.target.adapterId),
    methodAdapter=registry.requireAdapter('explanation',manifest.method.adapterId);
   const perTask=manifest.evaluations.filter(e=>!registry.requireAdapter('evaluation',e.adapterId).evaluateSuite);
   const analysisAdapter={evaluateTask:async ({baseline,edited,control,observations,matchedControlObservations,intervention,task})=>{
    const target=await targetAdapter.capture({selector:manifest.target,baseline,edited,control});
    const method=await methodAdapter.calculate({target,observations,matchedControlObservations,
      intervention,task,config:manifest.method.params||{}});
    if(!record(method)||typeof method.method!=='string')fail('INVALID_EXPLANATION','Registered method returned invalid evidence');
    const tests=[];
    for(const e of perTask){
     const adapter=registry.requireAdapter('evaluation',e.adapterId);
     if(typeof adapter.evaluate!=='function')fail('INVALID_EVALUATOR','Evaluator must implement evaluate');
     const result=await adapter.evaluate({target,method,observations,matchedControlObservations,
      config:e.params||{},task,intervention});
     if(!record(result)||typeof result.verdict!=='string')fail('INVALID_EVALUATION','Evaluator returned no verdict');
     tests.push({plugin:e.adapterId,...json(result)});
    }
    return json({schema:'scept.u7.task.evidence.v1',target,method,tests,
     source:'actual-frozen-onnx-runtime-inference',limits:[
      'Numeric target observations do not prove pathology dependence',
      'No causal interpretation without scientifically valid controls',
      'Mask anatomy and co-registration are not validated by U7']});
   }};
   const spec={mode:manifest.intervention.mode||'ab',pluginId:manifest.intervention.pluginId,
    config:manifest.intervention.config||{},parameter:manifest.intervention.parameter,
    intensities:manifest.intervention.intensities,controls:manifest.intervention.controls,
    baseSeed:manifest.intervention.baseSeed??240216};
   const fresh=schedulerFactory.make({session,interventionEngine,observationAdapter,dataAdapter,core,
    analysisAdapter,onProgress:()=>{if(onProgress)onProgress(snapshot())}});
   const prepared=await fresh.create(spec,{feeds,referenceFeeds,profileId:manifest.observation.profileId||'generic'});
   active=id;queue=fresh;return prepared;
  }
  function suiteTests(manifest,tasks){
   const results=[];
   const incomplete=tasks.some(t=>t.status!=='completed');
   for(const e of manifest.evaluations){
    const adapter=registry.requireAdapter('evaluation',e.adapterId);
    if(!adapter.evaluateSuite)continue;
    if(incomplete){results.push({plugin:e.adapterId,test:'suite',verdict:'not_evaluable',
      reason:'One or more experiment tasks did not complete'});continue}
    const result=adapter.evaluateSuite({tasks,config:e.params||{}});
    results.push({plugin:e.adapterId,...json(result)});
   }
   return results;
  }
  function exportEvidence(){
   if(!queue||!active)fail('EMPTY_EXPERIMENT','No scheduled experiment');
   const m=manifests.get(active),s=queue.exportAudit();
   const taskEvidence=s.tasks.map(t=>({
    id:t.id,variant:t.variant,intensity:t.intensity,seed:t.seed,attempts:t.attempts,status:t.status,
    error:t.error,intervention:t.result?.intervention||null,
    target:t.result?.analysis?.target||null,explanation:t.result?.analysis?.method||null,
    tests:t.result?.analysis?.tests||[],observations:t.result?.outputs||[],
    controlObservations:t.result?.matchedControlOutputs||null}));
   return json({schema:'scept.u7.evidence.v1',engine:VERSION,
    experiment:{id:m.id,version:m.version,manifest:m},
    provenance:{modelSha256:s.modelSha256,inputHashes:s.inputHashes,referenceHashes:s.referenceHashes,
     modelId:s.modelId,profileId:s.profileId},
    execution:{status:s.status,total:s.total,completed:s.completed,failed:s.failed,
     cancelled:s.cancelled,pending:s.pending},tasks:taskEvidence,
    suiteTests:suiteTests(m,s.tasks),
    claims:{computational:'verified runtime where completed',clinical:'NOT_ESTABLISHED',causal:'NOT_ESTABLISHED',
     externalValidation:'NOT_PERFORMED'}});
  }
  async function run(options){if(!queue)fail('EMPTY_EXPERIMENT','Prepare an experiment');await queue.run(options);return exportEvidence()}
  function snapshot(){return queue?.snapshot()||{status:'idle',total:0,tasks:[]}}
  function retryFailed(ids){if(!queue)fail('EMPTY_EXPERIMENT','No queue');return queue.retryFailed(ids)}
  function cancel(){if(!queue)fail('EMPTY_EXPERIMENT','No queue');return queue.cancel()}
  function resume(){if(!queue)fail('EMPTY_EXPERIMENT','No queue');return queue.resume()}
  return Object.freeze({VERSION,registerBuiltins:()=>registerBuiltins(registry,core),
   registerExperiment,listExperiments,prepare,run,exportEvidence,snapshot,retryFailed,cancel,resume});
 }
 return Object.freeze({VERSION,SCHEMA,registerBuiltins,validateManifest,make});
});
