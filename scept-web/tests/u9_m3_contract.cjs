'use strict';
const assert=require('node:assert/strict');
const CORE=require('../core/scept_core.js'),DATA=require('../core/scept_data_adapter.js');
const OBS=require('../core/scept_observation.js'),U5=require('../core/scept_intervention.js');
const U6=require('../core/scept_scheduler.js'),U7=require('../core/scept_xai.js');
const SDK=require('../core/scept_sdk.js'),AUD=require('../core/scept_audit.js');
const PERF=require('../core/scept_performance.js'),EXT=require('../plugins/u9_method_evaluation_sdk_template.js');
const INTERVENTION_EXTENSION=require('../plugins/u9_intervention_sdk_template.js');
(async()=>{
 const registry=CORE.createRegistry(),data=DATA.make(CORE),observations=OBS.make();
 const interventions=U5.make(registry,CORE,data);interventions.registerBuiltins();U7.registerBuiltins(registry,CORE);
 const ext=EXT.install({sdk:SDK,core:CORE,registry});
 assert.equal(ext.length,2);assert.equal(ext[0].schema,'scept.u9.plugin.v1');
 const extension=INTERVENTION_EXTENSION.install({sdk:SDK,core:CORE,registry});
 assert.equal(extension.kind,'intervention');
 assert.equal(registry.requireAdapter('explanation','research.example-numeric-response').kind,'explanation');
 assert.throws(()=>EXT.install({sdk:SDK,core:CORE,registry}),e=>e.code==='DUPLICATE_PLUGIN');
 assert.throws(()=>SDK.validateAdapter({id:'research.insecure',kind:'explanation',capabilities:['method.explain']},CORE),
  e=>e.code==='INVALID_ENTRYPOINT');
 assert.throws(()=>SDK.validateAdapter({id:'research.fake',kind:'evaluation',capabilities:['method.explain'],evaluate(){}},CORE),
  e=>e.code==='MISSING_CAPABILITY');
 assert.equal(SDK.template({kind:'target'}).requiredMethods[0],'capture');
 assert.throws(()=>SDK.template({kind:'unknown'}),e=>e.code==='INVALID_TEMPLATE');
 const session={manifest:{modelId:'m3-unit-mock',probes:['hidden']},sha256:'a'.repeat(64),
  description:{inputs:[{name:'x',type:'float32',shape:[1,4]}],
   outputs:[{name:'scores',type:'float32',shape:[1,4]},{name:'hidden',type:'float32',shape:[1,4]}],
   capabilities:['inference','observations.output','observations.intermediate']},
  async run(input){const x=input.x.data;
   return{scores:{type:'float32',dims:[1,4],data:new Float32Array(x)},
    hidden:{type:'float32',dims:[1,4],data:new Float32Array(x.map(v=>v*2))}}}};
 const feeds={x:{type:'float32',dims:[1,4],data:new Float32Array([1,2,3,4])}};
 const manifest={schema:'scept.u7.experiment.v1',id:'research.m3-unit',version:'1.0.0',
  model:{sha256:session.sha256,modelId:'m3-unit-mock',requiredCapabilities:['inference']},
  target:{adapterId:'u7.target.tensor-element',tensor:'scores',index:0},
  intervention:{pluginId:'u5.occlusion',mode:'ab',config:{region:{bounds:[[0,1],[0,2]]},fill:0},baseSeed:47},
  observation:{profileId:'generic',tensors:['hidden']},
  method:{adapterId:'research.example-numeric-response',params:{scale:2}},
  evaluations:[{adapterId:'research.example-numeric-test',params:{minimum:1}}]};
 const compat=SDK.compatibility({core:CORE,registry,xai:U7,session,manifest});
 assert.equal(compat.compatible,true,JSON.stringify(compat.failures));
 assert.equal(SDK.compatibility({core:CORE,registry,xai:U7,session,
  manifest:{...manifest,model:{sha256:'b'.repeat(64)}}}).compatible,false);
 assert.equal(SDK.compatibility({core:CORE,registry,xai:U7,session,
  manifest:{...manifest,observation:{...manifest.observation,tensors:['nonexistent']}}}).compatible,false);
 assert.equal(SDK.exampleManifest({modelSha256:session.sha256,targetName:'scores'}).schema,'scept.u7.experiment.v1');
 const make=()=>U7.make({registry,session,interventionEngine:interventions,observationAdapter:observations,
  dataAdapter:data,core:CORE,schedulerFactory:U6});
 const exp=make();exp.registerExperiment(manifest);await exp.prepare(manifest.id,{feeds});
 const evidence=await exp.run();
 assert.equal(evidence.execution.completed,1);
 assert.equal(evidence.tasks[0].explanation.method,'example-numeric-response');
 assert.equal(evidence.tasks[0].tests[0].verdict,'criterion_met');
 const bundle=await AUD.create({core:CORE,record:evidence,definition:manifest});
 assert.equal(bundle.kind,'u7');assert.match(bundle.contentSha256,/^[0-9a-f]{64}$/);
 assert(!JSON.stringify(bundle).includes('"data":['));
 const validated=await AUD.verify({core:CORE,bundle,modelSha256:session.sha256,feeds,definition:manifest});
 assert.equal(validated.verified,true);
 const replay=await AUD.replay({core:CORE,bundle,session,feeds,definition:manifest,engine:make()});
 assert.equal(replay.status,'matched');assert(replay.comparedNumbers>3);
 const tampered=JSON.parse(JSON.stringify(bundle));tampered.record.tasks[0].target.delta=-800;
 await assert.rejects(AUD.verify({core:CORE,bundle:tampered,modelSha256:session.sha256,feeds,definition:manifest}),
   e=>e.code==='AUDIT_HASH_MISMATCH');
 await assert.rejects(AUD.verify({core:CORE,bundle,modelSha256:'b'.repeat(64),feeds,definition:manifest}),
   e=>e.code==='MODEL_MISMATCH');
 await assert.rejects(AUD.verify({core:CORE,bundle,modelSha256:session.sha256,feeds,
  definition:{...manifest,id:'research.wrong'}}),e=>e.code==='DEFINITION_MISMATCH');
 const badFeeds={x:{type:'float32',dims:[1,4],data:new Float32Array([2,2,3,4])}};
 await assert.rejects(AUD.verify({core:CORE,bundle,modelSha256:session.sha256,feeds:badFeeds,
  definition:manifest}),e=>e.code==='INPUT_HASH_MISMATCH');
 assert.throws(()=>AUD.compare({kind:'u7',expected:bundle.record,
  observed:{...evidence,tasks:evidence.tasks.map(t=>({...t,target:{...t.target,delta:17}}))}}),
  e=>e.code==='REPLAY_DIVERGENCE');
 assert.throws(()=>AUD.normalize(JSON.parse('{"__proto__":{"unsafe":1}}')),e=>e.code==='UNSAFE_KEY');
 const privacy=await AUD.create({core:CORE,definition:manifest,record:{...evidence,
  experiment:{...evidence.experiment,manifest:{...manifest,extra:{mask:{indices:[1,2,3],
   patientName:'subject-name-private',data:[1,2,3]}}}}}}).catch(e=>e);
 assert.equal(privacy.code,'DEFINITION_MISMATCH','Manifests cannot be changed after the experiment');
 const privacyAudit=await AUD.create({core:CORE,record:{...evidence,tasks:evidence.tasks.map(t=>
   ({...t,intervention:{...t.intervention,indices:[1,2,3],data:[1,2,3],patientId:'PRIVATE'}}))},
  definition:manifest});
 assert(!JSON.stringify(privacyAudit).includes('"indices":[1,2,3]'));
 assert(!JSON.stringify(privacyAudit).includes('PRIVATE'));
 const sdkPlan={mode:'ab',pluginId:'research.input-scale-probe',baseSeed:18,
  config:{inputName:'x',indices:[0,1],factor:0}};
 const sdkQueue=U6.make({session,interventionEngine:interventions,
  observationAdapter:observations,dataAdapter:data,core:CORE});
 await sdkQueue.create(sdkPlan,{feeds});await sdkQueue.run();
 const sdkResult=sdkQueue.exportAudit();
 assert.equal(sdkResult.completed,1);
 assert.equal(sdkResult.tasks[0].result.intervention.plugin,'research.input-scale-probe');
 assert(sdkResult.tasks[0].result.outputs[0].mean_abs_delta>0);
 assert.deepEqual([...feeds.x.data],[1,2,3,4],'SDK intervention cannot alter original input');
 const plan={mode:'ab',pluginId:'u5.occlusion',baseSeed:33,
  config:{region:{bounds:[[0,1],[0,2]]},fill:0}};
 const queue=()=>U6.make({session,interventionEngine:interventions,
  observationAdapter:observations,dataAdapter:data,core:CORE});
 const u6=queue();await u6.create(plan,{feeds});await u6.run();
 const u6bundle=await AUD.create({core:CORE,record:u6.exportAudit(),definition:plan});
 assert.equal(u6bundle.kind,'u6');
 assert.equal((await AUD.replay({core:CORE,bundle:u6bundle,session,feeds,definition:plan,engine:queue()})).status,'matched');
 await assert.rejects(AUD.create({core:CORE,record:{...u6.exportAudit(),
  tasks:u6.exportAudit().tasks.map(t=>({...t,status:'failed'}))},definition:plan}),
  e=>e.code==='INCOMPLETE_EXPERIMENT');
 const perf=await PERF.benchmark({session,feeds,runs:3,warmup:1,maxInputMiB:1});
 assert.equal(perf.status,'measured');assert.equal(perf.timesMs.length,3);
 assert(perf.meanMs>=0);assert(perf.inputMiB<1);
 await assert.rejects(PERF.benchmark({session,feeds,runs:200}),e=>e.code==='RESOURCE_LIMIT');
 console.log('PASS U9 M3 contract: independent plugin SDK, compatibility, content hash audit, U6/U7 real reruns, tamper/privacy fail-closed, numeric divergence and measured inference budgets');
})().catch(e=>{console.error(e);process.exitCode=1});
