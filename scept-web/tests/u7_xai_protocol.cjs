'use strict';
const assert=require('node:assert/strict');
const CORE=require('../core/scept_core.js'),DATA=require('../core/scept_data_adapter.js'),OBS=require('../core/scept_observation.js');
const U5=require('../core/scept_intervention.js'),U6=require('../core/scept_scheduler.js'),U7=require('../core/scept_xai.js');
(async()=>{
 const registry=CORE.createRegistry(),data=DATA.make(CORE),interventions=U5.make(registry,CORE,data),observer=OBS.make();
 interventions.registerBuiltins();U7.registerBuiltins(registry,CORE);
 const inputs=[{name:'image_data',type:'float32',shape:[1,1,4]}],outputs=[{name:'score',type:'float32',shape:[1,1,4]}];
 const session={description:{inputs,outputs,capabilities:['inference','observations.output']},
  sha256:'a'.repeat(64),manifest:{modelId:'m2-engineering-model',probes:[]},
  async run(feeds){return{score:{type:'float32',dims:[1,1,4],data:new Float32Array(feeds.image_data.data)}}}};
 const feeds={image_data:{type:'float32',dims:[1,1,4],data:new Float32Array([1,2,3,4])}};
 const make=()=>U7.make({registry,session,interventionEngine:interventions,observationAdapter:observer,
  dataAdapter:data,core:CORE,schedulerFactory:U6});
 const sweep={
  schema:U7.SCHEMA,id:'m2-sweep',version:'1.0.0',
  model:{sha256:session.sha256,modelId:'m2-engineering-model'},
  target:{adapterId:'u7.target.tensor-element',tensor:'score',index:0},
  intervention:{pluginId:'u5.occlusion',mode:'sweep',
   config:{region:{bounds:[[0,1],[0,1],[0,1]]}},
   parameter:'fill',intensities:[1,2,3],baseSeed:42},
  observation:{profileId:'generic',tensors:['score']},
  method:{adapterId:'u7.method.target-contrast',params:{}},
  evaluations:[
   {adapterId:'u7.eval.directional-shift',params:{direction:'absolute',minimum:0}},
   {adapterId:'u7.eval.dose-monotonicity',params:{tolerance:0}}
  ]
 };
 const runner=make();
 assert.throws(()=>runner.registerExperiment({...sweep,schema:'unknown'}),e=>e.code==='INVALID_SCHEMA');
 const missing={...sweep,method:{adapterId:'unregistered-explanation'}};
 assert.throws(()=>runner.registerExperiment(missing),e=>e.code==='ADAPTER_NOT_FOUND');
 assert.deepEqual(runner.registerExperiment(sweep),{id:sweep.id,version:sweep.version,schema:sweep.schema});
 assert.throws(()=>runner.registerExperiment(sweep),e=>e.code==='DUPLICATE_EXPERIMENT');
 assert.equal(runner.listExperiments().length,1);
 const badModel=make();
 badModel.registerExperiment({...sweep,id:'wrong-model',model:{sha256:'b'.repeat(64)}});
 await assert.rejects(badModel.prepare('wrong-model',{feeds}),e=>e.code==='MODEL_MISMATCH');
 const badObs=make();badObs.registerExperiment({...sweep,id:'wrong-observation',
  observation:{profileId:'generic',tensors:['invented_layer']}});
 await assert.rejects(badObs.prepare('wrong-observation',{feeds}),e=>e.code==='OBSERVATION_UNAVAILABLE');
 await runner.prepare('m2-sweep',{feeds});
 const report=await runner.run();
 assert.equal(report.schema,'scept.u7.evidence.v1');
 assert.equal(report.execution.completed,3);
 assert.equal(report.execution.failed,0);
 assert.deepEqual(report.tasks.map(t=>t.target.delta),[0,1,2]);
 assert(report.tasks.every(t=>t.explanation?.method==='target-contrast'));
 assert(report.tasks.every(t=>t.tests[0].verdict==='criterion_met'));
 assert.equal(report.suiteTests.length,1);
 assert.equal(report.suiteTests[0].verdict,'criterion_met');
 assert.deepEqual(report.suiteTests[0].points.map(x=>x.absoluteTargetDelta),[0,1,2]);
 assert.equal(report.claims.causal,'NOT_ESTABLISHED');
 assert(!JSON.stringify(report).includes('"data"'),'never export raw input tensors');
 assert.equal(feeds.image_data.data[0],1,'source must not mutate');
 // New method and evaluation packages can be registered without editing U7/U6 code.
 registry.register('explanation',{id:'research.method.signed-score',kind:'explanation',capabilities:['method.explain'],
  calculate:({target})=>({schema:'custom-method.v1',method:'signed-score',score:target.delta*2})});
 registry.register('evaluation',{id:'research.eval.nonzero',kind:'evaluation',capabilities:['evaluation.score'],
  evaluate:({method})=>({schema:'custom-eval.v1',verdict:method.score!==0?'criterion_met':'criterion_not_met',
   test:'signed-nonzero'})});
 const extension={...sweep,id:'external-addon',intervention:{
  pluginId:'u5.occlusion',mode:'ab',config:{region:{bounds:[[0,1],[0,1],[0,1]]},fill:0}},
  method:{adapterId:'research.method.signed-score'},evaluations:[{adapterId:'research.eval.nonzero'}]};
 const ext=make();ext.registerExperiment(extension);await ext.prepare('external-addon',{feeds});
 const evidence=await ext.run();
 assert.equal(evidence.tasks[0].explanation.score,-2);
 assert.equal(evidence.tasks[0].tests[0].verdict,'criterion_met');
 // Mask-based lesion removal is a runnable plugin only when an explicit mask is supplied.
 const maskHash=await CORE.sha256(Uint32Array.from([0,1]));
 const lesion={...extension,id:'lesion-hypothesis-synthetic-not-mri',
  requirements:{lesionMask:true},intervention:{pluginId:'u7.masked-occlusion',mode:'ab',baseSeed:1337,
   config:{fill:0,mask:{inputName:'image_data',shape:[1,1,4],indices:[0,1],
    indicesSha256:maskHash,coordinateSpace:'model-input-flat-indices',sourceId:'synthetic-ROI-not-anatomy'}}}};
 const maskTest=make();maskTest.registerExperiment(lesion);await maskTest.prepare(lesion.id,{feeds});
 const lesionEvidence=await maskTest.run();
 assert.equal(lesionEvidence.execution.completed,1);
 assert.equal(lesionEvidence.tasks[0].intervention.selected,2);
 assert.equal(lesionEvidence.tasks[0].intervention.maskIndicesSha256,maskHash);
 assert.equal(lesionEvidence.tasks[0].intervention.anatomicalValidation,'NOT_VERIFIED');
 const invalidMask=make();invalidMask.registerExperiment({...lesion,id:'bad-mask',
  intervention:{...lesion.intervention,config:{...lesion.intervention.config,
   mask:{...lesion.intervention.config.mask,indicesSha256:'0'.repeat(64)}}}});
 await invalidMask.prepare('bad-mask',{feeds});
 const invalidEvidence=await invalidMask.run();
 assert.equal(invalidEvidence.execution.failed,1);
 assert.equal(invalidEvidence.tasks[0].error.code,'MASK_HASH_MISMATCH');
 assert.equal(invalidEvidence.tasks[0].target,null);
 assert.equal(invalidEvidence.suiteTests.length,0);
 const blocked=make();blocked.registerExperiment({...lesion,id:'missing-mask',
  intervention:{...lesion.intervention,config:{fill:0}}});
 await assert.rejects(blocked.prepare('missing-mask',{feeds}),e=>e.code==='MASK_REQUIRED');
 console.log('PASS U7 unit: manifest-to-U6 execution, independent target/method/evaluation plugins, dose and task evidence, external algorithm extension, sparse mask hypothesis with fail-closed checks and no medical claims');
})().catch(e=>{console.error(e);process.exitCode=1});
