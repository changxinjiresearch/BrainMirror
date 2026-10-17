'use strict';
/* U10 formal science/algorithm contracts with deterministic transparent MOCK.
 * Does not count as real ONNX or clinical verification. Browser CI below uses ONNX.
 */
const assert=require('node:assert/strict');
const CORE=require('../core/scept_core.js'),DATA=require('../core/scept_data_adapter.js');
const OBS=require('../core/scept_observation.js'),U5=require('../core/scept_intervention.js');
const U6=require('../core/scept_scheduler.js'),U7=require('../core/scept_xai.js');
const SDK=require('../core/scept_sdk.js'),U10=require('../plugins/u10_evidential_dependence.js');
const sha='a'.repeat(64);
const session={sha256:sha,manifest:{modelId:'u10-explicit-test-mock',probes:['hidden']},
 description:{inputs:[{name:'x',type:'float32',shape:[1,8]}],
 outputs:[{name:'scores',type:'float32',shape:[1,1]},
  {name:'hidden',type:'float32',shape:[1,8]}],capabilities:['inference','observations.output','observations.intermediate']},
 async run(feeds){
  const x=feeds.x.data,weights=[5,4,1,1,1,1,.5,.5];
  let y=0;for(let i=0;i<x.length;i++)y+=weights[i]*x[i];
  return{scores:{type:'float32',dims:[1,1],data:Float32Array.of(y)},
   hidden:{type:'float32',dims:[1,8],data:new Float32Array(x)}};
 }};
const feeds={x:{type:'float32',dims:[1,8],data:Float32Array.from([1,1,1,1,1,1,1,1])}};
const mask=async(indices,sourceId)=>({inputName:'x',shape:[1,8],
 sourceId,coordinateSpace:'model-input-flat-indices',
 indices,indicesSha256:await CORE.sha256(Uint32Array.from(indices))});
(async()=>{
 const registry=CORE.createRegistry(),data=DATA.make(CORE),interventions=U5.make(registry,CORE,data);
 interventions.registerBuiltins();U7.registerBuiltins(registry,CORE);
 const installed=U10.install({sdk:SDK,registry,core:CORE});
 assert.equal(installed.length,4);
 assert.equal(registry.requireAdapter('explanation',U10.IDs.explanation).kind,'explanation');
 assert.equal(registry.requireAdapter('intervention',U10.IDs.intervention).kind,'intervention');
 assert.throws(()=>U10.install({sdk:SDK,registry,core:CORE}),e=>e.code==='DUPLICATE_PLUGIN');
 const targetMask=await mask([0,1],'flat-target-experimental-region');
 const controlMask=await mask([6,7],'flat-matched-control-region');
 const params={modelSha256:sha,inputName:'x',shape:[1,8],targetMask,controlMask,
  targetTensor:'scores',targetIndex:0,profileId:'generic',
  nuisanceSigma:.003,nuisanceSeeds:[101,202,303],strengths:[0,.5,1],
  fill:0,meanTolerance:.1,stdTolerance:.1,
  outputScale:10,minTargetEffect:.5,minSpecificity:.5,minPositiveFraction:1,
  maxSpread:.03,minimumSignAgreement:1,expectedSign:-1,doseTolerance:.003,nullTolerance:.00001};
 const manifest=U10.plan(params);
 assert.equal(manifest.intervention.controls.length,9);
 assert.deepEqual(manifest.evaluations.map(x=>x.adapterId),[U10.IDs.perTask,U10.IDs.suite]);
 const make=()=>U7.make({registry,session,interventionEngine:interventions,
  observationAdapter:OBS,dataAdapter:data,core:CORE,schedulerFactory:U6});
 const engine=make();
 engine.registerExperiment(manifest);
 await engine.prepare(manifest.id,{feeds});
 const result=await engine.run();
 assert.equal(result.execution.completed,9,JSON.stringify(result.tasks.filter(t=>t.status!=='completed')));
 assert(result.tasks.every(t=>t.explanation?.method==='SCEPT-Evidential-Dependence-v1'));
 assert(result.tasks.every(t=>t.intervention?.controlCount===2&&t.intervention?.targetCount===2));
 assert(result.tasks.every(t=>t.intervention?.targetMaskSha256===targetMask.indicesSha256));
 assert(result.tasks.every(t=>t.target?.control!==null));
 const suite=result.suiteTests.find(t=>t.plugin===U10.IDs.suite);
 assert.equal(suite.verdict,'criterion_met',JSON.stringify(suite));
 assert.equal(suite.gates.length,7);
 assert(suite.metrics.medianSpecificity>.5);
 assert(suite.metrics.maxZeroDose<.00001);
 assert(suite.metrics.signAgreement===1);
 assert.equal(suite.nReplicates,3);
 assert.equal(suite.metrics.doseViolations,0);
 assert(!JSON.stringify(result.tasks.map(t=>t.explanation)).includes('"data":['),
  'No raw model input in U10 method output');
 const repeat=make();repeat.registerExperiment(manifest);await repeat.prepare(manifest.id,{feeds});
 const rerun=await repeat.run();
 assert.deepEqual(result.tasks.map(t=>t.target),rerun.tasks.map(t=>t.target),
  'Matched nuisance seeded results must replay deterministically');
 assert.deepEqual([...feeds.x.data],[1,1,1,1,1,1,1,1],'Original caller input MUST remain unchanged');
 // Stable response but NO TARGET effect and NO specificity cannot meet evidence gate.
 const fake=JSON.parse(JSON.stringify(result));
 for(const t of fake.tasks){
  if(t.intervention.strength===1){
   t.explanation.targetedEffect=0;t.explanation.matchedControlEffect=0;t.explanation.score=0;
   t.explanation.targetSignedDelta=0;
  }
 }
 const mockedTasks=fake.tasks.map(t=>({status:t.status,id:t.id,seed:t.seed,
  result:{genuineInference:true,intervention:t.intervention,
   analysis:{method:t.explanation,target:t.target}}}));
 const falsePositive=U10.suite({tasks:mockedTasks,config:manifest.evaluations[1].params});
 assert.equal(falsePositive.verdict,'criterion_not_met','Stable zero response is not evidence');
 const deleted=mockedTasks.slice(0,8);
 assert.equal(U10.suite({tasks:deleted,config:manifest.evaluations[1].params}).verdict,'not_evaluable');
 const badSeeds=U10.plan({...params,nuisanceSeeds:[101,101,303]});
 assert.fail('duplicate seeds must not pass plan',badSeeds);
})().catch(e=>{
 if(e.code==='INVALID_PLAN'&&String(e.message).includes('nuisance seeds')){
  console.log('PASS U10: independent SDK plug-ins, real U6/U7 mock transparent target/control test, stable/dose formal gate, zero-effect not falsely validated, reproducibility and fail-closed seeds');
 }else{console.error(e);process.exitCode=1}
});
