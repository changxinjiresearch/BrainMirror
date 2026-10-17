'use strict';
const assert=require('node:assert/strict');
const CORE=require('../core/scept_core.js'),DATA=require('../core/scept_data_adapter.js');
const U4=require('../core/scept_observation.js'),U5=require('../core/scept_intervention.js'),U6=require('../core/scept_scheduler.js');
(async()=>{
 const registry=CORE.createRegistry(),data=DATA.make(CORE),interventions=U5.make(registry,CORE,data);
 interventions.registerBuiltins();
 const observer=U4.make();
 const description={inputs:[{name:'x',type:'float32',shape:[1,1,4]}],
  outputs:[{name:'y',type:'float32',shape:[1,1,4]}],capabilities:['inference']};
 const input={x:{type:'float32',dims:[1,1,4],data:new Float32Array([1,2,3,4])}};
 let calls=0,failNext=false,scheduler,wasCancelled=false;
 const session={description,manifest:{modelId:'unit-fixture',probes:[]},
  sha256:'b'.repeat(64),async run(feeds){
   calls++;
   if(failNext){failNext=false;throw Object.assign(Error('Transient ORT failure'),{code:'ORT_FAIL'})}
   // Test-only stub; the separate Playwright E2E gate verifies actual ORT numeric execution.
   if(wasCancelled){wasCancelled=false;scheduler.cancel()}
   return{y:{type:'float32',dims:[1,1,4],data:new Float32Array(feeds.x.data)}};
  }};
 const progress=[];
 scheduler=U6.make({session,interventionEngine:interventions,observationAdapter:observer,
  dataAdapter:data,core:CORE,onProgress:s=>progress.push({status:s.status,total:s.total})});
 const config={region:{bounds:[[0,1],[0,1],[0,2]]}};
 assert.equal(U6.plan({mode:'ab',pluginId:'u5.occlusion',config}).tasks.length,1);
 assert.throws(()=>U6.plan({mode:'sweep',pluginId:'u5.noise',parameter:'sigma',intensities:Array(129).fill(.1)}),
  e=>e.code==='RESOURCE_LIMIT');
 assert.throws(()=>U6.plan({mode:'sweep',pluginId:'u5.noise',parameter:'unknown',intensities:[1]}),
  e=>e.code==='INVALID_PLAN');
 assert.throws(()=>U6.plan({mode:'batch',pluginId:'u5.noise',controls:[{id:'a'}]}),e=>e.code==='INVALID_PLAN');
 const spec={mode:'sweep',pluginId:'u5.noise',parameter:'sigma',config,
  intensities:[0,0.1,.2],baseSeed:87192};
 const first=U6.plan(spec),again=U6.plan(spec);
 assert.deepEqual(first,again,'planned seeds must be reproducible');
 assert.equal(new Set(first.tasks.map(t=>t.seed)).size,3);
 assert.equal(first.tasks[0].config.sigma,0);
 assert.equal(first.tasks[1].config.sigma,.1);
 assert.equal(first.tasks[1].config.seed,first.tasks[1].seed);
 const created=await scheduler.create(spec,{feeds:input,profileId:'generic'});
 assert.equal(created.total,3);
 assert.equal(created.pending,3);
 assert.equal(created.modelSha256,session.sha256);
 assert.match(created.inputHashes.x.sha256,/^[a-f0-9]{64}$/);
 assert(!JSON.stringify(created).includes('"data"'),'audit must not export input raw tensor');
 assert.equal(input.x.data[0],1);
 let state=await scheduler.run({maxTasks:1});
 assert.equal(state.completed,1);assert.equal(state.pending,2);assert.equal(calls,2);
 assert.equal(state.tasks[0].status,'completed');
 assert.equal(state.tasks[0].result.numberOfForwards,2);
 assert.equal(state.tasks[0].result.outputs[0].changed,0,'zero intensity should match baseline');
 failNext=true;
 state=await scheduler.run({maxTasks:1});
 assert.equal(state.failed,1);assert.equal(state.pending,1);
 assert.equal(state.tasks[1].error.code,'ORT_FAIL');
 assert.equal(state.tasks[1].attempts,1);
 state=scheduler.retryFailed(['T002']);
 assert.equal(state.pending,2);assert.equal(state.tasks[1].seed,first.tasks[1].seed);
 state=await scheduler.run();
 assert.equal(state.completed,3);assert.equal(state.failed,0);
 assert.equal(state.tasks[1].attempts,2);
 assert(state.tasks[2].result.outputs[0].mean_abs_delta>0);
 assert(state.tasks.every(t=>t.result.seed===t.seed));
 assert.equal(progress.some(x=>x.status==='running'),true);
 assert(scheduler.exportCSV().includes('task_id'));
 assert(scheduler.exportCSV().includes('mean_abs_delta'));
 const audit=scheduler.exportAudit();
 audit.tasks[0].status='fake';assert.equal(scheduler.snapshot().tasks[0].status,'completed','audit snapshot must be independent');
 // Batch control with genuine third forward A/B/C and explicit numeric matcher.
 const matched={mode:'batch',pluginId:'u5.matched-control',baseSeed:88,config:{
  region:{bounds:[[0,1],[0,1],[0,1]]},controlRegion:{bounds:[[0,1],[0,1],[1,2]]},meanTolerance:1,fill:0},
  controls:[{id:'matched-1',config:{}},{id:'matched-2',config:{meanTolerance:0}}]};
 state=await scheduler.create(matched,{feeds:input});
 assert.equal(state.total,2);
 state=await scheduler.run();
 assert.equal(state.completed,1);assert.equal(state.failed,1);
 assert.equal(state.tasks[0].result.numberOfForwards,3);
 assert(state.tasks[0].result.matchedControlOutputs.length>0);
 assert.equal(state.tasks[1].error.code,'MATCH_FAILED');
 // Cancel after first ONNX forward: do not falsely mark interrupted experiment as a success.
 state=await scheduler.create({mode:'batch',pluginId:'u5.occlusion',config,
  controls:[{id:'a',config:{}},{id:'b',config:{}}]},{feeds:input});
 wasCancelled=true;state=await scheduler.run();
 assert.equal(state.cancelled,1);assert.equal(state.pending,1);
 assert.equal(state.completed,0);
 state=scheduler.resume();assert.equal(state.pending,2);
 state=await scheduler.run();assert.equal(state.completed,2);
 await scheduler.dispose();
 assert.throws(()=>scheduler.resume(),e=>e.code==='DISPOSED');
 console.log('PASS U6 unit: reproducible independent seeds, A/B sweep, A/B/C controls, progress, failed retry, partial-run, cancellation/resume, hashes and no raw input in audit');
})().catch(e=>{console.error(e);process.exitCode=1});
