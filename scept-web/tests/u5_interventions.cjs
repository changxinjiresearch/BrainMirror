'use strict';
const assert=require('node:assert/strict');
const CORE=require('../core/scept_core.js');
const DATA=require('../core/scept_data_adapter.js');
const U5=require('../core/scept_intervention.js');
(async()=>{
 const registry=CORE.createRegistry(),data=DATA.make(CORE),engine=U5.make(registry,CORE,data);
 registry.register('data',data);
 assert.deepEqual(engine.registerBuiltins().map(x=>x.id),
  ['u5.noise','u5.occlusion','u5.replace','u5.matched-control','u5.internal-mask']);
 assert.throws(()=>engine.registerBuiltins(),e=>e.code==='DUPLICATE_ADAPTER');
 const signature={inputs:[{name:'image',type:'float32',shape:[1,1,4,4]}],
  outputs:[{name:'y',type:'float32',shape:[1,1]}],capabilities:[]};
 const base={image:{type:'float32',dims:[1,1,4,4],data:Float32Array.from({length:16},(_,i)=>i)}};
 const original=Array.from(base.image.data);
 const region={bounds:[[0,1],[0,1],[0,1],[0,2]]};
 const controlRegion={bounds:[[0,1],[0,1],[1,2],[0,2]]};
 const run=(pluginId,config,referenceFeeds)=>engine.execute({pluginId,description:signature,feeds:base,config,referenceFeeds});
 const n1=await run('u5.noise',{sigma:.2,seed:240216,region}),n2=await run('u5.noise',{sigma:.2,seed:240216,region});
 assert.deepEqual(Array.from(n1.edited.image.data),Array.from(n2.edited.image.data),'same seed and region must replay EXACTLY');
 const n3=await run('u5.noise',{sigma:.2,seed:123,region});
 assert.notDeepEqual(Array.from(n3.edited.image.data),Array.from(n1.edited.image.data),'changed seed needs different perturbation');
 assert.equal(n1.intervention.selected,2);assert.equal(n1.intervention.seed,240216);
 assert.equal(n1.edited.image.data[2],2,'noise must not change outside selected ROI');
 assert.deepEqual(Array.from(base.image.data),original,'caller source input cannot mutate');
 const z=await run('u5.noise',{sigma:0,seed:240216,region});
 assert.equal(z.intervention.changed,0);
 const o=await run('u5.occlusion',{region,fill:0});
 assert.equal(o.edited.image.data[0],0);
 assert.equal(o.edited.image.data[1],0);
 assert.equal(o.edited.image.data[2],2);
 assert.equal(o.intervention.changed,1);
 const ref={image:{type:'float32',dims:[1,1,4,4],data:Float32Array.from({length:16},(_,i)=>100+i)}};
 const repl=await run('u5.replace',{region},ref);
 assert.equal(repl.edited.image.data[0],100);
 assert.equal(repl.edited.image.data[1],101);
 assert.equal(repl.edited.image.data[2],2);
 assert.equal(repl.intervention.changed,2);
 const matched=await run('u5.matched-control',{region,controlRegion,fill:0,meanTolerance:4});
 assert.equal(matched.intervention.primaryCount,matched.intervention.controlCount);
 assert.equal(matched.intervention.baselineMeanDifference,4);
 assert.equal(matched.edited.image.data[1],0);
 assert.equal(matched.control.image.data[4],0);
 assert.equal(matched.control.image.data[1],1);
 assert.equal(matched.intervention.matching.includes('mean'),true);
 await assert.rejects(run('u5.matched-control',{region,controlRegion,fill:0,meanTolerance:3.99}),e=>e.code==='MATCH_FAILED');
 await assert.rejects(run('u5.matched-control',{region,controlRegion:region,fill:0,meanTolerance:5}),e=>e.code==='MATCH_FAILED');
 await assert.rejects(run('u5.occlusion',{region:{bounds:[[0,1],[0,1],[0,1],[1,5]]}}),e=>e.code==='INVALID_REGION');
 await assert.rejects(run('u5.occlusion',{region,fill:Infinity}),e=>e.code==='INVALID_PARAMETER');
 await assert.rejects(run('u5.noise',{sigma:-.2,seed:12}),e=>e.code==='INVALID_PARAMETER');
 await assert.rejects(run('u5.noise',{sigma:.2,seed:2.5}),e=>e.code==='INVALID_PARAMETER');
 await assert.rejects(run('u5.replace',{region}),e=>e.code==='REFERENCE_REQUIRED');
 await assert.rejects(run('u5.replace',{region},{image:{type:'float32',dims:[1,1,2,8],data:new Float32Array(16)}}),e=>e.code==='SHAPE_MISMATCH'||e.code==='SHAPE_MISMATCH'||e.code==='INVALID_TENSOR'||e.code==='INVALID_SHAPE'||e.code==='TYPE_MISMATCH'||e.message.includes('model metadata'));
 await assert.rejects(run('u5.random-invented',{}),e=>e.code==='ADAPTER_NOT_FOUND');
 await assert.rejects(run('u5.internal-mask',{indices:[0]}),e=>e.code==='UNSUPPORTED_CAPABILITY');
 // Spoofed input signature without a verified ONNX Mul must not enable internal interventions.
 const withMask={...signature,inputs:[...signature.inputs,{name:'scept_mask',type:'float32',shape:[1,1,3,3]}]};
 const maskFeeds={image:base.image,scept_mask:{type:'float32',dims:[1,1,3,3],data:new Float32Array(9).fill(1)}};
 await assert.rejects(engine.execute({pluginId:'u5.internal-mask',description:withMask,feeds:maskFeeds,
   config:{indices:[0]}}),e=>e.code==='UNSUPPORTED_CAPABILITY');
 const verified={...withMask,capabilities:['hidden.intervention'],
  internalMasks:[{input:'scept_mask',target:'conv_hidden',source:'conv_hidden__scept_unmasked',
   shape:[1,1,3,3],node:'SCEPT_ActivationMask'}]};
 const internal=await engine.execute({pluginId:'u5.internal-mask',description:verified,feeds:maskFeeds,config:{indices:[0,4],value:0}});
 assert.equal(internal.baseline.scept_mask.data[0],1);
 assert.equal(internal.edited.scept_mask.data[0],0);
 assert.equal(internal.edited.scept_mask.data[4],0);
 assert.equal(internal.edited.scept_mask.data[3],1);
 assert.equal(internal.baseline.image.data[1],1);
 assert.equal(internal.intervention.targetTensor,'conv_hidden');
 assert.equal(internal.intervention.weightsModified,false);
 await assert.rejects(engine.execute({pluginId:'u5.internal-mask',description:verified,feeds:maskFeeds,config:{indices:[9]}}),
   e=>e.code==='INVALID_PARAMETER');
 // Third party plugins can join the same registry but may not mutate baseline or bypass validation.
 registry.register('intervention',{id:'u5.extension-sample',kind:'intervention',capabilities:['input.intervention'],
   apply({baseline}){return {baseline,edited:DATA.prepare(signature,baseline,CORE),metadata:{selected:1,kind:'extension'}}}});
 const extension=await run('u5.extension-sample',{});
 assert.equal(extension.intervention.plugin,'u5.extension-sample');
 assert.deepEqual(Array.from(base.image.data),original);
 console.log('PASS U5: plugin registration, deterministic Gaussian, ROI occlusion/replacement, numerical matched control and fail-closed instrumented internal mask');
})().catch(e=>{console.error(e);process.exitCode=1});