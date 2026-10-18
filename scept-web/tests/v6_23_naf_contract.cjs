'use strict';
/* Synthetic unit fixtures are only contract checks, never experimental findings. */
const assert=require('node:assert/strict'),N=require('../core/scept_naf.js'),
 K=require('../core/scept_naf_calibration.js'),O=require('../core/scept_observation.js'),
 R=require('../core/scept_run_record.js'),C=require('../core/scept_capture_once.js');
const hash=x=>x.repeat(64),mk=(x,model=hash('a'))=>({
 schema:'scept.naf.v1',runId:'run-'+x,profile:'vit-token-channel',
 provenance:{model:{sha256:model},preprocessing:{sha256:hash('b')},inputs:{pixel_values:hash(x)}},
 groups:[{tensor:'hidden',status:'measured',shape:[1,2,3],role:'token',
   units:[{index:0,rms:Number.parseInt(x,16)/100+1},{index:1,rms:.3}]}]
});
(async()=>{
 const feeds={image:{type:'float32',dims:[1,3,2],data:new Float32Array([2,0,1,1,0,0])}};
 const actual={input:{type:'float32',dims:[1,3,2],data:new Float32Array([1,0,0,1,0,0])},
   hidden:{type:'float32',dims:[1,3,2],data:new Float32Array([2,0,1,1,0,0])}};
 let calls=0;
 const ctx={sha256:hash('a'),description:{
  inputs:[{name:'image',type:'float32',shape:[1,3,2]}],
  outputs:[{name:'residual_before',type:'float32',shape:[1,3,2]},
    {name:'residual_after',type:'float32',shape:[1,3,2]}]},
  async run(){calls++;return{residual_before:actual.input,residual_after:actual.hidden}}};
 const engine=C.make({observation:O,contract:R});
 const cap=await engine.captureOnce({session:ctx,feeds,modelId:'real-fixture-execution',
  probes:['residual_before','residual_after'],profileId:'vit-token-channel',
  preprocessing:{id:'identity-numeric-fixture',config:{kind:'fixture'}},
  runtime:{engine:'test-cpu',version:'0',provider:'numeric-fixture'}});
 assert.equal(calls,1);
 const graph={nodes:[{inputs:['residual_before'],outputs:['branch']},
   {inputs:['branch'],outputs:['residual_after']}]};
 const good={inputTensor:'residual_before',outputTensor:'residual_after',graph,
  coordinateSemantics:'same-residual-stream',declaredBy:'unit-known-same-space',
  verifiedIdentity:true};
 const naf=N.create(cap,{profile:'vit-token-channel',probes:['residual_before','residual_after'],
   updates:[good,{...good,verifiedIdentity:false}]});
 assert.equal(naf.groups.length,2);
 assert.equal(naf.groups[0].units.length,3);
 assert.equal(naf.updates[0].status,'measured');
 assert.equal(naf.updates[1].status,'not_evaluable');
 assert(Math.abs(naf.updates[0].units[0].rmsUpdate-Math.sqrt(1/2))<1e-6);
 assert.equal(N.getPattern(cap,'residual_after',0,'vit-token-channel').values.length,2);
 assert.deepEqual(N.getPattern(cap,'residual_after',0,'vit-token-channel').values,[2,0]);
 assert.equal(N.update(cap,{...good,graph:{nodes:[]},profile:'vit-token-channel'}).status,'not_evaluable');
 assert.throws(()=>N.getPattern(cap,'residual_after',3,'vit-token-channel'),/shape/i);
 assert.equal(K.scale(.5,{mode:'raw',rawMax:2}),.25);
 assert.equal(K.scale(.5,{mode:'raw',rawMax:1}),.5);
 assert.throws(()=>K.scale(1,{mode:'raw',rawMax:0}),/max/);
 assert.equal(K.empirical(2,[1,1,2,2,3]),.6);
 assert.throws(()=>K.buildReference([mk('1'),mk('2')],{label:'l',domain:'d'}),/>=5/);
 const refs=['1','2','3','4','5'].map(x=>mk(x));
 const reference=K.buildReference(refs,{label:'within-volume R',domain:'nonclinical synthetic fixture',
  kind:'within-volume'});
 assert.equal(reference.nSamples,5);
 const result=K.calibrated(mk('6'),reference);
 assert.equal(result.length,1);
 assert.equal(result[0].units[0].empiricalRank,1);
 assert.throws(()=>K.buildReference([mk('1'),mk('1'),...refs.slice(1,4)],{label:'l',domain:'d'}),/Duplicated/);
 assert.throws(()=>K.calibrated(mk('6',hash('c')),reference),/incompatible/);
 const cnn={type:'float32',dims:[1,2,2,3],data:new Float32Array(12).map((_,i)=>i-2)};
 const oc=O.observeOnce({description:{outputs:[{name:'cnn',type:'float32',shape:[1,2,2,3]}]},
   outputs:{cnn},probes:['cnn'],profileId:'cnn-nhwc'});
 assert.equal(oc[0].activity.axisSummaries.channel.length,3);
 const cnnN=N.units({observations:oc},'cnn-nhwc','cnn');
 assert.equal(cnnN.role,'channel');assert.equal(cnnN.units.length,3);
 cap.release();
 console.log('PASS V6.2–3 NAF: true single-run state pattern, verified update, CNN native axes, fixed raw scale, 5 distinct provenance-bound reference samples');
})().catch(e=>{console.error(e);process.exitCode=1});
