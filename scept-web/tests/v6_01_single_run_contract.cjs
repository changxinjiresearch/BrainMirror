'use strict';
/* V6.0-6.1 synthetic numerical CONTRACT tests only. No empirical XAI claims. */
const assert=require('node:assert/strict');
const observation=require('../core/scept_observation.js');
const record=require('../core/scept_run_record.js');
const capture=require('../core/scept_capture_once.js');
const h=c=>c.repeat(64);
const profile='vit-token-channel';
const a={type:'float32',dims:[1,3,2],data:new Float32Array([1,0,0,1,0,0])};
const b={type:'float32',dims:[1,3,2],data:new Float32Array([2,1,-2,0,1,1])};
const meta={name:'hidden',type:'float32',shape:[1,3,2]};
function session({mutate=false,bad=null}={}){
 let count=0;
 return{sha256:h('a'),description:{inputs:[{name:'image',type:'float32',shape:[1,3,2]}],
  outputs:[meta,{name:'logits',type:'float32',shape:[1,2]}]},
  get count(){return count},
  run:async feeds=>{
   count++;
   if(mutate)feeds.image.data[0]=15;
   return{hidden:bad==='shape'?{type:'float32',dims:[1,6],data:b.data}:
       bad==='nonfinite'?{type:'float32',dims:[1,3,2],data:new Float32Array([NaN,0,1,2,3,4])}:b,
    logits:{type:'float32',dims:[1,2],data:new Float32Array([.3,.7])}};
  }};
}
const protocol={modelId:'synthetic-cpu-contract',probes:['hidden','nonexistent_attention'],
  profileId:profile,preprocessing:{id:'identity-test',config:{normalize:false,resize:[1,3,2]}},
  runtime:{engine:'test-contract-only',version:'0',provider:'cpu-fixture'},
  feeds:{image:{...a,data:new Float32Array(a.data)}}};
const factory=capture.make({observation,contract:record});
(async()=>{
 const s=session();
 const cap=await factory.captureOnce({session:s,...protocol});
 assert.equal(s.count,1,'Capture must call the model ONCE');
 assert.equal(cap.record.schema,'scept.run-record.v1');
 assert.equal(cap.record.science.status,'observation_only');
 assert.equal(cap.record.science.stability,'NOT_EVALUATED');
 assert.equal(cap.record.observations.length,2);
 assert.equal(cap.record.availability[0].status,'measured');
 assert.equal(cap.record.availability[1].status,'not_observed');
 assert.equal(cap.record.observations[1].activity,null);
 assert.equal(cap.record.observations[0].activity.axisSummaries.token.length,3);
 assert.equal(cap.record.observations[0].activity.axisSummaries.token[0].rms,Math.sqrt(2.5));
 assert(cap.record.outputs[0].sha256.match(/^[a-f0-9]{64}$/));
 assert(cap.record.provenance.inputs.image.match(/^[a-f0-9]{64}$/));
 assert(!JSON.stringify(cap.exportRecord()).includes('"data"'));
 const raw=cap.getTensor('hidden');assert.equal(raw.data[0],2);
 raw.data[0]=999;assert.equal(cap.getTensor('hidden').data[0],2,'Retained bytes must be private defensive copy');
 
 cap.release();assert.equal(cap.isReleased(),true);
 assert.throws(()=>cap.getTensor('hidden'),/cleared/);
 // Independent second run exists without A/B; no reliance on previous record.
 const twice=await factory.captureOnce({session:session(),...protocol});
 assert.notEqual(twice.record.runId,cap.record.runId);
 assert.equal(twice.record.provenance.inputs.image,cap.record.provenance.inputs.image);
 const changedFeeds={image:{...a,data:new Float32Array(a.data)}};
 changedFeeds.image.data[2]=9;
 const changed=await factory.captureOnce({session:session(),...protocol,feeds:changedFeeds,retainTensors:false});
 assert.notEqual(changed.record.provenance.inputs.image,twice.record.provenance.inputs.image);
 assert.deepEqual(changed.listStored(),[]);
 await assert.rejects(factory.captureOnce({session:session({bad:'shape'}),...protocol}),/shape|contradicts|signature|mismatch/i);
 await assert.rejects(factory.captureOnce({session:session({bad:'nonfinite'}),...protocol}),/nonfinite|NaN|invalid/i);
 await assert.rejects(factory.captureOnce({session:session({mutate:true}),...protocol,
  feeds:{image:{...a,data:new Float32Array(a.data)}}}),/modified original input/);
 await assert.rejects(factory.captureOnce({session:session(),...protocol,
  limits:{maxTensorElements:2,maxCaptureBytes:1024,maxAxisUnits:3}}),/limit|exceed/i);
 await assert.rejects(factory.captureOnce({session:session(),...protocol,
  probes:['hidden','hidden']}),/unique/);
 await assert.rejects(factory.captureOnce({session:session(),...protocol,
  preprocessing:{id:'unsafe',config:{epsilon:Infinity}}}),/finite/);
 assert.throws(()=>record.record({runId:'pretend',createdAt:'now',provenance:{},
  description:{inputs:[meta],outputs:[meta]},outputs:[],observations:[],
  availability:[],limits:{maxTensorElements:1,maxCaptureBytes:1},claim:'clinical_validity'}));
 const lhs=observation.observeOnce({description:{outputs:[meta]},
  outputs:{hidden:b},probes:['hidden'],profileId:'vit-token-channel'});
 const rhs=observation.observeOnce({description:{outputs:[meta]},
  outputs:{hidden:{...b,data:new Float32Array([-2,-1,2,0,-1,-1])}},
  probes:['hidden'],profileId:'vit-token-channel'});
 assert.equal(lhs[0].activity.summary.rms,rhs[0].activity.summary.rms,
  'Opposite signed channel patterns can share RMS, so score cannot be interpretation');
 // Ensure original U4 paired observation remains available.
 const original=observation.observePair({description:{outputs:[meta]},
   baseline:{hidden:a},edited:{hidden:b},probes:['hidden'],profileId:profile});
 assert.equal(original.length,1);
 console.log('PASS V6.0/6.1: exactly-one-forward independent NAF, source hashes, private raw tensors, no false missing probe, fail-closed bad values and legacy U4 pair regression');
})().catch(e=>{console.error(e);process.exitCode=1});
