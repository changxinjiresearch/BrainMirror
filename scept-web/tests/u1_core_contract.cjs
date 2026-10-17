'use strict';
const assert=require('node:assert/strict');
const Core=require('../core/scept_core.js');
async function reject(fn,code){await assert.rejects(fn,e=>e.code===code,'Expected '+code)}
(async()=>{
 assert.equal(Core.VERSION,'1.0.0-u2');
 const registry=Core.createRegistry();
 const adapter={id:'test-real-adapter',kind:'model',capabilities:['inference','observations.output','graph.read'],
  async inspect(){return {inputs:[{name:'features',type:'float32',shape:[1,4]}],
    outputs:[{name:'actual_output',type:'float32',shape:[1,4]}],graph:{nodes:1,edges:0}}},
  async load(){let disposed=false;return{async run(feeds){if(disposed)throw Error('disposed');return{
      actual_output:{type:'float32',dims:[1,4],data:new Float32Array(feeds.features.data)}}},
      async dispose(){disposed=true}}}
 };
 registry.register('model',adapter);
 assert.equal(registry.list('model')[0].id,'test-real-adapter');
 assert.throws(()=>registry.register('model',adapter),e=>e.code==='DUPLICATE_ADAPTER');
 assert.throws(()=>registry.requireCapability(adapter,'gradients'),e=>e.code==='UNSUPPORTED_CAPABILITY');
 assert.throws(()=>Core.validateTensor({type:'float32',dims:[1,4],data:[0,NaN,2,3]}),e=>e.code==='INVALID_TENSOR');
 assert.throws(()=>Core.validateTensor({type:'float32',dims:[1,4],data:[0,1,2]}),e=>e.code==='INVALID_TENSOR');
 const bytes=new Uint8Array([1,4,9,16]);
 const hash=require('node:crypto').createHash('sha256').update(bytes).digest('hex');
 const oldCrypto=global.crypto;
 if(!global.crypto?.subtle)global.crypto=require('node:crypto').webcrypto;
 await reject(()=>Core.openModel(registry,{modelId:'new',adapterId:adapter.id,sha256:'0'.repeat(64)},bytes),'MODEL_HASH_MISMATCH');
 const m=await Core.openModel(registry,{modelId:'new',adapterId:adapter.id,sha256:hash},bytes);
 assert.equal(m.sha256,hash);
 const original={type:'float32',dims:[1,4],data:new Float32Array([3,1,4,1])};
 const res=await m.run({features:original});
 assert.deepEqual(Array.from(res.actual_output.data),[3,1,4,1]);
 await reject(()=>m.run({other:original}),'INPUT_MISMATCH');
 await reject(()=>m.run({features:{...original,dims:[1,2],data:new Float32Array([1,2])}}),'SHAPE_MISMATCH');
 await reject(()=>m.run({features:original},{outputs:['invented_node']}),'OUTPUT_UNAVAILABLE');
 await m.dispose();
 await reject(()=>m.run({features:original}),'SESSION_DISPOSED');
 assert.deepEqual(Core.KINDS,['model','data','target','observation','intervention','explanation','evaluation']);
 console.log('PASS U1: 7-category adapter registry, no invented capability, verified hash, typed strict feed, output provenance, disposal; explicit refusals.');
})().catch(e=>{console.error(e);process.exit(1)});
