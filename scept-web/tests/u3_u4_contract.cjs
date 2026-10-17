'use strict';
const assert=require('node:assert/strict');
const core=require('../core/scept_core.js');
const DATA=require('../core/scept_data_adapter.js');
const OBS=require('../core/scept_observation.js');
const registry=core.createRegistry(),adapter=DATA.make(core);
registry.register('data',adapter);
const observer=OBS.make();registry.register('observation',observer);
assert.equal(registry.requireAdapter('observation','runtime-axis-v1'),observer);
assert.equal(registry.requireAdapter('data','typed-tensor-v1'),adapter);
const desc={inputs:[{name:'features',type:'float32',shape:[1,4]}],outputs:[]};
const original={features:{type:'float32',dims:[1,4],data:new Float32Array([1,2,3,4])}};
const feeds=adapter.prepare(desc,original);
original.features.data[0]=99;
assert.equal(feeds.features.data[0],1,'prepared tensor must be copied');
const {edited,intervention}=adapter.offset(feeds,desc,{delta:.25});
assert.equal(intervention.elementCount,1);
assert.equal(edited.features.data[0],1.25);
assert.equal(feeds.features.data[0],1,'baseline must stay immutable');
assert.throws(()=>adapter.prepare(desc,{bad:original.features}),/signature mismatch|Expected/);
assert.throws(()=>adapter.offset(feeds,desc,{inputName:'unknown',delta:1}),/Unknown/);
const shape=[1,5,7],n=shape.reduce((a,b)=>a*b,1),a=new Float32Array(n),b=new Float32Array(n);
b[7]=2;b[8]=1;
const model={outputs:[{name:'real_hidden',type:'float32',shape}]};
const observations=observer.observePair({description:model,baseline:{real_hidden:{type:'float32',dims:shape,data:a}},
 edited:{real_hidden:{type:'float32',dims:shape,data:b}},probes:['real_hidden'],profileId:'vit-token-channel'});
const o=observations[0];
assert.equal(o.total,35);
assert.equal(o.changed,2);
assert.equal(o.source,'selected_real_intermediate');
assert.equal(o.axisSummaries.token.length,5,'dynamic token count must not be 197');
assert.equal(o.axisSummaries.channel.length,7,'dynamic channel count must not be 192');
assert.equal(o.axisSummaries.token[1].changed,2);
assert.equal(o.mean_abs_delta,3/35);
assert.deepEqual(o.axes.map(x=>x.role),['batch','token','channel']);
assert.throws(()=>OBS.observePair({description:model,baseline:{real_hidden:{type:'float32',dims:shape,data:a}},
 edited:{real_hidden:{type:'float32',dims:shape,data:b}},profileId:'legacy-vit-197x192'}),e=>e.code==='LEGACY_LAYOUT_MISMATCH');
assert.throws(()=>OBS.observePair({description:model,baseline:{real_hidden:{type:'float32',dims:shape,data:a}},
 edited:{real_hidden:{type:'float32',dims:shape,data:b}},probes:['invented'],profileId:'vit-token-channel'}),e=>e.code==='PROBE_NOT_EXPOSED');
const cshape=[1,3,2,4],csize=24,ca=new Float32Array(csize),cb=new Float32Array(csize);
cb[8]=3;
const cnn=OBS.observePair({description:{outputs:[{name:'cnn_hidden',type:'float32',shape:cshape}]},
 baseline:{cnn_hidden:{type:'float32',dims:cshape,data:ca}},
 edited:{cnn_hidden:{type:'float32',dims:cshape,data:cb}},
 probes:['cnn_hidden'],profileId:'cnn-nchw'})[0];
assert.equal(cnn.axisSummaries.channel.length,3);
assert.equal(cnn.axisSummaries.height.length,2);
assert.equal(cnn.axisSummaries.width.length,4);
assert.equal(cnn.axisSummaries.channel[1].changed,1);
assert.equal(cnn.axisSummaries.channel[0].changed,0);
assert.equal(cnn.max_abs_delta,3);
const nhwc=OBS.observePair({description:{outputs:[{name:'c',type:'float32',shape:[1,2,4,3]}]},
 baseline:{c:{type:'float32',dims:[1,2,4,3],data:ca}},
 edited:{c:{type:'float32',dims:[1,2,4,3],data:cb}},profileId:'cnn-nhwc'})[0];
assert.equal(nhwc.axisSummaries.channel.length,3);
assert.throws(()=>OBS.observeTensor('bad',{type:'float32',dims:[1,2],data:new Float32Array([0,1])},
 {type:'float32',dims:[1,2],data:new Float32Array([0,NaN])},null,'generic','actual_graph_output'),
 e=>e.code==='NONFINITE_ACTIVATION');
console.log('PASS U3 typed feed contract, immutable paired intervention; U4 dynamic 5×7 ViT, NCHW/NHWC CNN and fail-closed legacy fixed 197×192.');
