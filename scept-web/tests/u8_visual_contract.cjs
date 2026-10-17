'use strict';
const assert=require('node:assert/strict');
const OBS=require('../core/scept_observation.js'),V=require('../core/scept_visual_data.js');
const data=(dims,seed)=>Float32Array.from({length:dims.reduce((a,b)=>a*b,1)},(_,i)=>i/37+seed);
const actual=(name,shape,profile,edit=0)=>{
 const a={type:'float32',dims:shape,data:data(shape,0)},
 b={type:'float32',dims:shape,data:data(shape,0)};
 if(edit)b.data[2]+=edit;
 return OBS.observeTensor(name,a,b,{type:'float32',shape},profile,'selected_real_intermediate');
};
const sha='a'.repeat(64);
const descriptor={outputs:[{name:'dynamic_token',type:'float32',shape:[1,5,7]}],
 graph:{operators:[{name:'first',op:'MatMul',inputs:['x'],outputs:['dynamic_token']},
  {name:'next',op:'Relu',inputs:['dynamic_token'],outputs:['out']}],
 dependencies:[{from:0,to:1,tensor:'dynamic_token'}]}};
const obs=actual('dynamic_token',[1,5,7],'vit-token-channel',1.5);
assert.equal(obs.axisSummaries.token.length,5);
assert(Math.abs(obs.axisSummaries.token[0].mean_abs_delta-1.5/7)<1e-6);
assert.equal(obs.axisSummaries.token[0].mean_abs_A>=0,true);
const scene=V.createScene({modelSha256:sha,description:descriptor,observations:[obs]});
assert.equal(scene.schema,'scept.u8.scene.v1');
assert.equal(scene.views[0].units.length,5);
assert.equal(scene.views[0].role,'token');
assert.equal(scene.graph.nodes[0].observed.tensor,'dynamic_token');
assert.equal(scene.graph.edges.length,1);
assert.equal(scene.graph.edges[0].tensor,'dynamic_token');
assert.equal(scene.views[0].units[0].delta,obs.axisSummaries.token[0].mean_abs_delta);
assert.equal(scene.graph.nodes[1].observed,null,'unmeasured nodes cannot appear activated');
const cnn=actual('conv_hidden',[1,3,2,4],'cnn-nchw',.5);
const view=V.tensorView(cnn);
assert.equal(view.role,'channel');
assert.equal(view.units.length,3);
assert.equal(view.shape[1],3);
const vol=actual('vol_hidden',[1,2,3,4,5],'cnn-volume-ncdhw',.5);
assert.equal(V.tensorView(vol).units.length,2);
assert.equal(vol.axisSummaries.depth.length,3);
const onlyOut={...descriptor,outputs:[{name:'out',type:'float32',shape:[1,5,7]}]};
assert.throws(()=>V.createScene({modelSha256:sha,description:onlyOut,observations:[obs]}),e=>e.code==='UNEXPOSED_TENSOR');
assert.throws(()=>V.tensorView({...obs,evidence:'synthetic'}),e=>e.code==='UNTRUSTED_OBSERVATION');
assert.throws(()=>V.tensorView({...obs,axisSummaries:{...obs.axisSummaries,token:[]}}),e=>e.code==='AXIS_MISMATCH');
const audit={originalModelSHA256:sha,schema:'scept.core.u5.intervention.v1',outputs:[obs]};
assert.equal(V.fromAudit({audit,description:descriptor,modelSha256:sha}).views[0].role,'token');
assert.throws(()=>V.fromAudit({audit,description:descriptor,modelSha256:'b'.repeat(64)}),e=>e.code==='AUDIT_MISMATCH');
const snapshot={modelSha256:sha,tasks:[{id:'T001',status:'completed',result:{genuineInference:true,outputs:[obs]}},
 {id:'T002',status:'failed',error:{code:'BROKEN'}}]};
assert.equal(V.fromQueue({snapshot,taskId:'T001',description:descriptor,modelSha256:sha}).taskId,'T001');
assert.throws(()=>V.fromQueue({snapshot,taskId:'T002',description:descriptor,modelSha256:sha}),e=>e.code==='TASK_UNAVAILABLE');
const evidence={schema:'scept.u7.evidence.v1',provenance:{modelSha256:sha},experiment:{id:'u8'},
 tasks:[{id:'T001',status:'completed',observations:[obs]}]};
assert.equal(V.fromEvidence({evidence,taskId:'T001',description:descriptor,modelSha256:sha}).provenance.experimentId,'u8');
// A genuine volumetric input: verifies all native dimension slices and absolute
// position of a single numeric edit without relying on a synthetic MRI atlas.
const vdims=[1,1,3,4,5],a=data(vdims,0),b=new Float32Array(a);b[2*4*5+1*5+3]=14.25;
const A={vol:{type:'float32',dims:vdims,data:a}},
 B={vol:{type:'float32',dims:vdims,data:b}};
const capture=V.captureInput({baseline:A,edited:B,inputName:'vol',layout:'NCDHW',maxSide:72});
assert.equal(capture.kind,'model-input-volume');
assert.equal(capture.changedSampled,1);
assert.equal(capture.anatomicalStatus,'UNVERIFIED');
assert.equal(capture.resampled,false);
const sliceA=V.plane(capture,'axial',2);
assert.equal(sliceA.rows,4);assert.equal(sliceA.cols,5);
assert(Math.abs(sliceA.delta[1*5+3]-(14.25-a[2*4*5+1*5+3]))<1e-6,'Float32 visual delta should match actual input to float32 tolerance');
assert.equal(V.plane(capture,'coronal',1).delta[2*5+3],sliceA.delta[8]);
assert.equal(V.plane(capture,'sagittal',3).delta[2*4+1],sliceA.delta[8]);
assert.throws(()=>V.plane(capture,'axial',4),e=>e.code==='INVALID_SLICE');
assert.throws(()=>V.captureInput({baseline:A,edited:B,inputName:'vol',layout:'NHWC'}),e=>e.code==='LAYOUT_MISMATCH');
const imageDims=[1,1,4,4],ia=data(imageDims,0),ib=new Float32Array(ia);ib[5]=9;
const cap2=V.captureInput({baseline:{x:{type:'float32',dims:imageDims,data:ia}},
 edited:{x:{type:'float32',dims:imageDims,data:ib}},inputName:'x',layout:'NCHW'});
assert.equal(cap2.spatialDims,2);
assert.equal(V.plane(cap2,'coronal'),null);
assert(Math.abs(V.plane(cap2,'axial').delta[5]-(9-ia[5]))<1e-6);
const bigDims=[1,1,128,128,128],big=new Float32Array(128**3);
const bigCap=V.captureInput({baseline:{v:{type:'float32',dims:bigDims,data:big}},
 edited:{v:{type:'float32',dims:bigDims,data:big}},inputName:'v',layout:'NCDHW',maxSide:64});
assert.deepEqual(bigCap.sampled,[64,64,64]);
assert.equal(bigCap.resampled,true);
assert.equal(bigCap.changedSampled,0);
console.log('PASS U8: dynamic real token/channel/3D CNN summaries, genuine ONNX-only graph edges, model/experiment provenance, rank-5 orthogonal A/B/Delta, rank-4 2D and fail-closed behavior.');
