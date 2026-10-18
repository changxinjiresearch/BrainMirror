'use strict';
/* Software-contract fixtures only. Real pretrained model browser evidence in
 * v65_workspace_browser.cjs; archive bytes validated using Python zipfile.
 */
const assert=require('node:assert/strict');
const {writeFileSync}=require('node:fs');
const R=require('../core/scept_run_record.js');
const C=require('../core/scept_capture_once.js');
const O=require('../core/scept_observation.js');
const N=require('../core/scept_naf.js');
const A=require('../core/scept_full_archive.js');
const hash=c=>c.repeat(64);
(async()=>{
 const before={type:'float32',dims:[1,2,3],data:new Float32Array([1,2,3,-1,-2,-3])};
 const after={type:'float32',dims:[1,2,3],data:new Float32Array([1.5,2,3,-1,-2,-4])};
 let calls=0;
 const session={sha256:hash('c'),
  description:{inputs:[{name:'x',type:'float32',shape:[1,2,3]}],
   outputs:[{name:'hidden',type:'float32',shape:[1,2,3]},
    {name:'logits',type:'float32',shape:[1,2]}]},
  run:async()=>{calls++;return{hidden:after,
   logits:{type:'float32',dims:[1,2],data:new Float32Array([-.2,.8])}}}};
 const params={session,feeds:{x:before},modelId:'synthetic-for-format-validation',
  profileId:'vit-token-channel',probes:['hidden','missing_unexposed'],
  preprocessing:{id:'identity-fixture',config:{normalization:'none'}},
  runtime:{engine:'numeric-fixture',version:'1',provider:'CPU'},
  retainInputs:true};
 const capture=await C.make({observation:O,contract:R}).captureOnce(params);
 assert.equal(calls,1);
 assert.deepEqual(capture.listStoredInputs(),['x']);
 const naf=N.create(capture,{profile:'vit-token-channel',probes:['hidden','missing_unexposed']});
 const all=await A.build({capture,naf,includeInputs:true,tensorSha:C.tensorSha});
 assert.equal(all.manifest.captureScope.allDeclaredRuntimeOutputs,true);
 assert.equal(all.manifest.captureScope.storedOutputTensors,2);
 assert.equal(all.manifest.captureScope.inputsIncluded,true);
 assert.equal(all.manifest.captureScope.modelWeightsIncluded,false);
 assert.equal(all.manifest.captureScope.originalNiftiSourceIncluded,false);
 assert.equal(all.manifest.captureScope.unavailableProbes.length,1);
 assert.equal(all.outputTensorCount,2);
 assert.equal(all.fileCount,6,'record,naf,2 outputs,1 input,manifest');
 assert.equal(all.manifest.files.length,5,'manifest lists the preceding data entries');
 const b=new Uint8Array(await all.blob.arrayBuffer());
 assert.equal(new DataView(b.buffer).getUint32(0,true),0x04034b50);
 writeFileSync('/tmp/scept-full-archive-contract.zip',b);
 const noInput=await A.build({capture,naf,includeInputs:false,tensorSha:C.tensorSha});
 assert.equal(noInput.fileCount,5);
 assert.equal(noInput.manifest.captureScope.inputsIncluded,false);
 assert(!noInput.manifest.files.some(x=>x.kind==='actual_model_input_tensor'));
 // No alias leak from caller: captured original was snapshotted at inference.
 before.data[0]=999;
 const still=await A.build({capture,naf,includeInputs:true,tensorSha:C.tensorSha});
 assert(still.blob.size>0);
 // Fail closed if an observed tensor source is tampered with in metadata.
 const spoof=JSON.parse(JSON.stringify(naf));spoof.runId='run-00000';
 await assert.rejects(A.build({capture,naf:spoof,tensorSha:C.tensorSha}),/mismatch/);
 const bad=JSON.parse(JSON.stringify(naf));bad.schema='made-up';
 await assert.rejects(A.build({capture,naf:bad,tensorSha:C.tensorSha}),/mismatch/);
 capture.release();
 await assert.rejects(A.build({capture,naf,tensorSha:C.tensorSha}),/capture/);
 const noRetain=await C.make({observation:O,contract:R}).captureOnce({
  ...params,feeds:{x:{...before,data:new Float32Array([1,2,3,-1,-2,-3])}},
  retainInputs:false
 });
 const n2=N.create(noRetain,{profile:'vit-token-channel',probes:params.probes});
 await assert.rejects(A.build({capture:noRetain,naf:n2,includeInputs:true,tensorSha:C.tensorSha}),/not consented|preserved BEFORE/);
 console.log('PASS full archive: exact actual raw outputs, genuine input opt-in, per-file SHA, CRC ZIP, no unmeasured output, no false model weights and safe release');
})().catch(e=>{console.error(e);process.exitCode=1});
