'use strict';
/* V6.1: real frozen public pretrained ONNX ViT, actual MNI MRI, ONE inference.
 * No fake A/B, no invented activity, no medical evidence.
 */
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1360,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8765/v5.html',{waitUntil:'domcontentloaded'});
  await page.addScriptTag({url:'http://127.0.0.1:8765/core/scept_run_record.js'});
  await page.addScriptTag({url:'http://127.0.0.1:8765/core/scept_capture_once.js'});
  await page.waitForFunction(()=>SCEPT_V3?.getSnapshot().ready||
    document.querySelector('#readyLabel').dataset.state==='error',null,{timeout:180000});
  assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,'Actual frozen ONNX/MRI not ready: '+await page.locator('#warning').innerText());
  const result=await page.evaluate(async()=>{
   const ctx=SCEPT_V3.getScientificContext();
   const probes=ctx.probes.map(p=>p.name);
   const source=SCEPT_MRI.makeSlice(false,2),dim=[1,3,224,224];
   let count=0;
   // Declared probe shapes are from this versioned, explicitly supported ViT profile.
   const session={sha256:ctx.sha256,
    description:{inputs:[{name:'pixel_values',type:'float32',shape:dim}],
      outputs:[{name:'logits',type:'float32',shape:[1,null]},
        ...probes.map(name=>({name,type:'float32',shape:[1,197,192]}))]},
    async run(feeds){
     count++;
     const r=await ctx.session.run({pixel_values:new ort.Tensor(feeds.pixel_values.type,
       feeds.pixel_values.data,feeds.pixel_values.dims)});
     // Select only REAL returned probes. No placeholder outputs.
     const picked={logits:r.logits};
     for(const name of probes){if(!r[name])throw Error('Missing real ONNX layer '+name);picked[name]=r[name]}
     return picked;
    }};
   const engine=SCEPT_CAPTURE_ONCE.make({
    observation:SCEPT_OBSERVATION,contract:SCEPT_RUN_RECORD});
   const one=await engine.captureOnce({session,feeds:{pixel_values:{type:'float32',dims:dim,data:source}},
     modelId:'verified-natural-image-ViT-Tiny',
     probes:[...probes,'unavailable/fake_probed_layer'],
     profileId:'vit-token-channel',
     preprocessing:{id:'scept-v3-mri-slice',config:{input:'MNI152 MRI',axis:2,
       method:'SCEPT_MRI.makeSlice(false,2)',dims:dim}},
     runtime:{engine:'onnxruntime-web',version:'1.20.1',provider:'wasm'}});
   const record=one.exportRecord(),first=record.observations[0],
     actual=one.getTensor(probes[0]);
   const digest=await SCEPT_CAPTURE_ONCE.tensorSha(actual);
   const serial=JSON.stringify(record);
   const state={count,recordSize:serial.length,runId:record.runId,
    probeCount:record.observations.filter(x=>x.status==='measured').length,
    totalTokenCells:record.observations.filter(x=>x.status==='measured')
      .reduce((s,o)=>s+(o.activity.axisSummaries.token?.length||0),0),
    hiddenShape:actual.dims,firstRms:first.activity.summary.rms,
    firstTokenRms:first.activity.axisSummaries.token[0].rms,
    outputShaMatch:record.outputs.some(o=>o.name===probes[0]&&o.sha256===digest),
    noFake:record.availability.at(-1).status==='not_observed'&&
      record.observations.at(-1).activity===null,
    noRaw:!serial.includes('"data"')&&!serial.includes('"pixel_values"')/* deliberate? input name is in provenance */,
    noComparison:SCEPT_V3.getSnapshot().hasExperiment===false,
    modelHash:record.provenance.model.sha256,sourceHash:record.provenance.inputs.pixel_values};
   state.noRaw=!serial.includes('"data":')&&!serial.includes('"rawTensor":');
   one.release();
   state.cleared=one.isReleased();
   return state;
  });
  assert.equal(result.count,1);
  assert.equal(result.probeCount,5);
  assert.equal(result.totalTokenCells,985);
  assert.deepEqual(result.hiddenShape,[1,197,192]);
  assert(result.firstRms>0&&result.firstTokenRms>0);
  assert(result.outputShaMatch&&result.noFake&&result.noRaw&&result.cleared);
  assert(result.noComparison,'NAF must work without any previous A/B intervention');
  assert.match(result.modelHash,/^[a-f0-9]{64}$/);
  assert.match(result.sourceHash,/^[a-f0-9]{64}$/);
  assert.equal(errors.length,0,'Unexpected page errors: '+errors.join(' | '));
  console.log('PASS V6.1 single REAL ONNX forward: independent NAF, five genuine ViT layer tensors, 985 Token RMS summaries, SHA verified, no A/B, no fake probes. '+JSON.stringify(result));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
