'use strict';
/* M2 real browser integration: a newly supplied manifest is loaded without
 * any main-program modification, then U6 executes genuine ONNX and U7 assesses.
 */
const {chromium}=require('playwright'),assert=require('node:assert/strict'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:960}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_U7_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
  await page.locator('#modelPreset').selectOption('local');
  await page.locator('#modelFile').setInputFiles(path.join(__dirname,'generated','u2_cnn.onnx'));
  await page.locator('#architectureProfile').selectOption('cnn-nchw');
  await page.locator('#inspect').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||
    document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:40000});
  assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).inspected,
   await page.locator('#modelStatus').innerText());
  await page.locator('#probes').fill('conv_hidden');
  await page.locator('#load').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||
    document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:50000});
  assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).ready,
   await page.locator('#modelStatus').innerText());
  const snap=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
  const image={type:'float32',dims:[1,1,4,4],data:Array.from({length:16},(_,i)=>i/37)};
  await page.locator('#inputJson').fill(JSON.stringify({image_data:image}));
  const region={bounds:[[0,1],[0,1],[0,1],[0,2]]};
  const manifest={
   schema:'scept.u7.experiment.v1',id:'m2-external-plugin-from-json',
   version:'1.0.0',model:{sha256:snap.hash,modelId:'user-uploaded-onnx',requiredCapabilities:['inference']},
   target:{adapterId:'u7.target.tensor-element',tensor:'conv_output',index:0},
   intervention:{pluginId:'u5.noise',mode:'sweep',parameter:'sigma',config:{region},
    intensities:[0,0.1,0.2],baseSeed:1337},
   observation:{profileId:'cnn-nchw',tensors:['conv_hidden']},
   method:{adapterId:'u7.method.activation-response',params:{topK:2}},
   evaluations:[
    {adapterId:'u7.eval.directional-shift',params:{direction:'absolute',minimum:0}},
    {adapterId:'u7.eval.dose-monotonicity',params:{tolerance:100}}
   ]
  };
  // This exact JSON is the only new experiment plugin; no main program edit.
  await page.locator('#u7Manifest').fill(JSON.stringify(manifest));
  await page.locator('#u7Register').click();
  await page.locator('#u7Prepare').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().total===3||
   document.querySelector('#u7Status').dataset.state==='error',null,{timeout:20000});
  let state=await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Snapshot());
  assert.equal(state.total,3,await page.locator('#u7Status').innerText());
  await page.locator('#u7Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().completed===3||
   document.querySelector('#u7Status').dataset.state==='error',null,{timeout:75000});
  const evidence=await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Evidence());
  assert(evidence,evidence||await page.locator('#u7Status').innerText());
  assert.equal(evidence.execution.completed,3);
  assert.equal(evidence.provenance.modelSha256,snap.hash);
  assert.equal(evidence.experiment.id,manifest.id);
  assert(evidence.tasks.every(t=>t.explanation?.method==='activation-response-ranking'));
  assert(evidence.tasks.every(t=>t.explanation.ranking.some(x=>x.tensor==='conv_hidden')));
  assert(evidence.tasks.every(t=>t.observations.some(x=>x.tensor==='conv_hidden'&&x.source==='selected_real_intermediate')));
  assert.equal(evidence.tasks[0].target.delta,0);
  assert(evidence.tasks[1].target.delta!==0);
  assert.equal(evidence.suiteTests[0].test,'dose-monotonicity');
  assert.equal(evidence.claims.clinical,'NOT_ESTABLISHED');
  assert(!JSON.stringify(evidence).includes('"data":['),'Export must not leak raw tensors');
  // Replay with identical model, inputs, manifest and seeds should reproduce.
  await page.locator('#u7Prepare').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().pending===3,null,{timeout:25000});
  await page.locator('#u7Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().completed===3,null,{timeout:75000});
  const replay=await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Evidence());
  assert.deepEqual(evidence.tasks.map(t=>t.target),replay.tasks.map(t=>t.target));
  // Test a second manifest derived from the same loader with no code modifications:
  // a sparse "lesion removal" hypothesis using an EXPLICIT synthetic tensor mask.
  const hash=await page.evaluate(async()=>SCEPT_CORE.sha256(Uint32Array.from([0,1])));
  const lesion={...manifest,id:'m2-lesion-removal-ENGINEERING-NOT-MRI',
   intervention:{pluginId:'u7.masked-occlusion',mode:'ab',baseSeed:1337,
    config:{fill:0,mask:{inputName:'image_data',shape:[1,1,4,4],indices:[0,1],
     indicesSha256:hash,coordinateSpace:'model-input-flat-indices',sourceId:'synthetic-only-no-lesion'}}},
   requirements:{lesionMask:true},method:{adapterId:'u7.method.target-contrast'},
   evaluations:[{adapterId:'u7.eval.directional-shift',params:{direction:'absolute',minimum:0}}]};
  await page.locator('#u7Manifest').fill(JSON.stringify(lesion));
  await page.locator('#u7Register').click();
  await page.locator('#u7Prepare').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().pending===1||
    document.querySelector('#u7Status').dataset.state==='error',null,{timeout:20000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Snapshot())).pending,1,
   await page.locator('#u7Status').innerText());
  await page.locator('#u7Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().completed===1||
    document.querySelector('#u7Status').dataset.state==='error',null,{timeout:75000});
  const lesionEvidence=await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Evidence());
  assert.equal(lesionEvidence.execution.completed,1,await page.locator('#u7Status').innerText());
  assert.equal(lesionEvidence.tasks[0].intervention.anatomicalValidation,'NOT_VERIFIED');
  assert.equal(lesionEvidence.tasks[0].intervention.maskIndicesSha256,hash);
  assert.equal(lesionEvidence.claims.causal,'NOT_ESTABLISHED');
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-u7-m2-experiment.png',fullPage:true});
  console.log('PASS M2 Chrome real ONNX: two new JSON experiment plugins executed without main edits, target/explanation/evaluation evidence, seeded replay, sparse lesion hypothesis fail-closed semantics');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
