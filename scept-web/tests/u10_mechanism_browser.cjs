'use strict';
/* U10 scientific mechanism E2E with real Chromium + real ONNX Web/WASM.
 * A deterministic tiny untrained CNN verifies software execution only.
 */
const assert=require('node:assert/strict');
const path=require('node:path');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1420,height:1020}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_U10_URL||'http://127.0.0.1:8765/universal.html',
   {waitUntil:'domcontentloaded'});
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
   document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:55000});
  const snap=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
  assert(snap.ready,await page.locator('#modelStatus').innerText());
  const input={type:'float32',dims:[1,1,4,4],data:Array(16).fill(1)};
  await page.locator('#inputJson').fill(JSON.stringify({image_data:input}));
  const ids=await page.evaluate(()=>SCEPT_UNIVERSAL.getU10Ids());
  assert.equal(ids.explanation,'scept.u10.method.evidential-dependence');
  const params={
   inputName:'image_data',targetTensor:'conv_output',targetIndex:0,
   targetSourceId:'synthetic-numeric-target-ROI-NOT-LESION',
   controlSourceId:'synthetic-numeric-control-ROI-NOT-LESION',
   targetIndices:[0,1],controlIndices:[14,15],fill:0,
   nuisanceSigma:.002,meanTolerance:.2,stdTolerance:.2,
   nuisanceSeeds:[101,202,303],strengths:[0,.5,1],
   outputScale:1,minTargetEffect:0,minSpecificity:0,
   minPositiveFraction:0,maxSpread:100,minimumSignAgreement:0,
   expectedSign:null,doseTolerance:100,nullTolerance:.00001
  };
  await page.locator('#u10Protocol').fill(JSON.stringify(params));
  await page.locator('#u10Generate').click();
  await page.waitForFunction(()=>document.querySelector('#u7Manifest').value.includes('scept.u10.method.evidential-dependence')||
   document.querySelector('#u10Status').dataset.state==='error',null,{timeout:18000});
  const manifest=JSON.parse(await page.locator('#u7Manifest').inputValue());
  assert.equal(manifest.model.sha256,snap.hash);
  assert.equal(manifest.intervention.controls.length,9);
  assert.equal(manifest.method.adapterId,ids.explanation);
  assert(manifest.intervention.controls.every(t=>t.config.targetMask.indicesSha256.length===64));
  assert(manifest.intervention.controls.every(t=>t.config.controlMask.indicesSha256.length===64));
  await page.locator('#u7Register').click();
  await page.locator('#u7Prepare').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().pending===9||
   document.querySelector('#u7Status').dataset.state==='error',null,{timeout:30000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Snapshot())).pending,9,
   await page.locator('#u7Status').innerText());
  await page.locator('#u7Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().completed===9||
   (SCEPT_UNIVERSAL.getU7Snapshot().failed>0&&SCEPT_UNIVERSAL.getU7Snapshot().pending===0)||
   document.querySelector('#u7Status').dataset.state==='error',
   null,{timeout:90000});
  const evidence=await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Evidence());
  assert.equal(evidence.execution.completed,9,
   'Unexpected failed task: '+JSON.stringify(evidence.tasks.filter(t=>t.status!=='completed')));
  assert.equal(evidence.execution.failed,0);
  assert.equal(evidence.provenance.modelSha256,snap.hash);
  assert(evidence.tasks.every(t=>t.target.control!==null),'Real third control forward is mandatory');
  assert(evidence.tasks.every(t=>t.intervention.schema==='scept.u10.intervention.v1'));
  assert(evidence.tasks.every(t=>t.explanation.schema==='scept.u10.explanation.v1'));
  assert(evidence.tasks.every(t=>t.explanation.layerResponses.some(v=>v.tensor==='conv_hidden')));
  assert(evidence.tasks.every(t=>t.observations.some(v=>v.evidence==='actual-onnx-runtime-graph-outputs')));
  const suite=evidence.suiteTests.find(t=>t.plugin===ids.suite);
  assert(suite&&suite.gates.length===7,JSON.stringify(evidence.suiteTests));
  assert(['criterion_met','criterion_not_met'].includes(suite.verdict));
  assert(suite.metrics.maxZeroDose<=.00001);
  assert(suite.metrics.doseViolations===0);
  assert.match(await page.locator('#u10Report').innerText(),/criterion_(met|not_met)/);
  const visual=await page.evaluate(()=>SCEPT_UNIVERSAL.getU8Snapshot());
  assert(visual.hasScene&&visual.source==='u7','U10 must drive existing U8 actual ONNX scene');
  assert(!JSON.stringify(evidence).includes('"data":['),
   'Export must not include raw patient/input tensors');
  // Fresh U7 scheduled inference must exactly replay true control/target scores on same seed.
  await page.locator('#u7Prepare').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().pending===9,null,{timeout:25000});
  await page.locator('#u7Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().completed===9,null,{timeout:90000});
  const repeat=await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Evidence());
  assert.deepEqual(repeat.tasks.map(t=>t.target),evidence.tasks.map(t=>t.target));
  assert.deepEqual(repeat.suiteTests[0],evidence.suiteTests[0]);
  assert.deepEqual(repeat.suiteTests[1],evidence.suiteTests[1]);
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-u10-scientific-mechanism.png',fullPage:true});
  console.log('PASS U10 real Chrome ONNX: independent paired ROI plugin, 27 actual A/B/C forwards x2, signed controlled effect, seven preregistered gates, U8 linkage, frozen-model reproducibility; no medical causal claims');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
