'use strict';
/* U2 actual public PRETRAINED models, tested through SAME Universal Core.
 * Prior MLP/CNN fixtures test previously unknown shapes and exact arithmetic;
 * this gate verifies the universal adapter also accepts actual frozen weights.
 */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.stack||e.message));
 await page.goto(process.env.SCEPT_U2_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
 for(const item of [
  {preset:'vit',expected:'pretrained-vit-tiny',inputName:'pixel_values',size:224,probe:'/vit/encoder/layer.6/attention/attention/value/MatMul_output_0'},
  {preset:'cnn',expected:'mri-cnn-experimental',inputName:'input',size:256,probe:null}
 ]){
  await page.locator('#modelPreset').selectOption(item.preset);
  await page.locator('#inspect').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:130000});
  assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).inspected,'Real model inspection failed: '+await page.locator('#modelStatus').innerText());
  if(item.probe)await page.locator('#probes').fill(item.probe);
  else await page.locator('#probes').fill('');
  // Explicitly resolve the ONNX dynamic batch dimension for each model.
  // Neither the universal core nor an arbitrary model adapter should guess it.
  const dims=item.preset==='cnn'?[1,256,256,3]:[1,3,224,224];
  await page.locator('#shapeJson').fill(JSON.stringify({[item.inputName]:dims}));
  await page.locator('#load').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:160000});
  const loaded=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
  assert(loaded.ready,'Frozen pretrained model could not run in generic adapter: '+await page.locator('#modelStatus').innerText());
  assert.equal(loaded.modelId,item.expected);
  assert(loaded.inputNames.includes(item.inputName),'Different architecture signature was not dynamically understood');
  if(item.probe)assert(loaded.outputNames.includes(item.probe));
  await page.locator('#run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().hasExperiment||document.querySelector('#runStatus').dataset.state==='error',null,{timeout:160000});
  const audit=await page.evaluate(()=>SCEPT_UNIVERSAL.getAudit());
  assert(audit,'Paired public pretrained ONNX inference not completed: '+await page.locator('#runStatus').innerText());
  assert(audit.outputs.every(o=>o.total>0&&o.first12.length>0&&o.mean_abs_delta>=0));
  assert(audit.outputs.some(o=>o.changed>0),'Input perturbation did not reach any real observed output for '+item.preset);
  assert.equal(audit.originalModelSHA256,loaded.hash);
  assert.equal(audit.inputSource,'deterministic-synthetic-nonmedical');
  console.log('PASS U2 SAME CORE real PRETRAINED '+item.preset+' weights SHA='+loaded.hash+' inputs='+loaded.inputNames+' observed_outputs='+audit.outputs.length);
 }
 assert.equal(errors.length,0,'Browser exceptions '+errors.join(' | '));
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
