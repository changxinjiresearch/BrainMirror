/* Separate Chrome test: MRI-trained public EfficientNetB3, real NIfTI and paired ROI response. */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({acceptDownloads:true,viewport:{width:1440,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_V3_URL||'http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.SCEPT_MRI?.state.loaded,null,{timeout:100000});
  await page.locator('#modelMode').selectOption('mri_cnn');
  await page.waitForFunction(()=>window.SCEPT_MRI_CNN?.isReady()||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:130000});
  assert(await page.evaluate(()=>SCEPT_MRI_CNN.isReady()),'MRI trained ONNX load failed: '+await page.locator('#warning').innerText());
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>window.SCEPT_MRI_CNN?.getAudit()||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:160000});
  const result=await page.evaluate(()=>SCEPT_MRI_CNN.getAudit());
  assert(result,'MRI trained on real MRI model did not run: '+await page.locator('#warning').innerText());
  assert.equal(result.model_sha256,'320b1ef648f3da5216f6ee0374349324b3cb00a76bebf7dbd1c3796e6155991c');
  assert.equal(result.observations.before.length,4);
  assert(result.observations.delta.some(x=>Math.abs(x)>1e-8),'No genuine model output changed');
  assert(result.mri.source.includes('niivue'),'Expected real MRI atlas input');
  assert.match(result.model_warning,/leakage/);
  await page.locator('#modelMode').selectOption('vit');
  assert.match(await page.locator('#modelInfo').innerText(),/非 MRI/);
  assert.equal(errors.length,0,'Browser exceptions '+JSON.stringify(errors));
  console.log('PASS MRI-TRAINED BROWSER MODEL: original publicly-trained MRI EfficientNetB3, real NIfTI input, paired ROI edit, actual 4-class output differences, leak-warning, correct model switching');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
