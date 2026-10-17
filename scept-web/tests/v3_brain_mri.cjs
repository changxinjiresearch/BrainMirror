/* v3 real public MRI NIfTI + multi-layer frozen pretrained ViT browser acceptance */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
const url=process.env.SCEPT_V3_URL||'http://127.0.0.1:8765/v3.html';
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:900},acceptDownloads:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push('CONSOLE '+m.text())});
  await page.goto(url,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.SCEPT_MRI?.state.loaded||document.querySelector('#mriStatus')?.textContent.includes('失败'),null,{timeout:80000});
  const mri=await page.evaluate(()=>SCEPT_MRI.state);
  assert(mri.loaded,'Authentic MNI NIfTI loading failed: '+await page.locator('#mriStatus').innerText());
  assert(mri.dimensions.length===3&&mri.dimensions.every(n=>n>30),'Not real 3D MRI');
  assert(mri.source.includes('niivue-demo-images'),'Not real template');
  await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().ready||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:110000});
  const init=await page.evaluate(()=>SCEPT_V3.getSnapshot());
  assert(init.ready,'ViT unable to initialize or MRI missing: '+await page.locator('#warning').innerText());
  assert(init.probeLayers.length>=3,'Insufficient independent real hidden layer probes');
  assert(!await page.locator('#runExperiment').isDisabled(),'Start button not enabled');
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().hasExperiment||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:160000});
  const first=await page.evaluate(()=>SCEPT_V3.getSnapshot());
  assert(first.hasExperiment,'Real MRI experiment failed: '+await page.locator('#warning').innerText());
  assert(first.changed>0,'ROI edit did not change observed true ViT hidden activations');
  assert((await page.locator('.layer-row').count())>=3,'No real multilevel activation panel');
  const radius=page.locator('#mriRadius');
  await radius.fill('16');await page.locator('#mriAmount').fill('0.35');
  const next=page.locator('#runProgress');
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>document.querySelector('#runProgress')?.textContent.includes('真实计算完成'),null,{timeout:160000});
  assert((await page.locator('#metricMean').innerText())!=='—','Computed metric missing');
  const dl=page.waitForEvent('download');await page.locator('#saveAudit').click();const file=await dl,stream=await file.createReadStream(),parts=[];for await(const c of stream)parts.push(c);
  const audit=JSON.parse(Buffer.concat(parts).toString('utf8'));
  assert(audit.layers.length>=3);
  assert(audit.layers.every(x=>x.before.length>128&&x.after.length===x.before.length));
  assert(audit.mri.source.includes('niivue'));
  assert.equal(audit.pretrained_source_sha256,'32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3');
  for(const width of [390,768,1024,1366,1440]){
   await page.setViewportSize({width,height:900});
   const v=await page.evaluate(()=>({width:document.documentElement.scrollWidth,inner:window.innerWidth}));
   assert(v.width<=v.inner+2,'Horizontal overflow at width '+width+': '+JSON.stringify(v));
  }
  await page.screenshot({path:'/tmp/scept-v3-brain-mri.png',fullPage:true});
  assert.equal(errors.filter(x=>!x.includes('favicon')).length,0,'Console/browser errors: '+errors.join('\n'));
  console.log('PASS SCEPT V3: real MNI152 MRI NIfTI, orthogonal slices, ROI 3D edits, real pretrained ViT multi-layer responses, rerun with changed intervention, experiment export, responsive layouts');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
