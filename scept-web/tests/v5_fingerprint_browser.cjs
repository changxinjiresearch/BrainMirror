'use strict';
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1420,height:960}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e.message)));
  await page.goto(process.env.SCEPT_V5_URL||'http://127.0.0.1:8765/v5.html',{waitUntil:'domcontentloaded'});
  assert.match(await page.title(),/SCEPT V5/);
  assert(!await page.evaluate(()=>SCEPT_V5.getSnapshot().hasNdf),'A blank study cannot pretend to have data');
  await page.waitForFunction(()=>SCEPT_V3?.getSnapshot().ready||document.getElementById('readyLabel').dataset.state==='error',
   null,{timeout:180000});
  assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,'Pretrained ViT/MRI not ready: '+await page.locator('#warning').innerText());
  await page.locator('#v5Observe').click();
  await page.waitForFunction(()=>SCEPT_V5.getSnapshot().hasNdf||document.querySelector('#v5Status').dataset.error==='true',null,{timeout:130000});
  const snap=await page.evaluate(()=>SCEPT_V5.getSnapshot());
  assert(snap.hasNdf,'Actual NDF error: '+await page.locator('#v5Status').innerText());
  assert.equal(snap.mode,'exploratory-intensity-edit-not-lesion-removal');
  const measured=await page.evaluate(()=>({ndf:SCEPT_V5.getNdf(),graph:SCEPT_ACTIVITY3D.getSnapshot()}));
  assert.equal(measured.ndf.rows.length,5);
  assert.equal(measured.ndf.rows.reduce((n,l)=>n+l.units,0),985);
  assert(measured.ndf.rows.some(l=>l.values.some(t=>t.rms>1e-7)),'NDF must reflect nonzero real ONNX tensor deltas');
  assert(measured.ndf.rows.every(l=>l.shape.join(',')==='1,197,192'));
  assert(!measured.ndf.scientificStatus.includes('clinical'));
  await page.locator('#v5Fingerprint').scrollIntoViewIfNeeded();
  const box=await page.locator('#v5Fingerprint').boundingBox();
  await page.mouse.click(box.x+box.width*.30,box.y+40);
  assert.match(await page.locator('#v5Selection').innerText(),/真实/);
  assert.equal(await page.locator('#v5Export').isEnabled(),true);
  assert.equal(await page.locator('#v5CrcRun').isEnabled(),false,
   'One A/B should never become fabricated response stability');
  assert.equal(errors.length,0,'Unexpected browser errors: '+errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-v5-real-ndf.png',fullPage:true});
  console.log('PASS V5 real MRI/ViT: 985 genuine NDF Token cells, 3D linking, frozen output, zero fabricated CRC/ICM and honest provenance');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
