'use strict';
/* Genuine remote-public pretrained model + public MNI, local-only workspace.
 * No fabricated NAF and no V6.4 claim. */
const assert=require('node:assert/strict'),{chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1400,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_V65_URL||'http://127.0.0.1:8765/v6.html',{waitUntil:'domcontentloaded'});
  assert.match(await page.title(),/V6.5/);
  assert.equal(await page.locator('#v65Adapter option').count(),3);
  assert.equal(await page.locator('#v65Protocols .v65-route').count(),6);
  assert.equal((await page.evaluate(()=>SCEPT_V65.getSnapshot())).unifiedComparisonReady,false,
    'Do not pretend V6.4 universal record migration exists');
  await page.locator('#v65Adapter').selectOption('cnn');
  assert.equal(await page.locator('#v6Model').inputValue(),'cnn','Model selection must actually wire to true CNN backend');
  await page.locator('#v65Adapter').selectOption('vit');
  assert.equal(await page.locator('#v6Model').inputValue(),'vit');
  await page.waitForFunction(()=>SCEPT_V3?.getSnapshot().ready||
    document.querySelector('#readyLabel').dataset.state==='error',null,{timeout:180000});
  assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,
    await page.locator('#warning').innerText());
  await page.locator('#v6Run').click();
  await page.waitForFunction(()=>SCEPT_V65.getSnapshot().runs.length===1||
    document.querySelector('#v6Status').dataset.error==='true',null,{timeout:180000});
  const snap=await page.evaluate(()=>({v6:SCEPT_V6.getSnapshot(),v65:SCEPT_V65.getSnapshot()}));
  assert(snap.v6.hasNaf,await page.locator('#v6Status').innerText());
  assert.equal(snap.v65.runs.length,1,'One genuine forward must automatically append to journal');
  assert.equal(snap.v65.runs[0].source,'live_runtime');
  assert(snap.v65.runs[0].measuredUnits>=985);
  await page.locator('#v65Note').fill('User-defined research note; no output importance claim');
  await page.locator('#v65SaveNote').click();
  assert.match(await page.locator('#v65Status').innerText(),/笔记/);
  const saved=page.waitForEvent('download');
  await page.locator('#v65ExportAll').click();
  const dl=await saved;
  assert.match(dl.suggestedFilename(),/journal\.json/);
  const json=await page.evaluate(()=>SCEPT_V65.getManifest());
  assert.equal(json.runs.length,1);
  assert.equal(json.runs[0].note,'User-defined research note; no output importance claim');
  assert(!JSON.stringify(json).includes('"data":'),'Workspace must never export private raw voxels');
  assert.equal(json.runs[0].record.science.status,'observation_only');
  // Browser import is a provenance-sensitive operation. Imported archives
  // retain numeric data, but ALWAYS downgrade to unverified outside origin.
  const bundle={...json,runs:json.runs.map(x=>({...x,source:'live_runtime'}))};
  await page.locator('#v65Remove').click();
  assert.equal((await page.evaluate(()=>SCEPT_V65.getSnapshot())).runs.length,0);
  await page.locator('#v65Import').setInputFiles({
   name:'workspace.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(bundle))
  });
  await page.waitForFunction(()=>SCEPT_V65.getSnapshot().runs.length===1);
  const back=await page.evaluate(()=>SCEPT_V65.getSnapshot());
  assert.equal(back.runs[0].source,'imported_unverified',
   'No imported journal can claim local runtime verification');
  assert.match(await page.locator('#v65Runs').innerText(),/未验证真实性/);
  // A generic model option may NOT falsely use the dedicated pretrained ViT runtime.
  await page.locator('#v65Adapter').selectOption('generic');
  assert.equal(await page.locator('#v65Generic').isVisible(),true);
  assert.equal(await page.locator('#v6Run').isDisabled(),true,'Generic ONNX must never silently execute pinned ViT');
  await page.locator('#v65Adapter').selectOption('vit');
  assert.equal(await page.locator('#v6Run').isEnabled(),true,'Real ViT mode restores actual NAF button');
  await page.locator('#v65Adapter').selectOption('generic');
  assert.match(await page.locator('#v65Generic a').getAttribute('href'),/universal\.html/);
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-v65-workspace-actual.png',fullPage:true});
  console.log('PASS SCEPT V6.5 real browser: model-first home, real independent ViT NAF, faithful journal/note/export/import downgrade, functional plugin nav, V6.4 explicitly incomplete');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
