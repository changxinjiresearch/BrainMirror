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
  // Restored science inspector: real layer select, real token select and all
  // exact underlying channel values, not just coloured RMS bars.
  const layerChooser=page.locator('#v6LayerSelect'),
   tokenChooser=page.locator('#v6UnitSelect'),exact=page.locator('#v6ChannelExact');
  assert((await layerChooser.locator('option').count())>=5,
   'All true ViT measured layers must be selectable');
  assert.match(await layerChooser.locator('option').first().innerText(),/层/);
  await layerChooser.selectOption({index:1});
  assert.equal(await tokenChooser.locator('option').count(),197,
   'Layer selection must populate actual 197 tokens from real ViT');
  await tokenChooser.selectOption('7');
  const orig=await page.evaluate(()=>SCEPT_V6.getSelectedPattern());
  assert.equal(orig.unit,7);
  assert.equal(orig.values.length,192);
  assert.equal(orig.provenance,'verified_same_forward_private_runtime_tensor');
  assert.equal(await page.locator('#v6ChannelIndex').isEnabled(),true);
  await page.locator('#v6ChannelIndex').fill('37');
  await page.locator('#v6ChannelIndex').press('Tab');
  const expected=orig.values[37].toPrecision(9);
  assert((await exact.innerText()).includes(expected),
   'Numeric readout must exactly match the retained real ONNX tensor');
  await page.locator('#v6ValuesDetails summary').click();
  assert.equal(await page.locator('#v6ValuePage').innerText(),'第 2 / 6 页 · 共 192 个实测数值');
  assert((await page.locator('#v6ChannelRows').innerText()).includes(expected),
   'Paged table must expose exact real float32 values');
  await page.locator('#v6ValueNext').click();
  assert.match(await page.locator('#v6ValuePage').innerText(),/第 3 \/ 6 页/);
  await page.locator('#v6ValuePrev').click();
  assert.match(await page.locator('#v6ValuePage').innerText(),/第 2 \/ 6 页/);
  const prevId=await page.evaluate(()=>SCEPT_V6.getRecord().runId);
  await layerChooser.selectOption({index:0});
  const alternate=await page.evaluate(()=>SCEPT_V6.getSelectedPattern());
  assert.notEqual(alternate.tensor,orig.tensor,'Selecting another actual layer must swap source tensor');
  assert.equal(await tokenChooser.inputValue(),'0',
   'Changing layers resets the unit to an actually present token');
  assert.equal((await page.evaluate(()=>SCEPT_V6.getRecord().runId)),prevId,
   'Reading/selecting layers must never trigger a new model forward');
  await page.locator('#v6Metric').selectOption('update');
  assert.equal(await page.locator('#v6ChannelIndex').isDisabled(),true,
   'Update cannot fabricate per-channel differences when not implemented');
  await page.locator('#v6Metric').selectOption('state');
  // NAF 3D orbit operates ONLY on the existing measured tensor units.
  // No mouse gesture can start a new inference or change the frozen record.
  const cloud=page.locator('#v6Cloud');
  await cloud.scrollIntoViewIfNeeded();
  const box=await cloud.boundingBox();
  assert(box&&box.width>300&&box.height>120,'Real 3D viewport must be visible');
  const before=await page.evaluate(()=>({view:SCEPT_V6.getCloudView(),
    run:SCEPT_V6.getSnapshot().runCount,
    record:SCEPT_V6.getRecord().runId,
    selected:document.getElementById('v6Selected').textContent}));
  const origin={x:box.x+box.width*.48,y:box.y+box.height*.55};
  await page.mouse.move(origin.x,origin.y);
  await page.mouse.down();
  await page.mouse.move(origin.x+85,origin.y+42,{steps:8});
  await page.mouse.up();
  const moved=await page.evaluate(()=>({view:SCEPT_V6.getCloudView(),
    run:SCEPT_V6.getSnapshot().runCount,record:SCEPT_V6.getRecord().runId,
    selected:document.getElementById('v6Selected').textContent}));
  assert.notEqual(moved.view.yaw,before.view.yaw,'Horizontal drag must orbit');
  assert.notEqual(moved.view.pitch,before.view.pitch,'Vertical drag must orbit');
  assert.equal(moved.selected,before.selected,'Drag must not select a fake NAF node');
  assert.equal(moved.run,before.run,'Orbit cannot rerun the neural network');
  assert.equal(moved.record,before.record,'Orbit cannot change provenance');
  await page.mouse.wheel(0,-310);
  const wheel=await page.evaluate(()=>SCEPT_V6.getCloudView());
  assert(wheel.zoom>moved.view.zoom,'Wheel over cloud must zoom in');
  await page.mouse.down({button:'right'});
  await page.mouse.move(origin.x+137,origin.y+63,{steps:6});
  await page.mouse.up({button:'right'});
  const panned=await page.evaluate(()=>SCEPT_V6.getCloudView());
  assert.notEqual(panned.panX,wheel.panX,'Right button drag must pan');
  await page.locator('#v6ZoomOut').click();
  const minus=await page.evaluate(()=>SCEPT_V6.getCloudView().zoom);
  assert(minus<panned.zoom,'Zoom-out button must function');
  await page.locator('#v6ZoomIn').click();
  assert((await page.evaluate(()=>SCEPT_V6.getCloudView().zoom))>minus,'Zoom-in button must function');
  await page.locator('#v6Reset').click();
  assert.deepEqual(await page.evaluate(()=>SCEPT_V6.getCloudView()),
    {yaw:0.48,pitch:0.28,zoom:1.35,panX:0,panY:0},
    'Reset must restore original angle, zoom and pan');
  assert.equal((await page.evaluate(()=>SCEPT_V6.getSnapshot().runCount)),1,
    'Camera changes must never invoke another session.run');
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
