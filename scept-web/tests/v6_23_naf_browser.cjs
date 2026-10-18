'use strict';
/* Real public pretrained ViT and independent pretrained CNN, public MRI.
 * NOT patient diagnosis, no fake activation, no A/B experiment.
 */
const assert=require('node:assert/strict'),{chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1490,height:970}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_V6_URL||'http://127.0.0.1:8765/v6.html',{waitUntil:'domcontentloaded'});
  assert.match(await page.title(),/SCEPT V6/);
  assert(!(await page.evaluate(()=>SCEPT_V6.getSnapshot().hasNaf)),'Unmeasured page must start blank');
  await page.waitForFunction(()=>SCEPT_V3?.getSnapshot().ready||
   document.querySelector('#readyLabel').dataset.state==='error',null,{timeout:180000});
  assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,
   'Real pinned ViT + public MRI unavailable: '+await page.locator('#warning').innerText());
  await page.locator('#v6Run').click();
  await page.waitForFunction(()=>SCEPT_V6.getSnapshot().hasNaf||
   document.querySelector('#v6Status').dataset.error==='true',null,{timeout:150000});
  const first=await page.evaluate(()=>({snap:SCEPT_V6.getSnapshot(),naf:SCEPT_V6.getNaf(),record:SCEPT_V6.getRecord()}));
  assert(first.snap.hasNaf,'True ViT capture failed: '+await page.locator('#v6Status').innerText());
  assert.equal(first.snap.architecture,'vit-token-channel');
  assert.equal(first.snap.runCount,1);
  assert.equal(first.snap.groups.filter(g=>g.role==='token').length>=5,true);
  assert.equal(first.snap.groups.filter(g=>g.role==='token').slice(0,5).reduce((n,g)=>n+g.units,0),985);
  assert(first.naf.groups[0].units.some(u=>u.rms>0));
  assert(!first.record.science?.comparison_required);
  assert(first.record.outputs.every(v=>/^[a-f0-9]{64}$/.test(v.sha256)));
  assert(!JSON.stringify(first.record).includes('"raw"'));
  assert.equal(await page.locator('#v6Export').isEnabled(),true);
  assert.equal(await page.locator('#v6BuildReference').isEnabled(),false);
  // Actual in-run residual update only where both real model tensors are exported.
  assert(first.snap.updates.length>=1);
  assert(first.snap.updates[0].status==='measured'||first.snap.updates[0].status==='not_evaluable');
  await page.locator('#v6Metric').selectOption('update');
  assert.match(await page.locator('#v6Counts').innerText(),/单次运行/);
  await page.locator('#v6Metric').selectOption('pattern');
  await page.locator('#v6Fingerprint').scrollIntoViewIfNeeded();
  const bbox=await page.locator('#v6Fingerprint').boundingBox();
  await page.mouse.click(bbox.x+bbox.width*.32,bbox.y+56);
  assert.match(await page.locator('#v6PatternNote').innerText(),/真实|不可评价/);
  await page.locator('#v6Scale').selectOption('reference');
  assert.equal(await page.locator('#v6Scale').inputValue(),'raw','No reference may falsely be shown');
  await page.locator('#sliceK').evaluate(el=>{
   const next=Math.max(Number(el.min),Math.min(Number(el.max),Number(el.value)+3));
   el.value=String(next);el.dispatchEvent(new Event('input',{bubbles:true}));
  });
  await page.locator('#v6Metric').selectOption('state');
  await page.locator('#v6Run').click();
  await page.waitForFunction(()=>SCEPT_V6.getSnapshot().runCount>=2||
   document.querySelector('#v6Status').dataset.error==='true',null,{timeout:150000});
  const second=await page.evaluate(()=>({snap:SCEPT_V6.getSnapshot(),record:SCEPT_V6.getRecord()}));
  assert.equal(second.snap.runCount,2,
   'Second true independent NAF failure: '+await page.locator('#v6Status').innerText());
  assert.notEqual(second.snap.runId,first.snap.runId);
  assert.notEqual(second.record.provenance.inputs.pixel_values,first.record.provenance.inputs.pixel_values);
  assert(!first.snap.hasReference&&!second.snap.hasReference);
  await page.screenshot({path:'/tmp/scept-v6-actual-naf.png',fullPage:true});
  // CNN is an independently pretrained Keras/EfficientNet ONNX architecture,
  // not the synthetic Conv software fixture used in early U2 engineering tests.
  await page.locator('#v6Model').selectOption('cnn');
  await page.locator('#v6Run').click();
  await page.waitForFunction(()=>SCEPT_V6.getSnapshot().runCount>=3||
   document.querySelector('#v6Status').dataset.error==='true',null,{timeout:250000});
  const third=await page.evaluate(()=>({snap:SCEPT_V6.getSnapshot(),cnn:SCEPT_V6_CNN.getSnapshot(),
   record:SCEPT_V6.getRecord()}));
  assert.equal(third.snap.runCount,3,
    'Genuine CNN capture failed: '+await page.locator('#v6Status').innerText());
  assert(['cnn-nhwc','cnn-nchw'].includes(third.snap.architecture));
  assert(third.snap.groups.some(g=>g.units>0),'CNN must expose actual channel or final output feature units');
  assert.equal(third.record.science.clinical,'NOT_ESTABLISHED');
  assert(third.cnn.modelSha256,'Real CNN hash absent');
  assert(third.record.outputs.some(v=>v.name==='dense_2'));
  assert.equal(errors.length,0,errors.join(' | '));
  console.log('PASS genuine V6 NAF: single forward ViT, two independent Run Records, channel details, true residual update/non-evaluable gate, no fake reference, real pretrained CNN, exact SHA provenance; CNN internals exposed '+third.cnn.realIntermediateProbes);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
