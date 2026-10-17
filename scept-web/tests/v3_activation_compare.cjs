/* Real-data 3D activity regression: A/B synchronized 3D scenes use actual
   hidden tensor values. Frozen weights, locked scales, true zero perturbation.
   No fake response coloring at unmeasured nodes. */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.stack||e.message));
 await page.goto(process.env.SCEPT_V3_URL||'http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().ready||document.getElementById('readyLabel')?.dataset.state==='error',null,{timeout:140000});
 assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,'Not ready for real MRI test');
 await page.locator('#runExperiment').click();
 await page.waitForFunction(()=>SCEPT_V3.getSnapshot().hasExperiment||document.getElementById('readyLabel')?.dataset.state==='error',null,{timeout:160000});
 const first=await page.evaluate(()=>{
   const v=SCEPT_NETWORK3D.getSnapshot(),a=SCEPT_V3.getAudit();
   return{v,runId:a.at,summary:a.layers.map(l=>({layer:l.layer,delta:l.summary.mean,meanA:l.before.reduce((x,y)=>x+Math.abs(y),0)/l.before.length,meanB:l.after.reduce((x,y)=>x+Math.abs(y),0)/l.after.length}))}
 });
 assert.equal(first.v.visualMode,'compare','A/B must be default');
 assert.equal(first.v.measuredCount,5,'Only actual five published ViT probes can carry activation');
 assert.equal(first.v.measuredNodes.length,5,'Overview should contain only 5 illuminated measured producer nodes');
 assert(first.v.measuredNodes.every(n=>n.meanDelta>0),'Measured delta must be real nonzero values');
 assert(first.v.edgesVerified,'Display must retain only exact ONNX dependencies');
 assert(first.v.deltaScale>0&&first.v.signalScale>0);
 for(const p of first.v.measuredNodes){
  const truth=first.summary.find(s=>s.layer===p.layer);
  assert(truth,'Observed node does not correspond to a real probe layer');
  assert(Math.abs(truth.delta-p.meanDelta)<1e-6,'3D delta is not the real hidden tensor mean');
  assert(Math.abs(truth.meanA-p.meanA)<1e-6,'Baseline A does not match the actual tensor');
  assert(Math.abs(truth.meanB-p.meanB)<1e-6,'Modified B does not match actual tensor');
 }
 await page.locator('#responseDelta').click();
 let mode=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
 assert.equal(mode.visualMode,'delta');
 assert.equal(await page.locator('#responseDelta').getAttribute('aria-pressed'),'true');
 const nodeId=mode.measuredNodes.find(n=>n.layer===6).nodeId;
 await page.evaluate(id=>SCEPT_NETWORK3D.selectNode(id),nodeId);
 const desc=await page.locator('#networkNodeDetails').innerText();
 assert.match(desc,/原始 A/);assert.match(desc,/修改 B/);assert.match(desc,/真实平均/);
 await page.locator('#responseBefore').click();
 assert.equal((await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot())).visualMode,'before');
 await page.locator('#responseAfter').click();
 assert.equal((await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot())).visualMode,'after');
 await page.locator('#responseCompare').click();
 assert.equal((await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot())).visualMode,'compare');
 assert.equal(await page.locator('#networkPairLabels').isVisible(),true,'A/B labels must be visible together');
 const visualBefore=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
 await page.locator('#mriAmount').fill('0.55');
 await page.locator('#runExperiment').click();
 await page.waitForFunction(old=>SCEPT_V3.getSnapshot().runId!==old||document.getElementById('readyLabel')?.dataset.state==='error',first.runId,{timeout:160000});
 const second=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
 assert(second.observationVersion>visualBefore.observationVersion,'New real inference did not update 3D visual state');
 assert.equal(second.deltaScale,first.v.deltaScale,'Scale must remain fixed after ROI rerun');
 assert.equal(second.signalScale,first.v.signalScale,'A/B scale must remain shared and fixed');
 assert(second.measuredNodes.some(p=>{
   const prior=first.v.measuredNodes.find(q=>q.layer===p.layer);
   return Math.abs(p.meanDelta-prior.meanDelta)>1e-5;
 }), 'Re-running with different MRI edits did not change measured 3D response');
 await page.locator('#mriAmount').fill('0');
 const old=await page.evaluate(()=>SCEPT_V3.getSnapshot().runId);
 await page.locator('#runExperiment').click();
 await page.waitForFunction(prev=>SCEPT_V3.getSnapshot().runId!==prev||document.getElementById('readyLabel')?.dataset.state==='error',old,{timeout:160000});
 const zero=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
 assert(zero.measuredNodes.every(n=>n.meanDelta===0),'Zero real ROI intensity edit MUST NOT produce fake bright activation deltas');
 assert.equal(zero.deltaScale,first.v.deltaScale,'0-delta test must not auto-calibrate');
 await page.locator('#responseDelta').click();
 const zeroDelta=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
 assert(zeroDelta.measuredNodes.every(n=>Math.abs(n.deltaSize-8)<1e-8),'Zero difference must render as a small neutral node, not fake activity');
 await page.locator('#responseCalibrate').click();
 const calibrated=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
 assert(calibrated.deltaScale>0,'Zero-calibrated scale must remain numerically safe');
 for(const width of [390,768,1440]){
  await page.setViewportSize({width,height:900});
  const d=await page.evaluate(()=>({screen:window.innerWidth,scroll:document.documentElement.scrollWidth}));
  assert(d.scroll<=width+2,'Regression: horizontal overflow '+width+': '+JSON.stringify(d));
 }
 assert.equal(errors.length,0,'Uncaught browser error '+errors.join('\n'));
 console.log('PASS MEASURED 3D A/B: 5 real ViT hidden-tensor probe producers; source-value parity; shared locked scales; changed ROI updates glow; zero-intensity ROI causes exactly zero activation delta; response modes, detail and mobile layouts.');
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
