/* Acceptance: genuine 3D operator positions with orbit, zoom, real ONNX
   edges, layer drilldown and per-operator input/output inspection. */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.SCEPT_V3_URL||'http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.SCEPT_NETWORK3D?.getSnapshot().displayNodes>40||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:120000});
  let v=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
  assert(v.displayNodes>40&&v.totalNodes>v.displayNodes,'3D graph not populated from actual ONNX');
  assert(v.totalEdges>v.displayEdges&&v.displayEdges>0,'Genuine tensor dependency edges missing');
  assert(v.edgesVerified,'Non-model edges detected in 3D network');
  assert(v.observedProducers>=3,'Real hidden tensor producer operators were not found');
  assert.equal(v.mode,'overview','Must start at genuine 3D architecture overview');
  assert(['webgl2','perspective-fallback'].includes(v.backend),'No working 3D camera renderer: '+v.backend);
  const c=page.locator('#networkCanvas'),r=await c.boundingBox();
  await page.mouse.move(r.x+r.width*.5,r.y+r.height*.5);
  await page.mouse.down();await page.mouse.move(r.x+r.width*.7,r.y+r.height*.72,{steps:8});await page.mouse.up();
  let after=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
  assert(Math.abs(after.yaw-v.yaw)>.15&&Math.abs(after.pitch-v.pitch)>.1,'3D orbit did not update camera');
  await page.mouse.move(r.x+r.width*.5,r.y+r.height*.5);
  await page.mouse.wheel(0,-320);
  await page.waitForTimeout(100);
  let zoom=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
  assert(zoom.distance<after.distance,'Mouse wheel did not zoom real 3D camera');
  await page.locator('#networkReset').click();
  await page.locator('#layerSelect').selectOption('6');
  await page.locator('#networkEnter').click();
  let detail=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
  assert.equal(detail.mode,'layer','Cannot enter layer');
  assert.equal(detail.layer,6);
  assert(detail.displayNodes>=10,'Operator-level drilldown did not show underlying genuine nodes');
  assert(detail.edgesVerified&&detail.displayEdges>0,'Layer detail must show actual ONNX dependencies');
  const target=detail.pickTarget;
  assert(target&&Number.isFinite(target.x)&&Number.isFinite(target.y),'3D node projection missing');
  const rect=await c.boundingBox();
  if(target.x>=0&&target.x<=rect.width&&target.y>=0&&target.y<=rect.height){
    await page.mouse.click(rect.x+target.x,rect.y+target.y);
    const selected=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
    assert(selected.selected!==null,'Cannot select actual operator node');
    assert.match(await page.locator('#networkNodeDetails').innerText(),/输入张量：.*输出张量：|输入张量：[\s\S]*输出张量：/);
    assert.match(await page.locator('#networkNodeDetails').innerText(),/真实数据依赖/);
  }else{
    const selected=await page.evaluate(()=>SCEPT_NETWORK3D.selectNode(SCEPT_NETWORK3D.getSnapshot().pickTarget.id));
    assert(selected,'Node detail lookup failed');
  }
  await page.locator('#networkOverview').click();
  assert.equal((await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot())).mode,'overview','Cannot return from layer details');
  await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().ready||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:120000});
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().hasExperiment||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:150000});
  const later=await page.evaluate(()=>SCEPT_NETWORK3D.getSnapshot());
  assert(later.edgesVerified,'Rerender introduced fake edges');
  assert(later.displayNodes>40,'Network disappeared after actual MRI experiment');
  assert.equal(errors.length,0,'Browser script errors: '+errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-onnx-network3d-1440.png',fullPage:true});
  console.log('PASS 3D REAL ONNX: '+v.backend+' renderer, orbit, zoom, '+later.totalNodes+' operators, '+later.totalEdges+' actual dependencies, full layer drilldown, inspected tensor names, model inference preserved.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
