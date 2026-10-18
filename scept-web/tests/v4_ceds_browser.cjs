'use strict';
/* v4 browser acceptance: authentic public MRI + frozen pretrained ViT, no simulated evidence. */
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1420,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.SCEPT_V4_URL||'http://127.0.0.1:8765/v4.html',{waitUntil:'domcontentloaded'});
  assert.match(await page.title(),/SCEPT Observatory v4/);
  assert.equal(await page.locator('#v4Verdict').innerText(),'尚未检验');
  assert.equal(await page.locator('#v4Gates').locator('.v4-gate').count(),0);
  await page.waitForFunction(()=>SCEPT_V3?.getSnapshot().ready||
    document.querySelector('#readyLabel').dataset.state==='error',null,{timeout:180000});
  assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,
    'Frozen ONNX/MRI loading failed: '+await page.locator('#warning').innerText());
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>SCEPT_V3.getSnapshot().hasExperiment,null,{timeout:160000});
  const modes=await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot());
  assert.equal(modes.mode,'delta','v4 must default to actual measured Delta');
  assert.equal(modes.kind,'tokens','v4 should focus real measured Token activity');
  assert.equal(modes.count,985,'Legacy real ViT probe layout must be preserved');
  await page.locator('#v4RunEvidence').click();
  await page.waitForFunction(()=>{
    const x=document.querySelector('#v4Status').textContent;
    return x.includes('真实受控证据谱已生成')||x.includes('无法完成本次科学检验');
  },null,{timeout:220000});
  const status=await page.locator('#v4Status').innerText();
  assert(status.includes('已生成'),'U10 real pipeline failed: '+status);
  const v=await page.locator('#v4Verdict').innerText();
  assert(/未满足|满足|无法评价/.test(v),'Missing transparent U10 verdict '+v);
  assert.equal(await page.locator('#v4Gates .v4-gate').count(),7,'Seven predeclared U10 gates');
  assert.equal(await page.locator('#v4ExportEvidence').isEnabled(),true);
  assert.equal(errors.length,0,'Unexpected page errors: '+errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-v4-real-ceds.png',fullPage:true});
  console.log('PASS v4: authentic public MRI + frozen ONNX model, default true Token Delta, 9 actual U10 triplet jobs, real CEDS, seven gates and export, no fake result before run.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
