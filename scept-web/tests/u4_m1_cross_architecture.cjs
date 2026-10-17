'use strict';
/* M1: Current verified pretrained Tiny ViT plus independently created Conv/Relu
 * are executed through the EXACT SAME U1/U2 core, U3 feed and U4 observer.
 * CNN fixture is intentionally NOT represented as pretrained or biomedical. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.SCEPT_U4_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
  async function runModel(cfg){
   await page.locator('#modelPreset').selectOption(cfg.preset);
   if(cfg.file)await page.locator('#modelFile').setInputFiles(path.join(__dirname,'generated',cfg.file));
   await page.locator('#architectureProfile').selectOption(cfg.profile);
   await page.locator('#inspect').click();
   await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:150000});
   assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).inspected,await page.locator('#modelStatus').innerText());
   await page.locator('#probes').fill(cfg.probe);
   await page.locator('#shapeJson').fill(JSON.stringify({[cfg.inputName]:cfg.dims}));
   await page.locator('#load').click();
   await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:180000});
   const snap=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
   assert(snap.ready,'Session failed: '+await page.locator('#modelStatus').innerText());
   assert.equal(snap.profile,cfg.profile);
   assert(snap.outputNames.includes(cfg.probe),'True internal tensor not exposed by ONNX graph');
   await page.locator('#run').click();
   await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().hasExperiment||document.querySelector('#runStatus').dataset.state==='error',null,{timeout:180000});
   const audit=await page.evaluate(()=>SCEPT_UNIVERSAL.getAudit());
   assert(audit,'Actual dual inference failed: '+await page.locator('#runStatus').innerText());
   assert.equal(audit.architectureProfile,cfg.profile);
   assert.equal(audit.dataAdapter,'typed-tensor-v1');
   assert.equal(audit.observationAdapter,'runtime-axis-v1');
   assert.equal(audit.adapter,'onnx-wasm');
   assert.equal(audit.originalModelSHA256,snap.hash);
   assert.equal(audit.inputSource,'deterministic-synthetic-nonmedical');
   const hidden=audit.outputs.find(o=>o.tensor===cfg.probe);
   assert(hidden&&hidden.source==='selected_real_intermediate','No actual ONNX internal observation');
   assert(hidden.changed>0&&hidden.mean_abs_delta>0,'Internal tensor did not respond to the real input edit');
   assert.deepEqual(hidden.axes.map(a=>a.role),cfg.roles);
   for(const axis of hidden.axes){
    assert.equal(hidden.axisSummaries[axis.role].length,hidden.shape[axis.index],
     'Axis output must be derived from actual model shape');
   }
   console.log('PASS M1 '+cfg.profile+' actual frozen inference + genuine intermediate '+cfg.probe+
     ' observed shape=['+hidden.shape+'] |Δ|='+hidden.mean_abs_delta+' modelSHA='+snap.hash);
   return{audit,hidden};
  }
  // Actual pretrained ViT: provenance verified against pinned SHA in universal.js
  const vit=await runModel({preset:'vit',profile:'vit-token-channel',probe:'/vit/encoder/layer.6/attention/attention/value/MatMul_output_0',
   inputName:'pixel_values',dims:[1,3,224,224],roles:['batch','token','channel']});
  // Conv/Relu with genuine ONNX runtime-computed intermediate graph tensor.
  const cnn=await runModel({preset:'local',file:'u2_cnn.onnx',profile:'cnn-nchw',
   probe:'conv_hidden',inputName:'image_data',dims:[1,1,4,4],roles:['batch','channel','height','width']});
  assert.equal(cnn.hidden.total,9,'CNN hidden data must have actual 3×3 spatial layout');
  assert(Math.abs(cnn.hidden.first12[0].delta-.06)<1e-6,'CNN ONNX convolution numerical reference mismatch');
  assert.notDeepEqual(vit.hidden.axes.map(x=>x.role),cnn.hidden.axes.map(x=>x.role),
   'Architecture labels must not be hardcoded to ViT');
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-u4-m1-cross-architecture.png',fullPage:true});
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
