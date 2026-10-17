'use strict';
/* U2 generic ONNX new-model acceptance: two previously unseen architectures.
 * Both models are created independently using Python ONNX tooling and must
 * execute ACTUAL ORT/WASM operations, including true intermediate graph probes. */
const {chromium}=require('playwright'),assert=require('node:assert/strict'),path=require('node:path');
(async()=>{
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.stack||e.message));
 await page.goto(process.env.SCEPT_U2_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
 for(const item of [{file:'u2_mlp.onnx',inName:'features',probe:'mm_hidden',dims:[1,4],out:'scores'},
                    {file:'u2_cnn.onnx',inName:'image_data',probe:'conv_hidden',dims:[1,1,4,4],out:'conv_output'}]){
  await page.locator('#modelPreset').selectOption('local');
  await page.locator('#modelFile').setInputFiles(path.join(__dirname,'generated',item.file));
  await page.locator('#inspect').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||document.querySelector('#modelStatus')?.dataset.state==='error',null,{timeout:40000});
  let initial=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
  assert(initial.inspected,'Could not inspect '+item.file+': '+await page.locator('#modelStatus').innerText());
  await page.locator('#probes').fill('FAKE_INTERMEDIATE_TENSOR');
  await page.locator('#load').click();
  await page.waitForFunction(()=>document.querySelector('#modelStatus')?.dataset.state==='error',null,{timeout:30000});
  assert.match(await page.locator('#modelStatus').innerText(),/not an ONNX annotated/);
  await page.locator('#probes').fill(item.probe);
  await page.locator('#load').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||document.querySelector('#modelStatus')?.dataset.state==='error',null,{timeout:50000});
  const ready=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
  assert(ready.ready,'Generic adapter could not load '+item.file+': '+await page.locator('#modelStatus').innerText());
  assert.deepEqual(ready.inputNames,[item.inName],'Hard-coded architecture-specific input leaked into SCEPT Core');
  assert(ready.outputNames.includes(item.out)&&ready.outputNames.includes(item.probe),'No true exposed intermediate');
  await page.locator('#run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().hasExperiment||document.querySelector('#runStatus')?.dataset.state==='error',null,{timeout:60000});
  const audit=await page.evaluate(()=>SCEPT_UNIVERSAL.getAudit());
  assert(audit,'No real ONNX results: '+await page.locator('#runStatus').innerText());
  assert.equal(audit.adapter,'onnx-wasm');
  assert.equal(audit.inputSource,'deterministic-synthetic-nonmedical');
  assert.deepEqual(audit.inputs[0].shape,item.dims);
  const y=audit.outputs.find(o=>o.tensor===item.out),hidden=audit.outputs.find(o=>o.tensor===item.probe);
  assert(y&&hidden,'Result missing real output and intermediate');
  assert(hidden.mean_abs_delta>0&&hidden.changed>0,'Intermediate not updated after genuine input edit');
  assert(y.max_abs_delta>0,'Actual output failed to reflect input change');
  if(item.file==='u2_mlp.onnx'){
   const expected=(0*.7 + 1/37*.2 + 2/37*(-.4) + 3/37*.3);
   assert(Math.abs(hidden.first12[0].A-expected)<1e-6,'MatMul hidden value differs from independent formula');
   assert(Math.abs(hidden.first12[0].delta-.07)<1e-6,'Actual MatMul perturbation effect differs from 0.1*0.7');
  }else{
   const expected=0*.6+(1/37)*(-.1)+(4/37)*.25+(5/37)*.5;
   assert(Math.abs(hidden.first12[0].A-expected)<1e-6,'Convolution output not equal to independent kernel evaluation');
   assert(Math.abs(hidden.first12[0].delta-.06)<1e-6,'Convolution perturbed first pixel does not contribute 0.1*0.6');
  }
  console.log('PASS U2 truly generic architecture '+item.file+': inputs='+ready.inputNames.join(',')+
    ' outputs='+ready.outputNames.join(',')+' hiddenDelta='+hidden.mean_abs_delta);
 }
 assert.equal(errors.length,0,'Browser exceptions '+errors.join('; '));
 await page.screenshot({path:'/tmp/scept-universal-u2.png',fullPage:true});
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
