'use strict';
/* U5 genuine Chrome ONNX/WASM A/B/C plugin acceptance.
 * The generated CNN is untrained and only an engineering fixture. */
const {chromium}=require('playwright'),assert=require('node:assert/strict'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1250,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_U5_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
  async function load(name,probe){
   await page.locator('#modelPreset').selectOption('local');
   await page.locator('#modelFile').setInputFiles(path.join(__dirname,'generated',name));
   await page.locator('#architectureProfile').selectOption('cnn-nchw');
   await page.locator('#inspect').click();
   await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:40000});
   assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).inspected,await page.locator('#modelStatus').innerText());
   await page.locator('#probes').fill(probe);
   await page.locator('#load').click();
   await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:40000});
   assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).ready,await page.locator('#modelStatus').innerText());
  }
  const image={type:'float32',dims:[1,1,4,4],data:Array.from({length:16},(_,i)=>i)};
  await load('u2_cnn.onnx','conv_hidden');
  await page.locator('#inputJson').fill(JSON.stringify({image_data:image}));
  const region={bounds:[[0,1],[0,1],[0,1],[0,2]]},
    controlRegion={bounds:[[0,1],[0,1],[1,2],[0,2]]};
  async function run(plugin,config,expectedControl=false,reference=null){
   await page.locator('#interventionMode').selectOption(plugin);
   if(reference)await page.locator('#referenceJson').fill(JSON.stringify(reference));
   await page.locator('#interventionJson').fill(JSON.stringify(config));
   await page.locator('#run').click();
   await page.waitForFunction(id=>SCEPT_UNIVERSAL.getAudit()?.intervention?.plugin===id||
      document.querySelector('#runStatus').dataset.state==='error',plugin,{timeout:75000});
   const audit=await page.evaluate(()=>SCEPT_UNIVERSAL.getAudit());
   assert(audit&&audit.intervention.plugin===plugin,'U5 '+plugin+' failed: '+await page.locator('#runStatus').innerText());
   assert.equal(audit.intervention.schema,'scept.u5.intervention.v1');
   assert.equal(audit.matchedControlOutputs!==null,expectedControl);
   const hidden=audit.outputs.find(x=>x.tensor==='conv_hidden');
   assert(hidden&&hidden.source==='selected_real_intermediate',plugin+' no true intermediate');
   assert(hidden.changed>0&&hidden.mean_abs_delta>0,plugin+' did not alter actual CNN intermediate');
   assert.equal(audit.adapter,'onnx-wasm');
   return audit;
  }
  const noise=await run('u5.noise',{sigma:.2,seed:123,region});
  const noiseReplay=await run('u5.noise',{sigma:.2,seed:123,region});
  assert.deepEqual(noise.outputs[0].first12,noiseReplay.outputs[0].first12,'seeded actual ONNX output must replay');
  await run('u5.occlusion',{region,fill:0});
  await run('u5.replace',{region},false,{image_data:{...image,data:image.data.map(x=>x+100)}});
  const matched=await run('u5.matched-control',{region,controlRegion,fill:0,meanTolerance:4},true);
  assert(matched.matchedControlOutputs.some(x=>x.changed>0),'genuine C forward pass missing');
  assert.equal(matched.intervention.primaryCount,matched.intervention.controlCount);
  // Fail closed on a noninstrumented original model despite a plugin option.
  await page.locator('#interventionMode').selectOption('u5.internal-mask');
  await page.locator('#interventionJson').fill(JSON.stringify({maskInput:'scept_mask',indices:[0],value:0}));
  await page.locator('#run').click();
  await page.waitForFunction(()=>document.querySelector('#runStatus').dataset.state==='error',{timeout:20000});
  assert.match(await page.locator('#runStatus').innerText(),/verified in-graph activation mask/i);
  // Offline instrumented CNN graph: Mul really changes downstream operator outputs.
  await load('u5_cnn_mask.onnx','conv_hidden');
  const mask={type:'float32',dims:[1,1,3,3],data:new Array(9).fill(1)};
  await page.locator('#inputJson').fill(JSON.stringify({image_data:image,scept_mask:mask}));
  const snap=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
  assert(snap.ready&&snap.inputNames.includes('scept_mask'));
  const internal=await run('u5.internal-mask',{maskInput:'scept_mask',indices:[0],value:0});
  assert.equal(internal.intervention.targetTensor,'conv_hidden');
  assert.equal(internal.intervention.weightsModified,false);
  assert(internal.outputs.some(x=>x.tensor==='conv_output'&&x.changed>0),'real output must change downstream of gated hidden tensor');
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-u5-plugin-real-cnn.png',fullPage:true});
  console.log('PASS U5 real Chrome WASM: noise/replay, occlusion, region replacement, A/B/C matched control, actual instrumented internal-mask');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});