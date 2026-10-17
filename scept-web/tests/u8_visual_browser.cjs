'use strict';
/* U8 real ONNX Chrome: operator graph, CNN axes, three orthogonal input planes,
   linked U5 -> U6 -> U7, no simulated activations. */
const {chromium}=require('playwright'),assert=require('node:assert/strict'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1450,height:1020}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_U8_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
  const snap=()=>page.evaluate(()=>SCEPT_UNIVERSAL.getU8Snapshot());
  async function load(filename,profile,probe){
   await page.locator('#modelPreset').selectOption('local');
   await page.locator('#modelFile').setInputFiles(path.join(__dirname,'generated',filename));
   await page.locator('#architectureProfile').selectOption(profile);
   await page.locator('#inspect').click();
   await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||
     document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:50000});
   assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).inspected,
    await page.locator('#modelStatus').innerText());
   await page.locator('#probes').fill(probe);
   await page.locator('#load').click();
   await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||
     document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:50000});
   assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).ready,
    await page.locator('#modelStatus').innerText());
   const neutral=await snap();
   assert.equal(neutral.hasScene,false);
   assert(neutral.graph.nodes.length>0,'Graph should draw real nodes before evidence');
   assert(neutral.graph.nodes.every(n=>n.observed===null),'No fabricated activation on load');
  }
  await load('u2_cnn.onnx','cnn-nchw','conv_hidden');
  await page.locator('#u8InputLayout').selectOption('NCHW');
  await page.locator('#u8InputName').fill('image_data');
  const image={type:'float32',dims:[1,1,4,4],data:Array.from({length:16},(_,i)=>i/37)};
  await page.locator('#inputJson').fill(JSON.stringify({image_data:image}));
  await page.locator('#interventionMode').selectOption('u5.occlusion');
  const region={bounds:[[0,1],[0,1],[0,1],[0,2]]};
  await page.locator('#interventionJson').fill(JSON.stringify({region,fill:0}));
  await page.locator('#run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU8Snapshot().hasScene||
   document.querySelector('#runStatus').dataset.state==='error',null,{timeout:75000});
  let view=await snap();
  assert(view.hasScene,await page.locator('#runStatus').innerText());
  assert.equal(view.source,'u5');
  assert(view.views.some(v=>v.tensor==='conv_hidden'&&v.role==='channel'));
  assert(view.graph.nodes.some(n=>n.observed?.tensor==='conv_hidden'));
  assert(view.renderedNodes.some(n=>n.measured&&n.delta>0));
  assert(view.renderedNodes.some(n=>!n.measured));
  assert.equal(view.hasModelInput,true,'real paired image input must be passed to U8');
  assert(view.modelInput.changed>0);
  assert.equal(view.modelInput.layout,'NCHW');
  assert.equal(view.modelInput.anatomicalStatus,'UNVERIFIED');
  assert.match(await page.locator('#u8GraphInfo').innerText(),/真实/);
  assert.match(await page.locator('#u8VolumeInfo').innerText(),/2D|model-input-image/);
  await page.locator('#u8Mode').selectOption('delta');
  assert.equal((await snap()).mode,'delta');
  await page.locator('#u8Mode').selectOption('compare');
  // U6 genuine batch/sweep flows automatically update the SAME U8 panels.
  await page.locator('#u6Mode').selectOption('sweep');
  await page.locator('#u6Plugin').selectOption('u5.noise');
  await page.locator('#u6Config').fill(JSON.stringify({region}));
  await page.locator('#u6Intensities').fill('[0,0.1,0.2]');
  await page.locator('#u6Create').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().pending===3,null,{timeout:25000});
  await page.locator('#u6Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().completed===3||
   document.querySelector('#u6Status').dataset.state==='error',null,{timeout:80000});
  view=await snap();assert.equal(view.source,'u6');
  assert.equal(view.hasScene,true);assert.equal(view.views[0].evidence,'actual-onnx-runtime-graph-outputs');
  await page.locator('#u8Task').selectOption('T002');
  view=await snap();assert.equal(view.taskId,'T002');assert.equal(view.hasModelInput,true);
  assert(view.views.some(v=>v.summary.delta>0));
  // U7 external plugin JSON controls the same U8 canvas, without main code edits.
  await page.locator('#u7ManifestFile').setInputFiles(path.join(__dirname,'..','plugins','m2_engineering_cnn.json'));
  await page.waitForFunction(()=>document.querySelector('#u7Manifest').value.includes('m2-engineering-cnn-demo'),null,{timeout:15000});
  await page.locator('#u7Register').click();await page.locator('#u7Prepare').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().pending===3||
   document.querySelector('#u7Status').dataset.state==='error',null,{timeout:25000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Snapshot())).pending,3,
   await page.locator('#u7Status').innerText());
  await page.locator('#u7Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().completed===3||
   document.querySelector('#u7Status').dataset.state==='error',null,{timeout:80000});
  view=await snap();assert.equal(view.source,'u7');assert.equal(view.hasScene,true);
  assert.equal(view.views[0].evidence,'actual-onnx-runtime-graph-outputs');
  assert(view.modelInput&&view.modelInput.source==='actual-model-feed-u7');
  // A genuinely different ONNX architecture and rank-5 model input. Not medical.
  await load('u8_3d_cnn.onnx','cnn-volume-ncdhw','vol_hidden');
  await page.locator('#u8InputLayout').selectOption('NCDHW');
  await page.locator('#u8InputName').fill('vol_input');
  const volume={type:'float32',dims:[1,1,3,4,4],data:Array.from({length:48},(_,i)=>(i+1)/37)};
  await page.locator('#inputJson').fill(JSON.stringify({vol_input:volume}));
  await page.locator('#interventionMode').selectOption('u5.occlusion');
  await page.locator('#interventionJson').fill(JSON.stringify({
   region:{bounds:[[0,1],[0,1],[0,1],[0,2],[0,2]]},fill:0}));
  await page.locator('#run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU8Snapshot().hasScene||
   document.querySelector('#runStatus').dataset.state==='error',null,{timeout:70000});
  view=await snap();assert(view.hasScene,await page.locator('#runStatus').innerText());
  assert.equal(view.source,'u5');
  assert.equal(view.modelInput.layout,'NCDHW');
  assert.deepEqual(view.modelInput.sampled,[3,4,4]);
  assert(view.modelInput.changed>0);
  assert.equal(view.modelInput.anatomicalStatus,'UNVERIFIED');
  assert(view.views.some(v=>v.tensor==='vol_hidden'&&v.shape.length===5&&v.role==='channel'));
  assert(view.graph.nodes.some(n=>n.observed?.tensor==='vol_hidden'));
  for(const orientation of ['axial','coronal','sagittal']){
   await page.locator('#u8SliceAxis').selectOption(orientation);
   assert.match(await page.locator('#u8VolumeInfo').innerText(),new RegExp(orientation));
  }
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-u8-unified-real-onnx.png',fullPage:true});
  console.log('PASS U8 real ONNX Chrome: measured graph and dynamic CNN units, genuine A/B/Delta 2D input, U5/U6/U7 automatic scene linkage and rank-5 volume three orthogonal planes');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
