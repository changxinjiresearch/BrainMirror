'use strict';
/* U6 real Chromium ONNX/WASM queue: multi-intensity and A/B/C actual tensor outputs. */
const {chromium}=require('playwright'),assert=require('node:assert/strict'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1260,height:940}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_U6_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
  await page.locator('#modelPreset').selectOption('local');
  await page.locator('#modelFile').setInputFiles(path.join(__dirname,'generated','u2_cnn.onnx'));
  await page.locator('#architectureProfile').selectOption('cnn-nchw');
  await page.locator('#inspect').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||
    document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:40000});
  assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).inspected,
   await page.locator('#modelStatus').innerText());
  await page.locator('#probes').fill('conv_hidden');
  await page.locator('#load').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||
    document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:50000});
  assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).ready,
   await page.locator('#modelStatus').innerText());
  const image={type:'float32',dims:[1,1,4,4],data:Array.from({length:16},(_,i)=>i/37)};
  await page.locator('#inputJson').fill(JSON.stringify({image_data:image}));
  const region={bounds:[[0,1],[0,1],[0,1],[0,2]]};
  await page.locator('#u6Mode').selectOption('sweep');
  await page.locator('#u6Plugin').selectOption('u5.noise');
  await page.locator('#u6Config').fill(JSON.stringify({region}));
  await page.locator('#u6Seed').fill('240216');
  await page.locator('#u6Parameter').selectOption('sigma');
  await page.locator('#u6Intensities').fill('[0,0.1,0.2]');
  await page.locator('#u6Create').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().total===3,null,{timeout:20000});
  const queued=await page.evaluate(()=>SCEPT_UNIVERSAL.getU6Snapshot());
  assert.equal(queued.pending,3);
  assert.equal(new Set(queued.tasks.map(t=>t.seed)).size,3);
  assert(queued.inputHashes?.image_data?.sha256);
  await page.locator('#u6Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().completed===3||
    document.querySelector('#u6Status').dataset.state==='error',null,{timeout:80000});
  const sweep=await page.evaluate(()=>SCEPT_UNIVERSAL.getU6Snapshot());
  assert.equal(sweep.completed,3,await page.locator('#u6Status').innerText());
  assert.equal(sweep.failed,0);
  assert.equal(sweep.tasks[0].result.outputs.find(o=>o.tensor==='conv_hidden').changed,0);
  assert(sweep.tasks[1].result.outputs.find(o=>o.tensor==='conv_hidden').changed>0);
  assert(sweep.tasks[2].result.outputs.find(o=>o.tensor==='conv_hidden').changed>0);
  assert(sweep.tasks.every(t=>t.result.genuineInference&&t.result.numberOfForwards===2));
  assert.equal(sweep.tasks[1].result.intervention.seed,sweep.tasks[1].seed);
  assert(!JSON.stringify(sweep).includes('image_data":{"type":"float32","dims":[1,1,4,4],"data"'));
  await page.locator('#u6Create').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().pending===3,null,{timeout:20000});
  await page.locator('#u6Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().completed===3,null,{timeout:80000});
  const replay=await page.evaluate(()=>SCEPT_UNIVERSAL.getU6Snapshot());
  assert.deepEqual(sweep.tasks.map(t=>t.result.outputs.map(o=>o.first12)),
   replay.tasks.map(t=>t.result.outputs.map(o=>o.first12)),'same seed and model must replay numerical outputs');
  await page.locator('#u6Mode').selectOption('batch');
  await page.locator('#u6Plugin').selectOption('u5.matched-control');
  await page.locator('#u6Config').fill(JSON.stringify({region,
   controlRegion:{bounds:[[0,1],[0,1],[1,2],[0,2]]},meanTolerance:1,fill:0}));
  await page.locator('#u6Controls').fill(JSON.stringify([
   {id:'control-compatible',config:{}},
   {id:'control-intentionally-invalid',config:{meanTolerance:0}}
  ]));
  await page.locator('#u6Create').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().pending===2,null,{timeout:25000});
  await page.locator('#u6Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().completed+
   SCEPT_UNIVERSAL.getU6Snapshot().failed===2,null,{timeout:75000});
  const batch=await page.evaluate(()=>SCEPT_UNIVERSAL.getU6Snapshot());
  assert.equal(batch.completed,1);assert.equal(batch.failed,1);
  assert.equal(batch.tasks[0].result.numberOfForwards,3);
  assert(batch.tasks[0].result.matchedControlOutputs.some(o=>o.changed>0));
  assert.equal(batch.tasks[1].error.code,'MATCH_FAILED');
  await page.locator('#u6Retry').click();
  const retried=await page.evaluate(()=>SCEPT_UNIVERSAL.getU6Snapshot());
  assert.equal(retried.pending,1);
  assert.equal(retried.tasks[1].seed,batch.tasks[1].seed);
  await page.locator('#u6Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().failed===1,null,{timeout:75000});
  const again=await page.evaluate(()=>SCEPT_UNIVERSAL.getU6Snapshot());
  assert.equal(again.tasks[1].attempts,2);
  assert.equal(again.tasks[0].attempts,1);
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-u6-queue-real-onnx.png',fullPage:true});
  console.log('PASS U6 real ONNX browser queue: three-level seeded sweep exact replay; A/B/C batch, per-task failure, retry, independent hashes and non-sensitive audit');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
