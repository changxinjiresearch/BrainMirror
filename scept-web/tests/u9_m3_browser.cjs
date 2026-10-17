'use strict';
/* M3 real Chrome / ONNX runtime acceptance, no server and no fabricated result.
 * Exports are genuine downloaded files, re-imported and rerun from original data.
 */
const {chromium}=require('playwright'),fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1380,height:940}}),errors=[];
  page.on('pageerror',e=>errors.push(e.stack||e.message));
  await page.goto(process.env.SCEPT_U9_URL||'http://127.0.0.1:8765/universal.html',{waitUntil:'domcontentloaded'});
  await page.locator('#modelPreset').selectOption('local');
  await page.locator('#modelFile').setInputFiles(path.join(__dirname,'generated','u2_cnn.onnx'));
  await page.locator('#architectureProfile').selectOption('cnn-nchw');
  await page.locator('#inspect').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().inspected||
    document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:50000});
  assert((await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot())).inspected,
   await page.locator('#modelStatus').innerText());
  await page.locator('#probes').fill('conv_hidden');
  await page.locator('#load').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getSnapshot().ready||
    document.querySelector('#modelStatus').dataset.state==='error',null,{timeout:50000});
  const snap=await page.evaluate(()=>SCEPT_UNIVERSAL.getSnapshot());
  assert(snap.ready,await page.locator('#modelStatus').innerText());
  const image={type:'float32',dims:[1,1,4,4],data:Array.from({length:16},(_,i)=>(i+1)/37)};
  await page.locator('#inputJson').fill(JSON.stringify({image_data:image}));
  // SDK scaffolding and capability / actual model fingerprint.
  await page.locator('#u9GenerateTemplate').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.status==='generated',null,{timeout:15000});
  let template=JSON.parse(await page.locator('#u7Manifest').inputValue());
  assert.equal(template.model.sha256,snap.hash);
  assert.equal(template.target.tensor,'conv_output');
  await page.locator('#u7ManifestFile').setInputFiles(path.join(__dirname,'..','plugins','m2_engineering_cnn.json'));
  await page.waitForFunction(()=>document.querySelector('#u7Manifest').value.includes('m2-engineering-cnn-demo'),null,{timeout:15000});
  await page.locator('#u9Compatibility').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.schema==='scept.u9.compatibility.v1',
    null,{timeout:15000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport)).compatible,true);
  await page.locator('#u9Benchmark').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.schema==='scept.u9.performance.v1',
    null,{timeout:30000});
  const perf=await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport);
  assert.equal(perf.status,'measured');
  assert.equal(perf.timesMs.length,3);
  assert(perf.meanMs>=0);
  // U6 real ONNX sweep -> U9 content-addressed archive -> file import -> input hashing -> rerun.
  const region={bounds:[[0,1],[0,1],[0,1],[0,2]]};
  await page.locator('#u6Mode').selectOption('sweep');
  await page.locator('#u6Plugin').selectOption('u5.noise');
  await page.locator('#u6Config').fill(JSON.stringify({region}));
  await page.locator('#u6Parameter').selectOption('sigma');
  await page.locator('#u6Intensities').fill('[0,0.1,0.2]');
  await page.locator('#u6Create').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().pending===3,null,{timeout:25000});
  await page.locator('#u6Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU6Snapshot().completed===3||
   document.querySelector('#u6Status').dataset.state==='error',null,{timeout:80000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU6Snapshot())).completed,3,
   await page.locator('#u6Status').innerText());
  const d1=page.waitForEvent('download');
  await page.locator('#u9ExportU6').click();
  const download1=await d1,bytes1=await fs.readFile(await download1.path(),'utf8');
  const bundle1=JSON.parse(bytes1);
  assert.equal(bundle1.schema,'scept.u9.audit.v1');
  assert.equal(bundle1.kind,'u6');
  assert(bundle1.contentSha256);
  assert.equal(bundle1.record.completed,3);
  assert(!bytes1.includes('"data":['),'No original input arrays exported');
  await page.locator('#u9BundleFile').setInputFiles({name:'u6-original.json',mimeType:'application/json',buffer:Buffer.from(bytes1)});
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().imported,null,{timeout:15000});
  await page.locator('#u9Verify').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.verified===true||
   document.querySelector('#u9Status').dataset.state==='error',null,{timeout:25000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport)).verified,true,
   await page.locator('#u9Status').innerText());
  await page.locator('#u9Replay').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.status==='matched'||
   document.querySelector('#u9Status').dataset.state==='error',null,{timeout:75000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport)).status,'matched',
   await page.locator('#u9Status').innerText());
  // Tampered JSON SHA detects changes before any new model forward.
  const modified=JSON.parse(bytes1);modified.record.tasks[0].status='failed';
  await page.locator('#u9BundleFile').setInputFiles({name:'tampered.json',mimeType:'application/json',
   buffer:Buffer.from(JSON.stringify(modified))});
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().imported,null,{timeout:15000});
  await page.locator('#u9Verify').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.code==='AUDIT_HASH_MISMATCH',
    null,{timeout:25000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport)).code,'AUDIT_HASH_MISMATCH');
  // Wrong real typed inputs must fail the same archive's verified replay.
  await page.locator('#u9BundleFile').setInputFiles({name:'restored.json',mimeType:'application/json',buffer:Buffer.from(bytes1)});
  const changed={...image,data:image.data.slice()};changed.data[0]=100;
  await page.locator('#inputJson').fill(JSON.stringify({image_data:changed}));
  await page.locator('#u9Verify').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.code==='INPUT_HASH_MISMATCH',
   null,{timeout:25000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport)).code,'INPUT_HASH_MISMATCH');
  await page.locator('#inputJson').fill(JSON.stringify({image_data:image}));
  // U7 independent method/evaluation manifest in the SAME real browser.
  await page.locator('#u7Register').click();
  await page.locator('#u7Prepare').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().pending===3||
   document.querySelector('#u7Status').dataset.state==='error',null,{timeout:30000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU7Snapshot())).pending,3,
   await page.locator('#u7Status').innerText());
  await page.locator('#u7Run').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU7Snapshot().completed===3||
   document.querySelector('#u7Status').dataset.state==='error',null,{timeout:80000});
  const d2=page.waitForEvent('download');
  await page.locator('#u9ExportU7').click();
  const download2=await d2,bytes2=await fs.readFile(await download2.path(),'utf8');
  const bundle2=JSON.parse(bytes2);
  assert.equal(bundle2.kind,'u7');
  assert(!bytes2.includes('"data":['),'No original MRI/input tensors in U7 audit');
  await page.locator('#u9BundleFile').setInputFiles({name:'u7-original.json',mimeType:'application/json',
   buffer:Buffer.from(bytes2)});
  await page.locator('#u9Verify').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.verified===true||
   document.querySelector('#u9Status').dataset.state==='error',null,{timeout:25000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport)).verified,true);
  await page.locator('#u9Replay').click();
  await page.waitForFunction(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport?.status==='matched'||
   document.querySelector('#u9Status').dataset.state==='error',null,{timeout:80000});
  assert.equal((await page.evaluate(()=>SCEPT_UNIVERSAL.getU9Snapshot().lastReport)).status,'matched',
   await page.locator('#u9Status').innerText());
  assert.equal(errors.length,0,errors.join(' | '));
  await page.screenshot({path:'/tmp/scept-u9-m3-real-browser.png',fullPage:true});
  console.log('PASS M3 Chrome real ONNX: new SDK plugin manifest, capability preflight, measured latency, U6 and U7 signed-content-free SHA256 audit download, tamper/inputs denial, genuine new forward replay matching');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
