/* Run with node after installing playwright and its Chromium browser.
   No patient data, no external model weights; tests use engineering-only ONNX fixtures. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const base='http://127.0.0.1:8765';
function syntheticMRI(){
 const out=Buffer.alloc(352+8*8*8*4);
 out.writeInt32LE(348,0);out.writeInt16LE(3,40);
 [8,8,8].forEach((v,i)=>out.writeInt16LE(v,42+i*2));
 out.writeInt16LE(16,70);out.writeInt16LE(32,72);
 [1,1,1].forEach((v,i)=>out.writeFloatLE(v,80+i*4));
 out.writeFloatLE(352,108);out.writeFloatLE(1,112);
 out.writeInt16LE(1,254);
 for(let i=0;i<3;i++)out.writeFloatLE(1,280+i*20);
 out.write('n+1',344,'ascii');
 for(let i=0;i<512;i++)out.writeFloatLE(i%17,352+4*i);
 return out;
}
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1360,height:900},acceptDownloads:true});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('requestfailed',r=>errors.push('REQUEST '+r.url()+' '+r.failure()?.errorText));
  page.on('console',m=>{if(m.type()==='error')errors.push('CONSOLE '+m.text())});
  await page.goto(base,{waitUntil:'domcontentloaded'});
  await page.locator('#demoLinear').click();
  await page.waitForFunction(()=>!document.querySelector('#runBtn').disabled || document.querySelector('#notify').textContent.length>0,null,{timeout:60000});
  if(await page.locator('#runBtn').isDisabled())throw Error('Baseline model load failed: '+await page.locator('#notify').innerText()+' | '+JSON.stringify(errors));
  await page.locator('#runBtn').click();
  await page.waitForFunction(()=>document.querySelector('#changed').textContent.includes('/3'),null,{timeout:60000});
  assert.notEqual(await page.locator('#mae').innerText(),'—');
  assert.equal(await page.locator('.barrow').count(),3);
  let downloadPromise=page.waitForEvent('download');
  await page.locator('#exportBtn').click();
  const download=await downloadPromise;assert(download.suggestedFilename().endsWith('.json'));
  await page.locator('#demoAblation').click();
  await page.waitForFunction(()=>document.querySelector('#inputDefs').textContent.includes('scept_mask') || document.querySelector('#notify').textContent.length>0,null,{timeout:60000});
  if(await page.locator('#runBtn').isDisabled())throw Error('Ablation model load failed: '+await page.locator('#notify').innerText()+' | '+JSON.stringify(errors));
  await page.locator('#runBtn').click();
  await page.waitForFunction(()=>document.querySelector('#resultType').textContent.includes('INTERNAL ABLATION'),null,{timeout:60000});
  const changed=(await page.locator('#changed').textContent()).trim();assert(changed.includes('/'));
  await page.locator('#mriUpload').setInputFiles({name:'test.nii',mimeType:'application/octet-stream',buffer:syntheticMRI()});
  await page.waitForFunction(()=>document.querySelector('#mriInfo').textContent.includes('8×8×8'),null,{timeout:30000});
  await page.locator('#mriApply').click();
  await page.waitForFunction(()=>document.querySelector('#mriInfo').textContent.includes('QC:'),null,{timeout:30000});
  await page.locator('#mriToModel').click();
  await page.waitForFunction(()=>document.querySelector('#inputReport').textContent.includes('MRI-derived'),null,{timeout:30000});
  assert.equal(errors.length,0,'page exceptions: '+JSON.stringify(errors));
  await page.screenshot({path:'/tmp/scept-browser-qa.png',fullPage:true});
  const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true});
  await mobile.goto(base,{waitUntil:'domcontentloaded'});
  assert.equal(await mobile.locator('.brand').count(),1);
  console.log('PASS: real ONNX baseline/perturbation, masked internal ablation, NIfTI load/edit, JSON export, mobile layout, no uncaught JS errors');
 }finally{await browser.close()}
})().catch(err=>{console.error(err);process.exit(1)});
