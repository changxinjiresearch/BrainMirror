/* One-click real pretrained Transformer test with ZERO user file uploads.
   Loads pinned ViT remotely in real Chrome; patches ONNX protobuf output to
   expose true layer-6 Attention value tensor without training. */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
   const page=await browser.newPage({viewport:{width:1366,height:900},acceptDownloads:true});
   const errors=[];
   page.on('pageerror',e=>errors.push('JS '+e.message));
   await page.goto(process.env.SCEPT_TEST_URL || 'http://127.0.0.1:8765',{waitUntil:'domcontentloaded'});
   await page.locator('#transformerDemo').click();
   await page.waitForFunction(()=>{
      const text=document.querySelector('#transformerStatus')?.textContent||'';
      return text.startsWith('PASS ·')||text.startsWith('Cannot complete');
   },null,{timeout:180000});
   const status=await page.locator('#transformerStatus').innerText();
   if(!status.startsWith('PASS ·'))throw Error('One-click ViT test failed: '+status+'; '+JSON.stringify(errors));
   assert.match(status,/layer-6 hidden tensor/);
   assert.match(await page.locator('#outputDefs').innerText(),/layer\.6\/attention/);
   assert.match(await page.locator('#resultType').innerText(),/PRETRAINED TRANSFORMER/);
   const delta=(await page.locator('#changed').innerText()).trim();
   const match=delta.match(/^(\d+)\/(\d+)$/);
   assert(match && Number(match[2])===37824 && Number(match[1])>0,delta);
   const downloadPromise=page.waitForEvent('download');
   await page.locator('#exportBtn').click();
   const downloaded=await downloadPromise;
   const stream=await downloaded.createReadStream();const chunks=[];for await(const chunk of stream)chunks.push(chunk);
   const audit=JSON.parse(Buffer.concat(chunks).toString('utf8'));
   assert.equal(audit.source_pretrained_model_sha256,'32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3');
   assert.equal(audit.model_classification,'VERIFIED_PRETRAINED_TRANSFORMER');
   const layer='/vit/encoder/layer.6/attention/attention/value/MatMul_output_0';
   assert.equal(audit.observations[layer].before.length,37824);
   assert.equal(errors.length,0,JSON.stringify(errors));
   console.log('PASS AUTO IMPORT/REAL VI-T: verified source weights, real WASM inference, 37824 hidden-layer activations, audit exported; changed '+delta);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
