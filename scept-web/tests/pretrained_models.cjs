/* Genuine pretrained ONNX models from published model repositories.
   Network is used only to DOWNLOAD PUBLIC MODEL files into the CI runner.
   User images/weights are NEVER uploaded. The inputs are synthetic numeric
   smoke tests and are not semantically meaningful images. */
const {chromium}=require('playwright');
const {createHash}=require('node:crypto');
const assert=require('node:assert/strict');
const fs=require('node:fs');

const sources=[
 {name:'ViT tiny float32 (pretrained transformer)',url:'https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/resolve/ebffd7b9c0c53b51bada44116abd8f01aed41be6/onnx/model.onnx',sha:'32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3'},
 {name:'SqueezeNet (pretrained CNN)',url:'https://huggingface.co/onnxmodelzoo/squeezenet1.1-7/resolve/main/squeezenet1.1-7.onnx',sha:'1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088'}
];
async function testModel(page,model){
 const res=await fetch(model.url,{signal:AbortSignal.timeout(90000)});
 if(!res.ok)throw Error(model.name+' download HTTP '+res.status);
 const buffer=Buffer.from(await res.arrayBuffer());
 const digest=createHash('sha256').update(buffer).digest('hex');
 assert.equal(digest,model.sha,model.name+' model integrity');
 if(model.name.includes('ViT tiny'))fs.writeFileSync('/tmp/scept-vit-original.onnx',buffer);
 await page.locator('#modelFile').setInputFiles({name:'pretrained.onnx',mimeType:'application/octet-stream',buffer});
 await page.locator('#loadBtn').click();
 await page.waitForFunction(()=>!document.getElementById('runBtn').disabled || document.getElementById('notify').textContent.length>0,null,{timeout:120000});
 if(await page.locator('#runBtn').isDisabled())throw Error(model.name+' load failed '+await page.locator('#notify').innerText());
 const inputs=await page.locator('#inputDefs').innerText();
 await page.evaluate(()=>{
   const d=new Float32Array(1*3*224*224);
   for(let i=0;i<d.length;i++)d[i]=((i*73)%256)/255;
   window.SCEPT_IMPORT_VOLUME_TENSOR({dims:[1,3,224,224],data:d});
 });
 await page.locator('#offset').fill('0');await page.locator('#count').fill('1');await page.locator('#strength').fill('.1');
 await page.locator('#runBtn').click();
 await page.waitForFunction(()=>document.querySelector('#resultType').textContent.includes('REAL COMPUTATION') || document.getElementById('notify').textContent.length>0,null,{timeout:180000});
 if((await page.locator('#notify').innerText()).trim())throw Error(model.name+' run failed '+await page.locator('#notify').innerText());
 const state=await page.locator('#changed').innerText();
 assert.match(state,/\d+\/\d+/);
 console.log('PASS: '+model.name+' · SHA '+digest.slice(0,12)+' · '+inputs+' · '+state+' changed outputs');
}
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
 const page=await browser.newPage({viewport:{width:1366,height:900}});
 await page.goto('http://127.0.0.1:8765',{waitUntil:'domcontentloaded'});
 for(const model of sources)await testModel(page,model);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
