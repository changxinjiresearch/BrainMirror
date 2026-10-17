/* Chrome CPU test of actual pretrained ViT hidden-layer ONNX output, post-export.
   Input is synthetic numeric data: tests observational mechanics, not medical efficacy. */
const {chromium}=require('playwright'),fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
const model=fs.readFileSync('/tmp/scept-vit-observed.onnx');
const name=fs.readFileSync('/tmp/scept-vit-probe-name.txt','utf8');
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
const page=await browser.newPage({viewport:{width:1300,height:850}});
await page.goto('http://127.0.0.1:8765/lab.html',{waitUntil:'domcontentloaded'});
await page.locator('#modelFile').setInputFiles({name:'pretrained_vit_with_internal_probe.onnx',mimeType:'application/octet-stream',buffer:model});
await page.locator('#loadBtn').click();
await page.waitForFunction(()=>!document.querySelector('#runBtn').disabled || document.querySelector('#notify').textContent.length>0,null,{timeout:120000});
if(await page.locator('#runBtn').isDisabled())throw Error('Instrumented pretrained model import failed: '+await page.locator('#notify').innerText());
assert((await page.locator('#outputDefs').innerText()).includes(name),'Instrumented real intermediate probe not visible');
await page.evaluate(()=>{const a=new Float32Array(3*224*224);for(let i=0;i<a.length;i++)a[i]=((i*73)%256)/255;window.SCEPT_IMPORT_VOLUME_TENSOR({dims:[1,3,224,224],data:a})});
await page.locator('#strength').fill('.15');
await page.locator('#runBtn').click();
await page.waitForFunction(()=>document.querySelector('#probeSelect').options.length>=2 || document.querySelector('#notify').textContent.length>0,null,{timeout:180000});
const msg=(await page.locator('#notify').innerText()).trim();if(msg)throw Error('Real pretrained ViT probe inference failed: '+msg);
await page.locator('#probeSelect').selectOption(name);
const result=(await page.locator('#changed').innerText()).trim();
const match=result.match(/^(\d+)\/(\d+)$/);
assert(match && Number(match[2])>=128,'Real internal tensor did not materialize');
assert(Number(match[1])>0,'Perturbation did not change measured internal ViT output');
assert.equal((await page.locator('#resultType').innerText()).includes('REAL COMPUTATION'),true);
console.log('PASS: verified pretrained ViT real intermediate tensor observation '+name+' · '+result+' changed');
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
