import { chromium } from 'playwright';
import { strict as assert } from 'node:assert';

const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
const page = await browser.newPage({viewport:{width:1440,height:960}});
const jsErrors=[];
page.on('pageerror', err=>jsErrors.push(err.message));
page.on('console', msg=>{if(msg.type()==='error') console.log('Browser console:',msg.text())});
const base='http://127.0.0.1:8999/';
try{
  await page.goto(base,{waitUntil:'domcontentloaded'});
  assert.equal(await page.title(),'SCEPT Studio · BrainMirror');
  assert.equal(await page.locator('#runExperiment').isDisabled(),true);
  await page.locator('#demoMnist').click();
  await page.waitForFunction(()=>document.querySelector('#stateLabel')?.textContent?.includes('已加载预训练模型') || document.querySelector('#runMessage')?.textContent?.includes('模型加载失败'),undefined,{timeout:60000});
  if(!(await page.locator('#stateLabel').textContent()).includes('已加载预训练模型'))throw Error('Model load failure: '+await page.locator('#runMessage').textContent());
  assert.equal(await page.locator('#runExperiment').isEnabled(),true);
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>document.querySelector('#runMessage')?.textContent?.includes('完成两次真实前向传播')||document.querySelector('#runMessage')?.textContent?.includes('实验失败'),undefined,{timeout:60000});
  if((await page.locator('#runMessage').textContent()).includes('实验失败'))throw Error('Inference failure: '+await page.locator('#runMessage').textContent());
  assert.match(await page.locator('#evidencePill').textContent(),/L1/);
  assert.notEqual(await page.locator('#statCount').textContent(),'—');
  assert.equal(await page.locator('#downloadJson').isEnabled(),true);
  await page.locator('#selfTest').click();
  await page.waitForFunction(()=>document.querySelector('#runMessage')?.textContent?.includes('恒等对照'),undefined,{timeout:120000});
  assert.match(await page.locator('#runMessage').textContent(),/PASS/);
  if(jsErrors.length)throw Error('Uncaught JS errors: '+jsErrors.join(' | '));
  console.log('PASS: real pretrained MNIST-8 browser ONNX inference, L1 trace, paired run, identity control');
  await page.locator('#demoSqueeze').click();
  await page.waitForFunction(()=>document.querySelector('#stateLabel')?.textContent?.includes('已加载预训练模型')||document.querySelector('#runMessage')?.textContent?.includes('模型加载失败'),undefined,{timeout:60000});
  if((await page.locator('#runMessage').textContent()).includes('模型加载失败'))throw Error('SqueezeNet load failure: '+await page.locator('#runMessage').textContent());
  assert.match(await page.locator('#modelName').textContent(),/SqueezeNet/);
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>document.querySelector('#runMessage')?.textContent?.includes('完成两次真实前向传播')||document.querySelector('#runMessage')?.textContent?.includes('实验失败'),undefined,{timeout:60000});
  if((await page.locator('#runMessage').textContent()).includes('实验失败'))throw Error('SqueezeNet inference failure: '+await page.locator('#runMessage').textContent());
  console.log('PASS: real pretrained SqueezeNet ONNX browser paired inference');
  await page.locator('#demoVit').click();
  await page.waitForFunction(()=>document.querySelector('#modelName')?.textContent?.includes('ViT-base')||document.querySelector('#runMessage')?.textContent?.includes('模型加载失败'),undefined,{timeout:180000});
  if((await page.locator('#runMessage').textContent()).includes('模型加载失败'))throw Error('Vision Transformer model failed: '+await page.locator('#runMessage').textContent());
  assert.match(await page.locator('#modelName').textContent(),/ViT-base/);
  await page.locator('#runExperiment').click();
  await page.waitForFunction(()=>document.querySelector('#runMessage')?.textContent?.includes('完成两次真实前向传播')||document.querySelector('#runMessage')?.textContent?.includes('实验失败'),undefined,{timeout:180000});
  if((await page.locator('#runMessage').textContent()).includes('实验失败'))throw Error('ViT inference failure: '+await page.locator('#runMessage').textContent());
  console.log('PASS: pretrained Vision Transformer in-browser inference and A/B trace');
}catch(e){
  console.log('E2E FAILURE:',e?.stack||e);
  try{await page.screenshot({path:'scept-e2e-failure.png',fullPage:true})}catch{}
  throw e;
}finally{
  await browser.close();
}
