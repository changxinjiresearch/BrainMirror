/* Regression: ONNX metadata and CSS grid must not push the right
   side of the interface beyond desktop Chromebook or mobile viewports. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const base=process.env.SCEPT_TEST_URL || 'http://127.0.0.1:8765';
const sizes=[390,768,1024,1280,1366,1536];
const js=()=>({
  innerWidth:window.innerWidth,
  bodyWidth:document.body.scrollWidth,
  htmlWidth:document.documentElement.scrollWidth,
  main:document.querySelector('main').getBoundingClientRect().toJSON(),
  cards:[...document.querySelectorAll('.modelgrid>.card')].map(e=>e.getBoundingClientRect().toJSON()),
  metadata:document.querySelector('.metadata')?.getBoundingClientRect().toJSON()
});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  for(const width of sizes){
   const page=await browser.newPage({viewport:{width,height:900}});
   await page.goto(base,{waitUntil:'domcontentloaded'});
   const before=await page.evaluate(js);
   assert(before.htmlWidth<=width+2,'horizontal overflow before loading at '+width+': '+JSON.stringify(before));
   assert(before.bodyWidth<=width+2,'body overflow before loading at '+width+': '+JSON.stringify(before));
   for(const c of before.cards)assert(c.right<=width+2,'model card exceeds screen width at '+width+': '+JSON.stringify(c));
   await page.locator('#demoLinear').click();
   await page.waitForFunction(()=>!document.querySelector('#runBtn').disabled || document.querySelector('#notify').textContent.length>0,null,{timeout:60000});
   assert(!(await page.locator('#runBtn').isDisabled()),'failed to initialize ONNX fixture at '+width+': '+await page.locator('#notify').innerText());
   const after=await page.evaluate(js);
   assert(after.htmlWidth<=width+2,'horizontal overflow after ONNX metadata at '+width+': '+JSON.stringify(after));
   assert(after.bodyWidth<=width+2,'body overflow after ONNX metadata at '+width+': '+JSON.stringify(after));
   for(const c of after.cards)assert(c.right<=width+2,'model card cut off at '+width+': '+JSON.stringify(c));
   if(width===1366)await page.screenshot({path:'/tmp/scept-layout-1366.png',fullPage:true});
   console.log('PASS layout '+width+'px: no horizontal scroll, model cards contained in viewport');
   await page.close();
  }
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
