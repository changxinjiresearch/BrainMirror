/* Actual browser acceptance for the simplified Chinese input-left / network-right UI. */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
const BASE=process.env.SCEPT_STUDIO_URL||'http://127.0.0.1:8765/';
const inspect=()=>({
 width:window.innerWidth,
 scroll:document.documentElement.scrollWidth,
 input:document.querySelector('.input-panel').getBoundingClientRect().toJSON(),
 output:document.querySelector('.output-panel').getBoundingClientRect().toJSON(),
 advanced:document.getElementById('advanced').open
});
async function ready(page){
 await page.waitForFunction(()=>window.SCEPT_STUDIO?.getSnapshot().modelReady||document.getElementById('modelState').dataset.mode==='error',null,{timeout:120000});
 const state=await page.evaluate(()=>window.SCEPT_STUDIO.getSnapshot());
 assert(state.modelReady,'Auto model load failed: '+await page.locator('#notice').innerText());
 assert(state.sourceVerified,'Pretrained source weight integrity was not checked');
}
(async()=>{
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
 const context=await browser.newContext({viewport:{width:1366,height:900},acceptDownloads:true});
 const page=await context.newPage(),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(BASE,{waitUntil:'domcontentloaded'});
 assert.match(await page.locator('h1').innerText(),/AI 模型的内部变化/);
 assert.equal(await page.locator('.action').count(),1,'Only one prominent start action in homepage');
 assert.equal(await page.locator('#advanced').getAttribute('open'),null,'Scientific details must start collapsed');
 await ready(page);
 await page.locator('#roiRadius').fill('30');
 await page.locator('#perturbAmount').fill('0.35');
 const canvas=page.locator('#inputCanvas'),rect=await canvas.boundingBox();
 await page.mouse.click(rect.x+rect.width*.24,rect.y+rect.height*.35);
 await page.locator('#startBtn').click();
 await page.waitForFunction(()=>window.SCEPT_STUDIO?.getSnapshot().hasAudit||document.querySelector('#modelState').dataset.mode==='error',null,{timeout:150000});
 const output=await page.evaluate(()=>window.SCEPT_STUDIO.getSnapshot());
 assert(output.hasAudit,'Real hidden activation experiment did not complete: '+await page.locator('#notice').innerText());
 assert.match(output.changed,/\d+[\s,]*\/ 37,824/);
 assert.equal(await page.locator('#tensorPlot canvas').count(),1,'Real measured delta chart missing');
 const info=await page.evaluate(inspect);
 assert(info.scroll<=1368,'Desktop horizontal overflow after ViT inference: '+JSON.stringify(info));
 assert(info.input.right<=info.output.left+1,'Input and output not side by side at 1366px');
 assert.equal(info.advanced,false,'Advanced drawer must stay collapsed');
 await page.locator('#advanced summary').click();
 const dl=page.waitForEvent('download');await page.locator('#exportJson').click();const d=await dl;
 const stream=await d.createReadStream(),parts=[];for await(const buf of stream)parts.push(buf);
 const audit=JSON.parse(Buffer.concat(parts).toString('utf8'));
 assert.equal(audit.source_model_sha256,'32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3');
 assert.equal(audit.observations['/vit/encoder/layer.6/attention/attention/value/MatMul_output_0'].before.length,37824);
 await page.screenshot({path:'/tmp/scept-studio-chinese-1366.png',fullPage:true});
 for(const w of [390,768,1024,1536]){
   await page.setViewportSize({width:w,height:900});
   const v=await page.evaluate(inspect);
   assert(v.scroll<=w+2,'Horizontal overflow at '+w+': '+JSON.stringify(v));
   assert(v.input.right<=w+2&&v.output.right<=w+2,'Panels extend beyond viewport at '+w+': '+JSON.stringify(v));
 }
 assert.equal(errors.length,0,'Uncaught browser errors '+JSON.stringify(errors));
 console.log('PASS CHINESE STUDIO: automatic pretrained ViT, one visible primary action, real ROI input intervention, hidden tensor chart, JSON audit, no horizontal overflow at 390/768/1024/1366/1536.');
 await context.close();
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
