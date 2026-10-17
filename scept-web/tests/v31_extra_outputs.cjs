const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
 const page=await browser.newPage({viewport:{width:1366,height:900}}),errs=[];
 page.on('pageerror',e=>errs.push(e.message));
 await page.goto(process.env.SCEPT_V31_URL||'http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>SCEPT_V3?.getSnapshot().ready||document.getElementById('readyLabel').dataset.state==='error',null,{timeout:140000});
 assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,'Pretrained MRI inference not ready');
 await page.locator('#runExperiment').click();
 await page.waitForFunction(()=>SCEPT_V3.getSnapshot().hasExperiment||document.getElementById('readyLabel').dataset.state==='error',null,{timeout:160000});
 const audit=await page.evaluate(()=>SCEPT_V3.getAudit());
 const extra=audit.extraObservations;
 assert(extra.length>=3,'Expected additional real Q/K/embedding/residual intermediates, found '+JSON.stringify(extra.map(x=>x.kind)));
 for(const p of extra){
  assert(p.before.length>200&&p.before.length===p.after.length);
  assert(p.shape.reduce((a,b)=>a*b,1)===p.before.length,'Shape not actual');
  let sum=0,max=0;for(let i=0;i<p.before.length;i++){const d=Math.abs(p.after[i]-p.before[i]);sum+=d;max=Math.max(max,d)}
  assert(Math.abs(sum/p.before.length-p.summary.mean)<1e-6,'Synthetic or miscomputed '+p.kind+' difference');
  assert(Math.abs(max-p.summary.max)<1e-6,'Incorrect max difference');
  assert.match(p.probe,/^\/vit\//);
 }
 assert.equal(audit.layers.length,5,'Existing 5 Value probes must remain unchanged');
 assert.equal((await page.locator('.extra-probe').count()),extra.length);
 const kinds=extra.map(x=>x.kind);
 assert(kinds.includes('query')&&kinds.includes('key'),'True Q and K projection probes missing');
 const changed=extra.filter(x=>x.summary.changed>0).length;
 assert(changed>0,'Additional internal probes exhibit no genuine numerical change');
 assert.equal(errs.length,0,'Uncaught browser errors: '+errs.join('; '));
 console.log('PASS SCEPT v3.1 expanded actual intermediates:',JSON.stringify(extra.map(x=>({kind:x.kind,shape:x.shape,delta:x.summary.mean}))));
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
