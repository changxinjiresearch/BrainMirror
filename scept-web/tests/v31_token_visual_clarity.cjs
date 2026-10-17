const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.stack||e.message));
 await page.goto(process.env.SCEPT_V31_VISUAL_URL||'http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().ready||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:150000});
 assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,'Not ready: '+await page.locator('#warning').innerText());
 await page.locator('#runExperiment').click();
 await page.waitForFunction(()=>SCEPT_V3.getSnapshot().hasExperiment||document.querySelector('#readyLabel')?.dataset.state==='error',null,{timeout:170000});
 await page.locator('#viewTokens').click();
 await page.locator('#token3dCanvas').scrollIntoViewIfNeeded();
 const s0=await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot());
 assert.equal(s0.scope,'layer','Readable 197-token single layer should be default after entering the Token view');
 assert.equal(s0.count,197);
 assert(Number.isFinite(s0.colorLo)&&Number.isFinite(s0.colorHi)&&s0.colorHi>s0.colorLo,'A/B legend must have real nonzero quantitative span');
 assert.equal(s0.renderedSamples.length,985);
 assert(s0.renderedSamples.every(p=>p.baseRadius<=3.31),'Token base radius excessively large');
 const uniqueA=new Set(s0.renderedSamples.map(p=>p.colorA)).size;
 const uniqueB=new Set(s0.renderedSamples.map(p=>p.colorB)).size;
 assert(uniqueA>12&&uniqueB>12,'A and B display must preserve observable token-to-token variation: '+uniqueA+'/'+uniqueB);
 assert(s0.renderedSamples.some(p=>p.markedB),'B should show some truthful measured difference indicators');
 const snap=async(name)=>{
  const image=await page.evaluate(()=>{
   const el=document.getElementById('token3dCanvas'),g=el.getContext('2d');
   const {width,height}=el,data=g.getImageData(0,0,width,height);
   let colored=0,bright=0,overdraw=0;const tones=new Set();
   for(let i=0;i<data.data.length;i+=4){
    const r=data.data[i],gg=data.data[i+1],b=data.data[i+2];
    if(r+gg+b>255&&gg>r*.9&&b>r*.7){colored++;if(gg>170)bright++;tones.add([r>>4,gg>>4,b>>4].join(':'));}
    if(gg>130&&r>50&&b>45)overdraw++;
   }
   return{width,height,coloredFraction:colored/(width*height),brightFraction:bright/(width*height),overdrawFraction:overdraw/(width*height),distinctQuantizedTones:tones.size};
  });
  await page.locator('#token3dCanvas').screenshot({path:'/tmp/scept-v31-'+name+'.png'});
  console.log('VISUAL_METRICS',name,JSON.stringify(image));
  assert(image.coloredFraction<.19,'The Token viewer still looks like a solid mass in '+name+': '+JSON.stringify(image));
  assert(image.distinctQuantizedTones>8,'Token visualization has no useful true data color variation in '+name);
  return image;
 };
 await page.locator('#responseCompare').click();
 const a=await snap('compare');
 await page.locator('#responseBefore').click();
 const b=await snap('before');
 await page.locator('#responseAfter').click();
 const c=await snap('after');
 await page.locator('#responseDelta').click();
 const d=await snap('delta');
 assert.equal((await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot())).mode,'delta');
 assert(b.coloredFraction>.003&&c.coloredFraction>.003,'A/B should draw visible isolated discs not blank frames');
 await page.locator('#responseCompare').click();
 const first=await page.evaluate(()=>({s:SCEPT_ACTIVITY3D.getSnapshot(),run:SCEPT_V3.getSnapshot().runId}));
 await page.locator('#mriAmount').fill('0.55');
 await page.locator('#runExperiment').click();
 await page.waitForFunction(id=>SCEPT_V3.getSnapshot().runId!==id||document.querySelector('#readyLabel')?.dataset.state==='error',first.run,{timeout:170000});
 const second=await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot());
 assert.equal(second.colorLo,first.s.colorLo,'Shared baseline/edited color domain should stay fixed on rerun');
 assert.equal(second.colorHi,first.s.colorHi,'Shared baseline/edited color range must not silently rescale');
 assert.equal(second.count,197);
 assert(second.renderedSamples.some((p,i)=>Math.abs(p.rms-first.s.renderedSamples[i].rms)>1e-6),'Visuals not driven by rerun');
 await page.locator('#mriAmount').fill('0');
 const old=await page.evaluate(()=>SCEPT_V3.getSnapshot().runId);
 await page.locator('#runExperiment').click();
 await page.waitForFunction(id=>SCEPT_V3.getSnapshot().runId!==id||document.querySelector('#readyLabel')?.dataset.state==='error',old,{timeout:170000});
 const zero=await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot());
 assert(zero.renderedSamples.every(p=>p.rms===0&&!p.markedB),'Zero intensity must remove true difference rings');
 await page.locator('#responseBefore').click();
 const beforeZero=await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot());
 assert(beforeZero.renderedSamples.every((p,i)=>p.colorA===zero.renderedSamples[i].colorB),'Identical baseline and edited tensors must have identical mapped colors');
 await page.locator('#tokenOverview').click();
 assert.equal((await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot())).count,985,'All 5 layers must remain accessible');
 for(const width of [390,768,1440]){
  await page.setViewportSize({width,height:900});
  const d=await page.evaluate(()=>document.documentElement.scrollWidth);
  assert(d<=width+2,'Horizontal overflow '+width+': '+d);
 }
 assert.equal(errors.length,0,'Browser exceptions: '+errors.join('\n'));
 console.log('PASS TOKEN VISUAL CLARITY: three A/B render modes non-blob; quantized color contrast; truthful B delta outlines; single-layer default; all-5 scope; immutable calibration; zero intervention.');
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
