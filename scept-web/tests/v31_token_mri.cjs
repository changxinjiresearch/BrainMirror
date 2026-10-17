const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
try{
 const page=await browser.newPage({viewport:{width:1366,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.stack||e.message));
 await page.goto(process.env.SCEPT_V31_URL||'http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().ready||document.getElementById('readyLabel').dataset.state==='error',null,{timeout:150000});
 assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,'Frozen pretrained ViT and true MNI input not ready: '+await page.locator('#warning').innerText());
 const initial=await page.evaluate(()=>({mri:SCEPT_MRI.state,patch:SCEPT_PATCH.getSnapshot()}));
 assert(initial.mri.loaded,'NIfTI not loaded');
 assert.equal(initial.patch.map.patches.length,196);
 assert(initial.patch.map.changedPatches>0,'Initial ROI must affect visible model input');
 // ROI centre and anatomical slice navigation are distinct state variables.
 await page.locator('#sliceI').fill('80');
 const moved=await page.evaluate(()=>SCEPT_MRI.state);
 assert.deepEqual(moved.center,initial.mri.center,'Slice navigation must never alter 3D ROI centre');
 assert.notDeepEqual(moved.slice,initial.mri.slice,'Slice slider must update only slice coordinate');
 await page.locator('#runExperiment').click();
 await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().hasExperiment||document.querySelector('#readyLabel').dataset.state==='error',null,{timeout:160000});
 const first=await page.evaluate(()=>{
  const r=SCEPT_V3.getAudit(),t=SCEPT_ACTIVITY3D.getSnapshot(),p=SCEPT_PATCH.getSnapshot().map;
  return{at:r.at,shape:r.layers.map(l=>l.shape),token:t,patch:p,changed:r.inputPatchMap.changedPatches};
 });
 assert(first.token.layers.length===5,'Five real observed ViT layers missing');
 assert.equal(first.token.layers.reduce((a,l)=>a+l.tokens.length,0),985);
 assert(first.shape.every(d=>JSON.stringify(d)==='[1,197,192]'),'Unsupported tensor order');
 assert.equal(first.changed,first.patch.changedPatches,'Real ROI patch counts mismatch');
 assert.equal(first.token.count,985,'3D token glyphs not all present in all-layer mode');
 assert(first.token.layers.some(l=>l.tokens.some(t=>t.rms>0)),'No true nonzero token changes');
 // Verify one Token summary independently against the unmodified raw ONNX output.
 const parity=await page.evaluate(()=>{
  const r=SCEPT_V3.getAudit().layers.find(x=>x.layer===6),
    t=SCEPT_ACTIVITY3D.getSnapshot().layers.find(x=>x.layer===6).tokens[85];
  let rms=0,mae=0,before=0,after=0;
  for(let ch=0;ch<192;ch++){const k=85*192+ch,a=r.before[k],b=r.after[k];
   rms+=(b-a)**2;mae+=Math.abs(b-a);before+=Math.abs(a);after+=Math.abs(b)}
  return{raw:{rms:Math.sqrt(rms/192),mae:mae/192,before:before/192,after:after/192},display:t};
 });
 for(const k of ['rms','mae','before','after'])assert(Math.abs(parity.raw[k]-parity.display[k])<1e-8,'Token '+k+' not traceable to true source tensor');
 await page.locator('#viewTokens').click();
 assert.equal(await page.locator('#token3dCanvas').isVisible(),true);
 assert.equal(await page.locator('#operator3dShell').isVisible(),false);
 assert.equal((await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot())).mode,'delta');
 await page.locator('#responseCompare').click();
 assert.equal((await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot())).mode,'compare');
 await page.locator('#tokenZoomIn').click();
 const zoom=await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot());
 const coords=await page.locator('#token3dCanvas').boundingBox();
 await page.mouse.move(coords.x+coords.width*.5,coords.y+coords.height*.5);await page.mouse.down();
 await page.mouse.move(coords.x+coords.width*.75,coords.y+coords.height*.7,{steps:5});await page.mouse.up();
 assert(Math.abs((await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot())).yaw-zoom.yaw)>.1,'3D Token orbit camera failed');
 await page.locator('#tokenEnter').click();
 assert.equal((await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot())).count,197,'Entering layer must show exactly 197 tokens');
 // Click input Patch and inspect true matching Token, 192 channels.
 const patch=page.locator('#patchInputCanvas'),box=await patch.boundingBox();
 await page.mouse.click(box.x+(7.5/14)*box.width,box.y+(6.5/14)*box.height);
 const linked=await page.evaluate(()=>({p:SCEPT_PATCH.getSnapshot().selected,t:SCEPT_ACTIVITY3D.getSnapshot().selected}));
 assert.equal(linked.t.token,linked.p+1,'Selected exact input patch not linked to corresponding measured token');
 assert.equal(await page.locator('#tokenDetails').isVisible(),true);
 await page.locator('#tokenChannelSelect').fill('117');
 assert.match(await page.locator('#tokenChannelInfo').innerText(),/通道 117/);
 // Rerun with a different physical 3D ROI perturbation while keeping the frozen weights.
 const firstRms=first.token.layers.find(l=>l.layer===6).tokens.map(t=>t.rms);
 await page.locator('#mriAmount').fill('0.65');
 const old=await page.evaluate(()=>SCEPT_V3.getSnapshot().runId);
 await page.locator('#runExperiment').click();
 await page.waitForFunction(id=>SCEPT_V3.getSnapshot().runId!==id||document.getElementById('readyLabel').dataset.state==='error',old,{timeout:160000});
 const changed=await page.evaluate(()=>SCEPT_ACTIVITY3D.getSnapshot());
 assert(changed.layers.find(l=>l.layer===6).tokens.some((t,i)=>Math.abs(t.rms-firstRms[i])>1e-6),'Token states do not reflect a second real inference');
 assert.equal(changed.scaleD,first.token.scaleD,'Color calibration must remain fixed across reruns');
 await page.locator('#mriAmount').fill('0');
 const prev=await page.evaluate(()=>SCEPT_V3.getSnapshot().runId);
 await page.locator('#runExperiment').click();
 await page.waitForFunction(id=>SCEPT_V3.getSnapshot().runId!==id||document.getElementById('readyLabel').dataset.state==='error',prev,{timeout:160000});
 const noIntervention=await page.evaluate(()=>({a:SCEPT_ACTIVITY3D.getSnapshot(),p:SCEPT_PATCH.getSnapshot()}));
 assert.equal(noIntervention.p.map.changedPatches,0);
 assert(noIntervention.a.layers.every(l=>l.tokens.every(t=>t.rms===0)),'Zero intervention must mean zero actual token deltas');
 assert.equal(errors.length,0,'Uncaught exceptions '+JSON.stringify(errors));
 for(const width of [390,768,1366]){await page.setViewportSize({width,height:900});const w=await page.evaluate(()=>document.documentElement.scrollWidth);assert(w<=width+2,'Mobile layout overflow '+width+': '+w)}
 await page.screenshot({path:'/tmp/scept-v31-real-tokens.png',fullPage:true});
 console.log('PASS SCEPT v3.1: actual MRI input patch correspondence; strict tensor axes; 985 true Tokens; 192-channel detail; 3D orbit, A/B; changed-ROI reproducibility and zero-perturbation falsification.');
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
