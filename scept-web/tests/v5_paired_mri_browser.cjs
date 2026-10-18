'use strict';
/* Real MNI152 MRI transformed into explicitly SYNTHETIC mask/B fixtures for
 * engineering pipeline acceptance; NEVER pathology validation. */
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1420,height:950}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.SCEPT_V5_URL||'http://127.0.0.1:8765/v5.html',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>SCEPT_V3?.getSnapshot().ready||
    document.querySelector('#readyLabel').dataset.state==='error',null,{timeout:180000});
  assert((await page.evaluate(()=>SCEPT_V3.getSnapshot())).ready,await page.locator('#warning').innerText());
  const prep=await page.evaluate(async()=>{
   const m=SCEPT_MRI.state;
   if(!m.affine)throw Error('MNI152 public fixture missing sform; strict research gate correctly refuses');
   const request=await fetch('https://niivue.github.io/niivue-demo-images/mni152.nii.gz');
   if(!request.ok)throw Error('Public authentic MRI unavailable');
   const base=await SCEPT_MRI.decodeFile(await request.arrayBuffer(),'mni152.nii.gz');
   const d=base.dimensions,n=base.values.length,c=m.center;
   const ix=(x,y,z)=>x+d[0]*(y+d[1]*z);
   const lesion=new Float32Array(n),positions=[];
   for(let z=c[2]-4;z<=c[2]+4;z++)for(let y=c[1]-5;y<=c[1]+5;y++)
   for(let x=c[0]-5;x<=c[0]+5;x++){
    if((x-c[0])**2+(y-c[1])**2+(z-c[2])**2>25)continue;
    if(x<0||y<0||z<0||x>=d[0]||y>=d[1]||z>=d[2])continue;
    const i=ix(x,y,z);lesion[i]=1;positions.push(i);
   }
   if(!positions.length)throw Error('No mask voxels created');
   const build=(vals)=>{
    const buf=new ArrayBuffer(352+n*4),view=new DataView(buf);
    view.setInt32(0,348,true);view.setInt16(40,3,true);
    d.forEach((v,i)=>view.setInt16(42+2*i,v,true));
    view.setInt16(70,16,true);view.setInt16(72,32,true);
    base.spacing.forEach((v,i)=>view.setFloat32(80+4*i,v,true));
    view.setFloat32(108,352,true);
    view.setInt16(254,1,true);
    for(let i=0;i<3;i++)for(let j=0;j<4;j++)
      view.setFloat32(280+16*i+4*j,base.affine[i][j],true);
    [110,43,49,0].forEach((v,i)=>view.setUint8(344+i,v));
    new Float32Array(buf,352,n).set(vals);
    return buf;
   };
   const maskData=new DataTransfer();maskData.items.add(new File([build(lesion)],'synthetic-mask.nii',{type:'application/octet-stream'}));
   document.getElementById('v5Mask').files=maskData.files;
   const bs=new DataTransfer();
   for(let k=1;k<=3;k++){
    const vals=new Float32Array(base.values);
    positions.forEach(i=>{vals[i]=vals[i]+k*(base.limits[1]-base.limits[0])*.06});
    bs.items.add(new File([build(vals)],'synthetic-b'+k+'.nii',{type:'application/octet-stream'}));
   }
   document.getElementById('v5BFiles').files=bs.files;
   return{count:positions.length,geometry:'public authentic MNI MRI transformed for numeric software test only'};
  });
  await page.locator('#v5PairRun').click();
  await page.waitForFunction(()=>SCEPT_V5.getSnapshot().measuredRealizations===3||
    document.querySelector('#v5Status').dataset.error==='true',null,{timeout:160000});
  const snap=await page.evaluate(()=>SCEPT_V5.getSnapshot());
  assert.equal(snap.measuredRealizations,3,'Real paired pipeline failed '+await page.locator('#v5Status').innerText());
  assert.equal(snap.mode,'uploaded-mri-counterfactual-not-validated-pathology');
  assert.equal(await page.locator('#v5CrcRun').isEnabled(),true);
  await page.locator('#v5CrcRun').click();
  assert.match(await page.locator('#v5Crc').innerText(),/Top-K/);
  assert.equal((await page.evaluate(()=>SCEPT_V5.getNdf())).rows.length,5);
  assert.equal(errors.length,0,'Uncaught errors '+JSON.stringify(errors));
  console.log('PASS genuine pretrained ONNX NDF + CRC 3 B: full real NIfTI geometry/mask, authentic runtime repeated inference; fabricated lesion fixture for engineering ONLY, not medical result');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
