'use strict';
/* Engineering contract fixtures are synthetic, never clinical evidence. */
const assert=require('node:assert/strict');
const fs=require('node:fs');const path=require('node:path');
const S=require('../scept_v5_science.js');
const h=x=>String(x).repeat(64);
function sample(seed,shift){
 const a=new Float32Array(1*5*3),b=new Float32Array(a.length);
 for(let u=0;u<5;u++)for(let j=0;j<3;j++){
  const k=u*3+j;a[k]=u/10+j/100;b[k]=a[k]+(u===2?.5:u/40)*shift+seed.charCodeAt(0)*.00001*u;
 }
 return S.ndf({modelSha256:h('a'),baselineHash:h('b'),counterfactualHash:h(seed),
  layers:[{name:'true_hidden',layer:6,a:{dims:[1,5,3],data:a},b:{dims:[1,5,3],data:b}}]});
}
const x=sample('c',1);
assert.equal(x.rows[0].units,5);
assert(x.rows[0].values[2].rms>x.rows[0].values[1].rms);
assert.equal(x.scientificStatus,'descriptive_internal_response_only');
assert.equal(x.rows[0].values[0].rms,0);
assert.throws(()=>S.ndf({...x,source:'simulation'}),/model SHA|layer data/);
const r=S.crc([sample('c',1),sample('d',1.04),sample('e',.98)]);
assert.equal(r.nRealizations,3);
assert.equal(r.rows[0].topKOverlap,1);
assert(r.rows[0].rankCorrelation>.9);
assert.equal(r.verdict,'measured');
assert.throws(()=>S.crc([sample('c',1),sample('c',1.01),sample('e',1)]),/duplicate/);
assert.throws(()=>S.crc([sample('c',1),sample('d',1.01)]),/at least 3/);
const real=S.icm({modelSha256:h('a'),layerName:'actual-onxx-hidden',unit:3,
 sA:2,sB:.5,sPatchedAB:1.7,sPatchedBA:.7,offA:2,offB:.5,tolerance:.001});
assert.equal(real.verdict,'measured');
assert(Math.abs(real.recoveryFraction-.8)<1e-9);
assert.throws(()=>S.icm({modelSha256:h('a'),layerName:'x',unit:0,
 sA:2,sB:.5,sPatchedAB:1.7,sPatchedBA:.7,offA:1.8,offB:.5,tolerance:.001}),/no-op/);
const zero=S.icm({modelSha256:h('a'),layerName:'x',unit:0,
 sA:.5,sB:.5,sPatchedAB:.5,sPatchedBA:.5,offA:.5,offB:.5,tolerance:.001});
assert.equal(zero.verdict,'not_evaluable');
const html=fs.readFileSync(path.join(__dirname,'..','v5.html'),'utf8');
for(const id of ['v5Observe','v5PairRun','v5BFiles','v5Mask','v5Fingerprint','v5CrcRun','v5Export','v5Status'])
 assert(html.includes('id="'+id+'"'),id+' missing');
assert(html.includes('scept_v5_science.js')&&html.includes('scept_v5_ui.js'));
assert(html.includes('scept_activity3d.js')&&html.includes('scept_v4_ceds.js'));
console.log('PASS V5: real-tensor-only NDF, multi-B CRC, real-runtime ICM parity gate, transparent NIfTI/UI contracts');
