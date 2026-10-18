'use strict';
/* v4 scientific visual CONTRACT tests: artificial data only, not U11 science. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {selectedPatch,summary}=require('../scept_v4_ceds.js');
const html=fs.readFileSync(path.join(__dirname,'..','v4.html'),'utf8');
for(const id of ['v4RunEvidence','v4Signature','v4Gates','v4Verdict','v4Target','v4Roi','v4ExportEvidence'])
  assert.match(html,new RegExp('id="'+id+'"'),'Missing '+id);
assert.match(html,/scept_v4_ceds\.js/);
assert.match(html,/scept_activity3d\.js/);
assert.match(html,/v3_mri\.js/);
assert.match(html,/u10_evidential_dependence\.js/);
assert.match(html,/M4：尚未通过/);
const input=new Float32Array(3*224*224);
const changed=new Float32Array(input);
for(let c=0;c<3;c++)for(let y=16;y<20;y++)for(let x=12;x<16;x++)
  changed[c*224*224+y*224+x]=.25;
const selected=selectedPatch(input,changed);
assert.equal(selected.p.x,12);assert.equal(selected.p.y,16);
assert.equal(selected.p.ids.length,48);assert.equal(selected.q.ids.length,48);
assert.equal(new Set(selected.p.ids).size,48);
assert.equal(selected.p.ids.filter(x=>selected.q.ids.includes(x)).length,0);
assert(selected.p.ids.every((x,i)=>i===0||x>selected.p.ids[i-1]));
assert.throws(()=>selectedPatch(input,input),/没有进入模型/);
assert.throws(()=>selectedPatch(new Float32Array(99),changed),/真实/);
const tasks=[];
for(const seed of [109,211,307])for(const dose of [0,.5,1])
  tasks.push({status:'completed',explanation:{
    schema:'scept.u10.explanation.v1',nuisanceSeed:seed,strength:dose,
    targetedEffect:.01+dose*.20,matchedControlEffect:.01+dose*.05,
    controlledMagnitude:dose*.15,score:dose*.15
  }});
const e={schema:'scept.u7.evidence.v1',tasks,suiteTests:[{
  plugin:'scept.u10.eval.preregistered-gates',verdict:'criterion_not_met',
  gates:[{name:'graded-sensitivity',value:1,verdict:'criterion_not_met'}]
}],provenance:{modelSha256:'a'.repeat(64)},claims:{causal:'NOT_ESTABLISHED'}};
const result=summary(e);
assert.equal(result.points.length,3);
assert.equal(result.points[0].p,result.points[0].q);
assert.equal(result.points[2].specificity,.15);
assert.equal(result.scientificVerdict,'criterion_not_met');
assert.equal(result.caseCount,9);
assert.equal(result.gates[0].verdict,'criterion_not_met');
assert.throws(()=>summary({...e,tasks:tasks.slice(1)}),/不完整/);
assert.throws(()=>summary({...e,suiteTests:[]}),/U10/);
assert.throws(()=>summary({...e,tasks:tasks.map((x,i)=>i===1?{...x,explanation:{...x.explanation,score:NaN}}:x)}),/无效/);
console.log('PASS v4 CEDS: real-evidence-only plot gate, valid P/Q input mapping, no fabricated empty signature, negative verdict kept');
