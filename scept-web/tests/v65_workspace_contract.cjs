'use strict';
/* V6.5 data contract tests, all records below are synthetic SOFTWARE fixtures
 * for manifest safety/logic only; NOT model evidence. */
const assert=require('node:assert/strict'),J=require('../core/scept_workspace_journal.js');
const sha=x=>x.repeat(64),ID=x=>'run-'+x.repeat(32);
function specimen(x='a'){
 const id=ID(x),provenance={model:{id:'synthetic-contract-only',sha256:sha('a')},
  preprocessing:{id:'test',sha256:sha('b')},inputs:{pixel_values:sha(x)}};
 return{record:{schema:'scept.run-record.v1',runId:id,createdAt:'2026-10-18T00:00:00Z',
   provenance,science:{status:'observation_only',causal:'NOT_ESTABLISHED'},
   outputs:[{name:'internal',sha256:sha('c'),shape:[1,2,3]}],
   observations:[{tensor:'internal',status:'measured',activity:{source:'actual-runtime-output'}}]},
   naf:{schema:'scept.naf.v1',runId:id,profile:'vit-token-channel',
    provenance:JSON.parse(JSON.stringify(provenance)),
    groups:[{tensor:'internal',status:'measured',role:'token',shape:[1,2,3],
     units:[{index:0,rms:.43},{index:1,rms:.79}]}],
    updates:[{status:'not_evaluable',reason:'probe_not_exposed',units:[]}],
    science:{status:'single_run_observation_only'}}
 };
}
const j=J.make(),a=specimen('a'),b=specimen('b');
assert.equal(j.addLive(a.record,a.naf).measuredUnits,2);
assert.equal(j.list()[0].source,'live_runtime');
assert.throws(()=>j.addLive(a.record,a.naf),/already/);
j.addLive(b.record,b.naf);assert.equal(j.list().length,2);
j.setNote(b.record.runId,'Directly observed numeric state only');
j.setProject('Nonclinical research project');
const archive=j.exportManifest();
assert.equal(archive.schema,'scept.workspace.journal.v1');
assert.equal(archive.exportKind,'summary_only');
assert(!JSON.stringify(archive).includes('"data"'));
assert(!JSON.stringify(archive).includes('"raw"'));
const restored=J.make();assert.equal(restored.importManifest(archive),2);
assert.equal(restored.list()[0].source,'imported_unverified');
assert.equal(restored.list()[1].note,'Directly observed numeric state only');
assert.throws(()=>restored.importManifest(archive),/already/);
const c=specimen('c');
assert.throws(()=>restored.addLive(c.record,{...c.naf,runId:'run-other_run_id'}),/identity mismatch/);
const fake=specimen('d');fake.naf.groups[0].tensor='invented_hidden';
assert.throws(()=>restored.addLive(fake.record,fake.naf),/unrecorded/);
const patient=specimen('e');patient.record.data=new Float32Array([1,2]);
assert.throws(()=>restored.addLive(patient.record,patient.naf),/Raw model|Unexpected runtime/);
const smuggle=specimen('e');smuggle.naf.values=[9,9,9];
assert.throws(()=>restored.addLive(smuggle.record,smuggle.naf),/Raw model/);
const suspicious=specimen('f');suspicious.record.provenance.model.sha256='z'.repeat(64);
assert.throws(()=>restored.addLive(suspicious.record,suspicious.naf),/Model\/preprocessing/);
const invalid=specimen('f');invalid.naf.groups[0].units[0].rms=Number.NaN;
assert.throws(()=>restored.addLive(invalid.record,invalid.naf),/Nonfinite|finite/);
const m=JSON.parse(JSON.stringify(archive));m.runs.push({record:fake.record,naf:fake.naf,note:''});
const previous=restored.list().length;
assert.throws(()=>restored.importManifest(m),/unrecorded/);
assert.equal(restored.list().length,previous,'Batch invalid import must be atomic');
restored.remove(a.record.runId);assert.equal(restored.list().length,1);
restored.clear();assert.equal(restored.list().length,0);
assert.throws(()=>restored.exportManifest(),/No runs/);
console.log('PASS V6.5 journal contract: local-only real-record shape, strict safe summary export, unverified import, atomic rejection and no raw MRI tensors');
