'use strict';
/* SCEPT V6.5: bounded, local-only, exportable scientific run journal.
 * Imported manifests are NEVER promoted to trusted runtime captures.
 * No persistence, external upload, network, A/B comparison or new inference.
 */
(function(root,factory){
 const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_WORKSPACE_JOURNAL=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='6.5.0',SCHEMA='scept.workspace.journal.v1';
 const HEX=/^[0-9a-f]{64}$/i,MAX_RUNS=32,MAX_BYTES=8*1024*1024;
 const fail=(code,m)=>{const e=Error('SCEPT Workspace Journal: '+m);e.code=code;throw e};
 const plain=o=>!!o&&typeof o==='object'&&!Array.isArray(o)&&Object.getPrototypeOf(o)===Object.prototype;
 function ensureNoRaw(value,depth=0){
  if(depth>22)fail('INVALID_DEPTH','Untrusted archive nesting too deep');
  if(value===null||typeof value==='string'||typeof value==='boolean')return;
  if(typeof value==='number'&&Number.isFinite(value))return;
  if(typeof value==='number')fail('NONFINITE','Nonfinite archived number');
  if(Array.isArray(value)){if(value.length>200000)fail('TOO_LARGE','Archive array too long');
   value.forEach(v=>ensureNoRaw(v,depth+1));return}
  if(!plain(value))fail('INVALID_DATA','Unexpected runtime tensor/buffer in export');
  if(Object.keys(value).length>128)fail('TOO_LARGE','Too many archive properties');
  for(const [key,v] of Object.entries(value)){
   if(['data','values','raw','image','pixels','patientData','voxelData','tensorData','rawTensors'].includes(key))
    fail('RAW_DATA','Raw model/patient tensor data must never be archived');
   ensureNoRaw(v,depth+1);
  }
 }
 function copy(value){const json=JSON.stringify(value);if(json.length>MAX_BYTES)fail('TOO_LARGE','Archive too large');
  const val=JSON.parse(json);ensureNoRaw(val);return val}
 function validatePair(record,naf){
  if(!plain(record)||record.schema!=='scept.run-record.v1'||
    !plain(naf)||naf.schema!=='scept.naf.v1')
   fail('INVALID_SCHEMA','Authentic V6 Run Record and NAF schemas required');
  if(typeof record.runId!=='string'||!/^(run-|import-)[a-z0-9_-]{10,120}$/i.test(record.runId)||
    naf.runId!==record.runId)fail('RUN_ID','Run identity mismatch');
  if(!HEX.test(record.provenance?.model?.sha256||'')||
   !HEX.test(record.provenance?.preprocessing?.sha256||'')||
   naf.provenance?.model?.sha256!==record.provenance.model.sha256||
   naf.provenance?.preprocessing?.sha256!==record.provenance.preprocessing.sha256)
   fail('PROVENANCE','Model/preprocessing identity mismatch');
  const inputs=record.provenance.inputs;
  if(!plain(inputs)||!Object.keys(inputs).length||Object.values(inputs).some(x=>!HEX.test(x))||
    JSON.stringify(inputs)!==JSON.stringify(naf.provenance.inputs))
    fail('PROVENANCE','Named input hashes do not match');
  if(record.science?.status!=='observation_only'||
     naf.science?.status!=='single_run_observation_only')
    fail('OVERCLAIM','Journal accepts numeric observation-only evidence');
  if(!Array.isArray(naf.groups)||naf.groups.length>64||!Array.isArray(record.outputs)||
   !record.outputs.length||record.outputs.length>64||
   record.outputs.some(o=>typeof o.name!=='string'||!HEX.test(o.sha256||'')))
   fail('INVALID_DATA','Measured provenance and groups required');
  const declared=new Set(record.observations?.filter(o=>o.status==='measured').map(o=>o.tensor)||[]);
  for(const g of naf.groups){
   if(typeof g.tensor!=='string'||typeof g.status!=='string'||!Array.isArray(g.units)||
    g.units.length>4096)fail('INVALID_GROUP','Invalid group or number of units');
   if(g.status==='measured'&&!declared.has(g.tensor))
    fail('UNTRUSTED_GROUP','NAF claims an unrecorded actual tensor');
   for(const u of g.units)if(!Number.isInteger(u.index)||u.index<0||
     !Number.isFinite(u.rms)||u.rms<0)
     fail('INVALID_GROUP','Each reported RMS must be finite nonnegative');
  }
  if(!Array.isArray(naf.updates)||naf.updates.length>32||
    naf.updates.some(u=>!['measured','not_evaluable'].includes(u.status)))
   fail('INVALID_UPDATE','Invalid update status');
 }
 function sanitizeNote(note){if(typeof note!=='string'||note.length>500)
  fail('INVALID_NOTE','Notes limited to 500 characters');return note}
 function make({maxRuns=MAX_RUNS}={}){
  if(!Number.isInteger(maxRuns)||maxRuns<1||maxRuns>MAX_RUNS)fail('INVALID_CONFIG','1..32 run budget');
  let entries=[],project='SCEPT research session';
  function add(record,naf,{source='live_runtime',note=''}={}){
   if(!['live_runtime','imported_unverified'].includes(source))
    fail('INVALID_SOURCE','Only real local captures and labelled unverified imports');
   const data=copy({record,naf});validatePair(data.record,data.naf);
   if(entries.some(e=>e.runId===record.runId))fail('DUPLICATE','Same runId already in journal');
   const out={runId:record.runId,source,note:sanitizeNote(note),record:data.record,naf:data.naf};
   entries.push(out);if(entries.length>maxRuns)entries.shift();
   return summarize(out);
  }
  function summarize(e){
   const {record:r,naf:n}=e;
   return{runId:e.runId,source:e.source,note:e.note,
    at:r.createdAt||'',model:r.provenance.model.id||'',modelSha256:r.provenance.model.sha256,
    inputSha256:Object.values(r.provenance.inputs)[0],
    profile:n.profile||'unknown',
    measuredTensors:n.groups.filter(g=>g.status==='measured').length,
    measuredUnits:n.groups.reduce((a,g)=>a+(g.status==='measured'?g.units.length:0),0),
    updateStatus:n.updates.map(u=>u.status)};
  }
  function list(){return entries.map(summarize)}
  function setProject(name){
   if(typeof name!=='string'||name.length>110||!name.trim())fail('INVALID_PROJECT','Project name required');
   project=name.trim();
  }
  function setNote(id,note){let e=entries.find(e=>e.runId===id);
   if(!e)fail('NOT_FOUND','Run not found');e.note=sanitizeNote(note)}
  function remove(id){let before=entries.length;entries=entries.filter(e=>e.runId!==id);
   if(entries.length===before)fail('NOT_FOUND','Run not found')}
  function clear(){entries=[]}
  function get(id){const e=entries.find(e=>e.runId===id);
   return e?copy(e):null}
  function exportManifest(ids){
   const selected=ids===undefined?entries:entries.filter(e=>ids.includes(e.runId));
   if(!selected.length)fail('EMPTY','No runs to export');
   // Claim provenance: a manifest is a *collection of self-reported records*,
   // not third-party verification of model weights or their scientific truth.
   const manifest={schema:SCHEMA,version:VERSION,project,exportKind:'summary_only',
    security:'no raw MRI or full tensor values; import never grants trust',
    runs:selected.map(e=>({source:e.source,note:e.note,record:e.record,naf:e.naf}))};
   return copy(manifest)
  }
  function importManifest(manifest){
   const checked=copy(manifest);
   if(checked.schema!==SCHEMA||checked.exportKind!=='summary_only'||
    !Array.isArray(checked.runs)||!checked.runs.length||checked.runs.length>maxRuns)
    fail('INVALID_SCHEMA','Bounded SCEPT workspace journal v1 required');
   // All-or-nothing; invalid imports cannot partially mutate a session.
   const unique=new Set(entries.map(e=>e.runId));
   const pending=[];
   for(const x of checked.runs){
    validatePair(x.record,x.naf);
    if(unique.has(x.record.runId))fail('DUPLICATE','Imported run ID already exists');
    unique.add(x.record.runId);
    pending.push({runId:x.record.runId,source:'imported_unverified',
     note:sanitizeNote(x.note||''),record:x.record,naf:x.naf});
   }
   entries.push(...pending);if(entries.length>maxRuns)entries=entries.slice(-maxRuns);
   return pending.length
  }
  return Object.freeze({addLive:(r,n)=>add(r,n),list,get,setProject,setNote,
   remove,clear,exportManifest,importManifest,projectName:()=>project});
 }
 return Object.freeze({VERSION,SCHEMA,make,validatePair});
});
