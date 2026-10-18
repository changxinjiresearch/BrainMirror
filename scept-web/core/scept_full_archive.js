'use strict';
/* SCEPT full-observation archive v1: bounded, self-verifying ZIP/STORE.
 * All genuine CAPTURED output tensors are exported in binary, not just NAF
 * summaries. Input tensor is OPTIONAL with explicit prior local consent.
 * No external JSZip/CDN/paid service, no network or cloud upload.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_FULL_ARCHIVE=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='6.5.2',SCHEMA='scept.full-observation.v1',
   MAX_FILES=100,MAX_ARCHIVE_BYTES=160*1024*1024,ENC=new TextEncoder();
 const fail=(code,message)=>{const e=Error('SCEPT Full Archive: '+message);e.code=code;throw e};
 const goodSha=x=>typeof x==='string'&&/^[a-f0-9]{64}$/i.test(x);
 function jsonBytes(x){
  return ENC.encode(JSON.stringify(x,null,2)+'\n');
 }
 function packed(bytes){
  if(!globalThis.crypto?.subtle)fail('CRYPTO_UNAVAILABLE','WebCrypto SHA-256 required');
  return globalThis.crypto.subtle.digest('SHA-256',bytes).then(b=>
   Array.from(new Uint8Array(b),v=>v.toString(16).padStart(2,'0')).join(''));
 }
 function crc32(data){
  let crc=0xffffffff;
  for(const b of data){crc^=b;for(let i=0;i<8;i++)crc=crc&1?(crc>>>1)^0xedb88320:crc>>>1}
  return (crc^0xffffffff)>>>0;
 }
 function zipStored(files){
  if(!Array.isArray(files)||files.length<1||files.length>MAX_FILES)
   fail('RESOURCE_LIMIT','Invalid ZIP file count');
  let byteTotal=22,offset=0;
  const archive=[],central=[],names=new Set();
  for(const f of files){
   if(!f||typeof f.name!=='string'||!/^[a-z0-9][a-z0-9/_.-]{0,160}$/i.test(f.name)||
     f.name.includes('..')||names.has(f.name))
    fail('INVALID_PATH','Unsafe or duplicate archive path');
   names.add(f.name);
   if(!(f.bytes instanceof Uint8Array)||f.bytes.byteLength>=0xffffffff)
    fail('INVALID_FILE','ZIP needs actual Uint8Array bytes');
   const fileName=ENC.encode(f.name),len=f.bytes.byteLength;
   const loc=new Uint8Array(30+fileName.length),h=new DataView(loc.buffer),crc=crc32(f.bytes);
   h.setUint32(0,0x04034b50,true);h.setUint16(4,20,true);
   h.setUint16(6,0x0800,true);h.setUint16(8,0,true); // UTF8 + STORE
   h.setUint32(14,crc,true);h.setUint32(18,len,true);h.setUint32(22,len,true);
   h.setUint16(26,fileName.length,true);loc.set(fileName,30);
   const cd=new Uint8Array(46+fileName.length),v=new DataView(cd.buffer);
   v.setUint32(0,0x02014b50,true);v.setUint16(4,20,true);v.setUint16(6,20,true);
   v.setUint16(8,0x0800,true);v.setUint16(10,0,true);v.setUint32(16,crc,true);
   v.setUint32(20,len,true);v.setUint32(24,len,true);v.setUint16(28,fileName.length,true);
   v.setUint32(42,offset,true);cd.set(fileName,46);
   archive.push(loc,f.bytes);central.push(cd);
   offset+=loc.length+len;byteTotal+=loc.length+len+cd.length;
   if(byteTotal>MAX_ARCHIVE_BYTES)fail('RESOURCE_LIMIT','Full archive >160MiB. Reduce observed probes; never omit tensors silently.');
  }
  const cdStart=offset;
  for(const c of central){archive.push(c);offset+=c.length}
  const footer=new Uint8Array(22),e=new DataView(footer.buffer);
  e.setUint32(0,0x06054b50,true);e.setUint16(8,files.length,true);e.setUint16(10,files.length,true);
  e.setUint32(12,offset-cdStart,true);e.setUint32(16,cdStart,true);
  archive.push(footer);
  return new Blob(archive,{type:'application/zip'});
 }
 function compatible(capture,naf){
  if(!capture||capture.isReleased?.()||typeof capture.exportRecord!=='function'||
   typeof capture.getTensor!=='function'||typeof capture.listStored!=='function')
   fail('NOT_AVAILABLE','Latest real capture with all raw outputs required');
  const r=capture.exportRecord();
  if(r.schema!=='scept.run-record.v1'||naf?.schema!=='scept.naf.v1'||naf.runId!==r.runId||
   !Array.isArray(r.outputs)||!r.outputs.length||
   r.science?.status!=='observation_only'||naf.science?.status!=='single_run_observation_only')
   fail('INVALID_PROVENANCE','Run Record / NAF scientific identity mismatch');
  if(!goodSha(r.provenance?.model?.sha256)||!Object.values(r.provenance.inputs||{}).every(goodSha))
   fail('INVALID_PROVENANCE','Model/input SHA missing');
  return r;
 }
 async function build({capture,naf,reference=null,includeInputs=false,tensorSha}={}){
  const record=compatible(capture,naf);
  if(typeof tensorSha!=='function')fail('INVALID_DEPENDENCY','V6.1 tensorSha required to verify actual source output bytes');
  if(typeof includeInputs!=='boolean')fail('INVALID_OPTION','includeInputs must be explicit');
  const stored=capture.listStored();
  const expected=record.outputs.map(o=>o.name),extra=stored.filter(name=>!expected.includes(name));
  if(extra.length||expected.some(name=>!stored.includes(name)))
   fail('INCOMPLETE_CAPTURE','One or more actual declared outputs are missing. Export fails closed.');
  const files=[],entries=[];
  async function push(name,bytes,meta={}){
   if(!(bytes instanceof Uint8Array))fail('INVALID_BYTES','Expected actual file bytes');
   const copy=new Uint8Array(bytes); // no alias to future mutating tensor memory
   const sha256=await packed(copy);
   files.push({name,bytes:copy});
   entries.push({path:name,bytes:copy.byteLength,sha256,...meta});
  }
  await push('record.json',jsonBytes(record),{kind:'run_record'});
  await push('naf.json',jsonBytes(naf),{kind:'single_run_naf'});
  if(reference!==null){
   if(reference?.schema!=='scept.naf.reference.v1'||
     reference.modelSha256!==record.provenance.model.sha256||
     reference.preprocessSha256!==record.provenance.preprocessing.sha256||
     reference.profile!==naf.profile)
    fail('INVALID_REFERENCE','Reference must be genuinely matched to this model and preprocessing');
   await push('reference.json',jsonBytes(reference),{kind:'empirical_reference',note:'within-volume ranks are not independent patient reference'});
  }
  let outputBytes=0;
  for(let i=0;i<record.outputs.length;i++){
   const item=record.outputs[i],t=capture.getTensor(item.name);
   if(!t||t.type!==item.type||JSON.stringify(t.dims)!==JSON.stringify(item.shape)||
     t.data.length!==item.elements)fail('TENSOR_MISMATCH','Raw output metadata mismatch: '+item.name);
   if(!goodSha(item.sha256)||await tensorSha(t)!==item.sha256.toLowerCase())
    fail('TENSOR_HASH_MISMATCH','Captured output does not match run-record source SHA: '+item.name);
   const bytes=new Uint8Array(t.data.buffer,t.data.byteOffset,t.data.byteLength),
     path='tensors/outputs/'+String(i).padStart(3,'0')+'.bin';
   outputBytes+=bytes.byteLength;
   await push(path,bytes,{kind:'observed_output_tensor',tensorName:item.name,
    dtype:t.type,shape:t.dims.slice(),tensorSha256:item.sha256,byteOrder:'little-endian-runtime-typed-array'});
  }
  const retainedInputs=typeof capture.listStoredInputs==='function'?capture.listStoredInputs():[];
  if(includeInputs){
   const required=record.signature.inputs.map(x=>x.name);
   if(!required.length||required.some(name=>!retainedInputs.includes(name)))
    fail('INPUT_NOT_RETAINED','Raw model inputs were not consented to / preserved BEFORE running. Rerun with input retention selected.');
   for(let i=0;i<record.signature.inputs.length;i++){
    const signature=record.signature.inputs[i],t=capture.getInputTensor(signature.name);
    if(!t||t.type!==signature.type||
       t.dims.length!==signature.shape.length||
       t.dims.some((d,j)=>typeof signature.shape[j]==='number'&&signature.shape[j]!==d)||
       await tensorSha(t)!==record.provenance.inputs[signature.name])
     fail('INPUT_HASH_MISMATCH','Original model feed is not verifiably identical to actual inference');
    await push('tensors/inputs/'+String(i).padStart(3,'0')+'.bin',
     new Uint8Array(t.data.buffer,t.data.byteOffset,t.data.byteLength),
     {kind:'actual_model_input_tensor',tensorName:signature.name,dtype:t.type,
      shape:t.dims.slice(),tensorSha256:record.provenance.inputs[signature.name],
      byteOrder:'little-endian-runtime-typed-array',userConsent:true});
   }
  }
  const availability=record.availability||[];
  const manifest={schema:SCHEMA,version:VERSION,runId:record.runId,createdAt:record.createdAt,
   scientificStatus:'OBSERVATION_ONLY_NOT_CAUSAL_OR_CLINICALLY_VALIDATED',
   captureScope:{allDeclaredRuntimeOutputs:true,storedOutputTensors:expected.length,
    rawOutputBytes:outputBytes,inputsIncluded:includeInputs,modelWeightsIncluded:false,
    originalNiftiSourceIncluded:false,otherModelInternalsNotInstrumented:true,
    unavailableProbes:availability.filter(o=>o.status!=='measured'),
    previousRunRawDataIncluded:false},
   notes:['A complete archive of ACTUALLY CAPTURED outputs, not every model intermediate.',
    'The model input is the exact preprocessed tensor, not the original NIfTI source.',
    'Model weights and uninstrumented layers are NOT included.',
    'Tensor paths are generic indices; tensorName gives the actual runtime identity.',
    'Verify SHA-256 and tensorSha256 (type+shape+raw bytes) before scientific reuse.'],
   files:entries.slice()};
  await push('manifest.json',jsonBytes(manifest),{kind:'archive_manifest'});
  const blob=zipStored(files);
  return {blob,manifest,size:blob.size,fileCount:files.length,
   outputTensorCount:expected.length,includesInputs:includeInputs,
   suggestedFilename:'scept-'+record.runId+'-full-observation.zip'};
 }
 return Object.freeze({VERSION,SCHEMA,build,zipStored,crc32,packed});
});
