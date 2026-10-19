'use strict';
/* Independent JS check of Python ZIP, same native crypto/ZIP primitive as web.
 * Python-exported binary should be readable with the browser's format rules.
 * File+tensor hash agreement cannot prove third-party author/model identity.
 */
const fs=require('node:fs'),assert=require('node:assert/strict');
const crypto=require('node:crypto'),Archive=require('../../scept-web/core/scept_full_archive.js');
const file=process.argv[2];
if(!file)throw Error('Usage: node verify_python_zip_in_node.cjs <archive.zip>');
const raw=fs.readFileSync(file),entries=new Map();
if(raw.length>160*1024*1024)throw Error('Oversized ZIP');
let p=0;
while(p+30<raw.length&&raw.readUInt32LE(p)===0x04034b50){
 const flags=raw.readUInt16LE(p+6),method=raw.readUInt16LE(p+8),
  crc=raw.readUInt32LE(p+14),count=raw.readUInt32LE(p+18),
  n=raw.readUInt16LE(p+26),ext=raw.readUInt16LE(p+28);
 assert.equal(flags&0x0008,0,'No data descriptor in portable ZIP');
 assert.equal(method,0,'Uncompressed ZIP/STORE for independent access');
 const name=raw.subarray(p+30,p+30+n).toString('utf8'),begin=p+30+n+ext;
 assert(!entries.has(name),'No duplicate ZIP entries');
 const value=raw.subarray(begin,begin+count);
 assert.equal(value.length,count);
 assert.equal(Archive.crc32(value),crc,'ZIP CRC must equal genuine raw payload');
 entries.set(name,value);
 p=begin+count;
}
assert(entries.has('manifest.json')&&entries.has('record.json')&&entries.has('naf.json'));
const record=JSON.parse(entries.get('record.json')),naf=JSON.parse(entries.get('naf.json'));
const manifest=JSON.parse(entries.get('manifest.json'));
assert.equal(manifest.schema,'scept.full-observation.v1');
assert.equal(record.runId,naf.runId);
assert.equal(record.runId,manifest.runId);
assert.equal(manifest.sourceRecordSchema,'scept.python-capture.v1');
assert.equal(entries.size,manifest.files.length+1);
assert.equal(manifest.captureScope.modelWeightsIncluded,false);
assert.equal(manifest.captureScope.originalNiftiSourceIncluded,false);
const outputs=new Map(record.outputs.map(x=>[x.name,x]));
let nOutputs=0;
for(const e of manifest.files){
 assert(entries.has(e.path),'All manifest file paths present');
 const data=entries.get(e.path);
 assert.equal(data.length,e.bytes);
 assert.equal(crypto.createHash('sha256').update(data).digest('hex'),e.sha256);
 if(e.kind==='observed_output_tensor'){
  nOutputs++;
  const meta=outputs.get(e.tensorName);
  assert(meta,'Name of true output exists in source record');
  assert.equal(e.tensorSha256,meta.sha256);
  assert.deepEqual(e.shape,meta.shape);
  const head=Buffer.from(JSON.stringify({shape:e.shape,type:e.dtype}));
  const hash=crypto.createHash('sha256').update(head).update(Buffer.from([0])).update(data).digest('hex');
  assert.equal(hash,meta.sha256,'Canonical browser and Python tensor bytes agree');
 }
}
assert.equal(nOutputs,record.outputs.length);
console.log('PASS A5 JavaScript verifies Python ZIP/STORE, all tensors, provenance and CRC/SHA ('+nOutputs+' outputs)');
