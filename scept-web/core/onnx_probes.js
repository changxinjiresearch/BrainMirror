'use strict';
/* U2: byte-safe ONNX GraphProto graph.output promotion.
   Only *existing annotated* value_info names may become extra graph outputs.
   No synthetic nodes, fake hidden tensors, or class-specific node names. */
(function(root,factory){
 const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_ONNX_PROBES=api;
})(typeof window!=='undefined'?window:null,function(){
 const MAX_BYTES=160*1024*1024,MAX_PROBES=16;
 function err(s){throw Error('SCEPT ONNX probe: '+s)}
 function readVar(b,p,end){
  let n=0,factor=1;for(let i=0;i<9&&p<end;i++){
   const c=b[p++];n+=(c&127)*factor;if(!Number.isSafeInteger(n))err('unsafe varint');
   if(!(c&128))return{value:n,end:p};factor*=128;
  }err('truncated varint');
 }
 function scan(b,start,end,visit){
  let p=start;
  while(p<end){
   const mark=p,key=readVar(b,p,end);p=key.end;
   const field=Math.floor(key.value/8),wire=key.value%8;
   if(wire===2){const len=readVar(b,p,end);p=len.end;const s=p,t=s+len.value;
    if(!Number.isSafeInteger(t)||t>end)err('invalid field length');
    visit({field,wire,start:mark,s,e:t,stop:t});p=t;
   }else if(wire===0){p=readVar(b,p,end).end}
   else if(wire===1){p+=8}
   else if(wire===5){p+=4}
   else err('unsupported protobuf wire type');
   if(p>end)err('field beyond graph bounds');
  }
 }
 function encodeVar(n){if(!Number.isSafeInteger(n)||n<0)err('invalid varint encoding');const out=[];
  do{let c=n%128;n=Math.floor(n/128);if(n)c|=128;out.push(c)}while(n);return Uint8Array.from(out);
 }
 function combine(parts){const length=parts.reduce((s,p)=>s+p.length,0);
  if(length>MAX_BYTES+1024*1024)err('probe size limit');
  const out=new Uint8Array(length);let at=0;for(const p of parts){out.set(p,at);at+=p.length}return out;
 }
 function promote(source,names){
  const b=source instanceof Uint8Array?source:new Uint8Array(source);
  if(b.length>MAX_BYTES)err('model exceeds local instrumentation limit');
  if(!Array.isArray(names)||names.length>MAX_PROBES||new Set(names).size!==names.length||
    names.some(n=>typeof n!=='string'||!n||n.length>512))err('invalid probe request');
  if(names.length===0)return{model:b,exposed:[],missing:[]};
  let graph=null;
  scan(b,0,b.length,x=>{if(x.field===7&&x.wire===2)graph=x});
  if(!graph)err('model lacks GraphProto');
  const decoder=new TextDecoder(),valueInfo=new Map(),existing=new Set();
  scan(b,graph.s,graph.e,x=>{
   if(x.field!==12&&x.field!==13)return;
   let name='';
   scan(b,x.s,x.e,f=>{if(f.field===1&&f.wire===2)name=decoder.decode(b.subarray(f.s,f.e))});
   if(name){
    if(x.field===12)existing.add(name);
    else valueInfo.set(name,b.subarray(x.s,x.e));
   }
  });
  const exposed=[],missing=[],parts=[b.subarray(graph.s,graph.e)];
  for(const name of names){
   if(existing.has(name)){exposed.push(name);continue}
   const record=valueInfo.get(name);
   if(!record){missing.push(name);continue}
   parts.push(encodeVar(98),encodeVar(record.length),record);exposed.push(name);
  }
  if(parts.length===1)return{model:b,exposed,missing};
  const newGraph=combine(parts);
  const out=combine([b.subarray(0,graph.start),encodeVar(58),encodeVar(newGraph.length),newGraph,b.subarray(graph.stop)]);
  return{model:out,exposed,missing};
 }
 return Object.freeze({promote,MAX_PROBES});
});
