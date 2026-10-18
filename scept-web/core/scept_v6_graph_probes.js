'use strict';
/* V6.3: promote ACTUAL existing ONNX Conv node output to a runtime graph output
 * even when exporter omitted value_info. FLOAT32 and rank 4 are DECLARATIONS
 * verified by ONNX Runtime: no invented feature values or synthetic nodes.
 * Raw protobuf field edits, fail closed if source graph lacks a real producer.
 */
(function(root,factory){const a=factory();if(typeof module==='object'&&module.exports)module.exports=a;if(root)root.SCEPT_V6_GRAPH_PROBES=a})(typeof window!=='undefined'?window:null,function(){
 const error=m=>{throw Error('SCEPT V6 graph probe: '+m)};
 const MAX=150*1024*1024,enc=new TextEncoder(),dec=new TextDecoder();
 const varint=n=>{if(!Number.isSafeInteger(n)||n<0)error('bad integer encoding');
  const v=[];do{let b=n&127;n=Math.floor(n/128);if(n)b|=128;v.push(b)}while(n);return Uint8Array.from(v)};
 const parts=chunks=>{const size=chunks.reduce((s,a)=>s+a.length,0);
  if(size>MAX+1024*1024)error('bounded patch size');
  const result=new Uint8Array(size);let at=0;
  for(const c of chunks){result.set(c,at);at+=c.length}return result};
 const tag=(n,w)=>varint(n*8+w),field=(n,v)=>parts([tag(n,2),varint(v.length),v]),
  stringField=(n,s)=>field(n,enc.encode(s)),
  scalar=(n,x)=>parts([tag(n,0),varint(x)]);
 function readNum(b,p,end){
  let number=0,m=1;for(let i=0;i<9&&p<end;i++){
   const v=b[p++];number+=(v&127)*m;
   if(!Number.isSafeInteger(number))error('unsafe varint');
   if(!(v&128))return{number,p};m*=128;
  }error('truncated varint');
 }
 function visit(b,start,end,fn){
  let p=start;while(p<end){
   const mark=p,t=readNum(b,p,end);p=t.p;
   const fieldNum=Math.floor(t.number/8),wire=t.number%8;
   if(wire===2){const len=readNum(b,p,end);p=len.p;
    const s=p,e=s+len.number;if(!Number.isSafeInteger(e)||e>end)error('invalid protobuf length');
    fn({field:fieldNum,s,e,mark,stop:e});p=e;
   }else if(wire===0)p=readNum(b,p,end).p;
   else if(wire===1)p+=8;else if(wire===5)p+=4;
   else error('unsupported encoding');
   if(p>end)error('malformed protobuf');
  }
 }
 function tensorInfo(name,rank){
  // ValueInfoProto {name;type:TypeProto{tensor_type:{elem_type:FLOAT,
  // shape:{dim:{dim_param:...},...}}}}. Shape symbols are NOT data.
  const dims=[];for(let i=0;i<rank;i++)dims.push(field(1,stringField(2,'scept_dynamic_dim_'+i)));
  const shape=parts(dims),tensorType=parts([scalar(1,1),field(2,shape)]),
    type=field(1,tensorType);
  return parts([stringField(1,name),field(2,type)]);
 }
 function promote(source,requests){
  const b=source instanceof Uint8Array?source:new Uint8Array(source);
  if(b.length>MAX||!Array.isArray(requests)||requests.length<1||requests.length>3)
   error('source size / 1..3 probe limit');
  if(requests.some(r=>typeof r?.name!=='string'||!r.name||r.shapeRank!==4||r.dtype!=='float32')||
   new Set(requests.map(r=>r.name)).size!==requests.length)error('strict float32 4D unique probe descriptors required');
  let graph=null;visit(b,0,b.length,x=>{if(x.field===7)graph=x});if(!graph)error('No GraphProto');
  const actual=new Map(),existing=new Set();
  visit(b,graph.s,graph.e,x=>{
   if(x.field===1){
    let op='',out=[];
    visit(b,x.s,x.e,v=>{
     if(v.field===4)op=dec.decode(b.subarray(v.s,v.e));
     if(v.field===2)out.push(dec.decode(b.subarray(v.s,v.e)));
    });
    for(const name of out)actual.set(name,op);
   }
   if(x.field===12){
    visit(b,x.s,x.e,v=>{if(v.field===1)existing.add(dec.decode(b.subarray(v.s,v.e)))});
   }
  });
  for(const r of requests)if(actual.get(r.name)!=='Conv'||existing.has(r.name))
   error('Requested output is not a real unexposed ONNX Conv producer: '+r.name);
  const append=requests.map(r=>field(12,tensorInfo(r.name,r.shapeRank)));
  const newer=parts([b.subarray(graph.s,graph.e),...append]);
  return{model:parts([b.subarray(0,graph.mark),field(7,newer),b.subarray(graph.stop)]),
   provenance:requests.map(r=>({name:r.name,producer:'real-onnx-Conv',
    declared:'float32 / dynamic rank4',numericalValidation:'must execute actual ONNX model'}))};
 }
 return Object.freeze({promote,VERSION:'6.3.1'});
});
