'use strict';
/* Read-only ONNX protobuf inspection. Parses only structural fields;
   never executes graph nodes or claims an edge is a causal explanation.
   Carefully skips potentially huge initializer binary payloads. */
(function(){
function inspect(arrayBuffer){
 const bytes=new Uint8Array(arrayBuffer),dv=new DataView(arrayBuffer);
 if(bytes.length>500000000)throw Error('Graph exceeds 500 MB inspector safeguard');
 function varint(pos,end){
  let value=0,shift=0;
  for(let count=0;count<10&&pos<end;count++){
   const v=bytes[pos++];if(shift<=49)value+=(v&127)*Math.pow(2,shift);
   if(!(v&128))return[value,pos];
   shift+=7;
  }throw Error('Malformed protobuf varint');
 }
 function entries(start,end,visit,depth=0){
  if(depth>8)throw Error('Excessive graph nesting');
  let pos=start,n=0;
  while(pos<end){
   if(++n>300000)throw Error('Graph field limit exceeded');
   const [tag,next]=varint(pos,end);pos=next;
   const field=Math.floor(tag/8),wire=tag%8;
   if(wire===0){const [v,p]=varint(pos,end);pos=p;visit(field,wire,v,undefined);}
   else if(wire===1){if(pos+8>end)throw Error('Bad protobuf fixed64');pos+=8;}
   else if(wire===2){const[len,p]=varint(pos,end);if(!Number.isSafeInteger(len)||len<0||p+len>end)throw Error('Bad protobuf length');pos=p;visit(field,wire,[pos,pos+len],undefined);pos+=len;}
   else if(wire===5){if(pos+4>end)throw Error('Bad protobuf fixed32');pos+=4;}
   else throw Error('Unsupported protobuf wire type '+wire);
  }
 }
 const utf8=new TextDecoder('utf-8',{fatal:false});
 const str=range=>utf8.decode(bytes.subarray(range[0],range[1]));
 function decodeNode(span){
  let name='',op='',inputs=[],outputs=[];
  entries(...span,(f,w,v)=>{
   if(w!==2)return;
   if(f===1)inputs.push(str(v));
   if(f===2)outputs.push(str(v));
   if(f===3)name=str(v);
   if(f===4)op=str(v);
  });
  return{name,op,inputs,outputs};
 }

 function valueInfo(span){
  let name='',tt=null,shape=[];
  entries(...span,(f,w,v)=>{
   if(f===1&&w===2)name=str(v);
   if(f===2&&w===2){entries(...v,(tf,tw,tv)=>{
    if(tf!==1||tw!==2)return;
    entries(...tv,(ff,fw,fv)=>{
     if(ff===1&&fw===0)tt=fv;
     if(ff===2&&fw===2)entries(...fv,(sf,sw,sv)=>{
      if(sf!==1||sw!==2)return;
      let dimension=null;
      entries(...sv,(df,dw,dv)=>{if(df===1&&dw===0)dimension=dv;if(df===2&&dw===2)dimension=str(dv)});
      shape.push(dimension??'?');
     });
    });
   });}
  });
  return{name,shape,type:({1:'float32',10:'float16',7:'int64',6:'int32',11:'float64'})[tt]||'unknown'};
 }
 let graph=null;
 entries(0,bytes.length,(f,w,v)=>{if(f===7&&w===2)graph=v;});
 if(!graph)throw Error('No ONNX graph found');
 const nodes=[],initializers=[],inputs=[],outputs=[],inputMetadata={},outputMetadata={},intermediateMetadata={};
 entries(...graph,(f,w,v)=>{
  if(w!==2)return;
  if(f===1){if(nodes.length<10000)nodes.push(decodeNode(v));}
  if(f===5){let name='';entries(...v,(sf,sw,sv)=>{if(sf===8&&sw===2)name=str(sv);});initializers.push(name);}
  if(f===11||f===12||f===13){const meta=valueInfo(v);if(f===13){if(Object.keys(intermediateMetadata).length<20000&&meta.name)intermediateMetadata[meta.name]=meta;}else{(f===11?inputs:outputs).push(meta.name);(f===11?inputMetadata:outputMetadata)[meta.name]=meta;}}
 });
 const producer=new Map(),edges=[];
 nodes.forEach((n,i)=>n.outputs.forEach(o=>producer.set(o,i)));
 nodes.forEach((n,i)=>n.inputs.forEach(inp=>{
  const p=producer.get(inp);if(p!==undefined&&p!==i)edges.push({from:p,to:i,tensor:inp});
 }));
 return {nodes,edges,initializers,inputs,outputs,inputMetadata,outputMetadata,intermediateMetadata,totalNodes:nodes.length,exceedsNodeLimit:nodes.length===10000};
}
window.SCEPT_INSPECT_ONNX=inspect;
})();