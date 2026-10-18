'use strict';
/* V6.3 bounded real pretrained CNN adapter, no hallucinated intermediate layers.
 * Model card explicitly reports train/test leakage; NOT medical diagnosis.
 */
(function(root){if(!root?.document)return;
 const SOURCE='https://huggingface.co/Udit013/brain-tumor-efficientnetb3/resolve/main/efficientnetb3.onnx',
 SHA='320b1ef648f3da5216f6ee0374349324b3cb00a76bebf7dbd1c3796e6155991c';
 let loaded=null,pending=null;
 const msg=(text)=>{const el=document.getElementById('v6Status');if(el)el.textContent=text};
 async function hash(bytes){
  return[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('')
 }
 function selectRealIntermediate(graph){
  const items=Object.entries(graph.intermediateMetadata||{}).filter(([name,m])=>
   m.type==='float32'&&m.shape?.length===4&&
   m.shape.every(d=>Number.isSafeInteger(d)&&d>0)&&
   m.shape.reduce((a,b)=>a*b,1)<=131072&&
   graph.nodes.some(n=>n.op==='Conv'&&n.outputs.includes(name)));
  // Order is the actual ONNX graph metadata. Prevent duplicate tensor picks.
  if(items.length<=3)return items.map(([name])=>name);
  return [items[0],items[Math.floor(items.length/2)],items[items.length-1]].map(([name])=>name);
 }
 async function load(){
  if(loaded)return loaded;
  if(pending)return pending;
  pending=(async()=>{
   if(!root.ort||!root.SCEPT_ONNX_PROBES||!root.SCEPT_INSPECT_ONNX)throw Error('CNN native adapters not loaded');
   msg('下载并校验真实 CNN 预训练模型（约46MB）…');
   const response=await fetch(SOURCE,{mode:'cors',cache:'force-cache'});
   if(!response.ok)throw Error('MRI CNN model HTTP '+response.status);
   const original=await response.arrayBuffer();
   if(await hash(original)!==SHA)throw Error('Real CNN pretrained model SHA mismatch');
   const inspected=root.SCEPT_INSPECT_ONNX(original),annotated=selectRealIntermediate(inspected);
   let selected=annotated,modelBytes,profile='cnn-nchw',promotionMode='annotated';
   if(selected.length){
    const p=root.SCEPT_ONNX_PROBES.promote(original,selected);
    if(p.missing.length)throw Error('Real CNN hidden probe promotion failed');
    modelBytes=p.model;
   }else{
    // ONNX exporter can omit every intermediate value_info. Only declare a
    // new graph output referencing a genuine pre-existing Conv node output.
    const convs=inspected.nodes.filter(n=>n.op==='Conv'&&n.outputs.length);
    if(!convs.length)throw Error('Pretrained CNN graph contains no real Conv producers');
    const positions=[Math.max(0,convs.length-3),Math.max(0,convs.length-2),convs.length-1];
    selected=[...new Set(positions.map(i=>convs[i].outputs[0]))].slice(0,3);
    const promoted=root.SCEPT_V6_GRAPH_PROBES.promote(original,
      selected.map(name=>({name,dtype:'float32',shapeRank:4})));
    modelBytes=promoted.model;promotionMode='actual-node-output';
   }
   const byte=modelBytes.buffer.slice(modelBytes.byteOffset,modelBytes.byteOffset+modelBytes.byteLength);
   const graph=root.SCEPT_INSPECT_ONNX(byte);
   const modelSHA=await hash(byte);
   msg('真实 CNN 图结构已核验，初始化 CPU/WASM 推理…');
   const session=await root.ort.InferenceSession.create(modelBytes,{executionProviders:['wasm']});
   const metas=[{name:'dense_2',type:'float32',shape:[1,4]},
    ...selected.map(name=>({name,type:'float32',shape:(graph.outputMetadata[name]?.shape||[]).map(d=>Number.isSafeInteger(d)&&d>0?d:null)}))];
   for(const m of metas)if(!session.outputNames.includes(m.name)||!m.shape.length||
     !m.shape.every(d=>d===null||Number.isSafeInteger(d)&&d>0))
     throw Error('Declared CNN output/probe not verified '+m.name);
   // ONNX Conv native activation layout is NCHW, even if input is NHWC.
   profile='cnn-nchw';
   loaded={sourceSha256:SHA,sha256:modelSHA,profile,selected,graph,promotionMode,
    description:{inputs:[{name:'input',type:'float32',shape:[1,256,256,3]}],outputs:metas},
    run:async feeds=>{
     const t=feeds.input;
     const out=await session.run({input:new root.ort.Tensor(t.type,t.data,t.dims)});
     const result={};for(const m of metas){const v=out[m.name];
      if(!v)throw Error('Missing genuine CNN tensor '+m.name);
      result[m.name]={type:v.type,dims:[...v.dims],data:v.data};
     }return result;
    }};
   return loaded;
  })();
  try{return await pending}finally{pending=null}
 }
 root.SCEPT_V6_CNN={load,getSnapshot:()=>({loaded:!!loaded,realIntermediateProbes:loaded?.selected.length||0,
  modelSha256:loaded?.sha256||null,promotionMode:loaded?.promotionMode||null})};
})(typeof window==='undefined'?null:window);
