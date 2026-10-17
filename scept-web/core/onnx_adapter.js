'use strict';
/** SCEPT U2 ONNX/WASM model adapter.
 * Supports ANY compatible float32 ONNX with declared dimensions and
 * independently selectable real graph.output/value_info tensors.
 * No preprocessing guesses, gradients, learned visualization, or retraining.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_ONNX_ADAPTER=api;
})(typeof window!=='undefined'?window:null,function(root){
 function issue(msg){throw Error('SCEPT ONNX adapter: '+msg)}
 const bytesOf=x=>x instanceof Uint8Array?x:new Uint8Array(x);
 function make(options={}){
  const ort=options.ort||root?.ort,inspect=options.inspect||root?.SCEPT_INSPECT_ONNX;
  const instrument=options.probes||root?.SCEPT_ONNX_PROBES;
  if(!ort||!inspect||!instrument)issue('ORT, ONNX inspector and real graph instrumentation required');
  function plan(original,context){
   const graph=inspect(bytesOf(original).slice().buffer),wanted=context.manifest?.probes||context.options?.probes||[];
   if(!Array.isArray(wanted))issue('probes must be an array of graph value_info names');
   const requested=[...new Set(wanted)];
   const known=new Set([...graph.outputs,...Object.keys(graph.intermediateMetadata||{})]);
   if(requested.some(name=>!known.has(name)))issue('Probe not an ONNX annotated intermediate/output: '+requested.find(n=>!known.has(n)));
   const promoted=instrument.promote(original,requested);
   if(promoted.missing.length)issue('Requested probe not exportable: '+promoted.missing.join(', '));
   const updated=inspect(promoted.model.buffer.slice(promoted.model.byteOffset,promoted.model.byteOffset+promoted.model.byteLength));
   function meta(name,collection){
    const md=collection[name];if(!md)issue('Missing ONNX tensor metadata '+name);
    if(!['float32','float64','int32','int64','uint8','bool'].includes(md.type))
      issue('Unsupported dtype '+md.type+' for '+name);
    return{name,type:md.type,shape:md.shape.slice()};
   }
   return {graph:updated,model:promoted.model,inputs:updated.inputs.map(x=>meta(x,updated.inputMetadata)),
    outputs:updated.outputs.map(x=>meta(x,updated.outputMetadata)),exposed:promoted.exposed,requested};
  }
  return Object.freeze({
   kind:'model',id:'onnx-wasm',version:'1.0.0-u2',
   capabilities:['inference','graph.read','observations.output','observations.intermediate'],
   async inspect(modelBytes,context={}){
    const p=plan(modelBytes,context);
    return{format:'onnx',runtime:'onnxruntime-web-wasm',inputs:p.inputs,outputs:p.outputs,
     graph:{nodes:p.graph.totalNodes,edges:p.graph.edges.length,operators:p.graph.nodes,dependencies:p.graph.edges},
     observableNames:p.exposed.slice(),annotatedIntermediateNames:Object.keys(p.graph.intermediateMetadata||{}),
     capabilities:p.exposed.length?this.capabilities.slice():this.capabilities.filter(x=>x!=='observations.intermediate')};
   },
   async load(modelBytes,context={}){
    const p=plan(modelBytes,context);
    if(!p.inputs.length||!p.outputs.length)issue('Model must have explicit inputs and outputs');
    const lib=context.options?.ort||ort;
    if(lib.env?.wasm){lib.env.wasm.numThreads=1;if(context.options?.wasmPaths)lib.env.wasm.wasmPaths=context.options.wasmPaths}
    let session;
    try{session=await lib.InferenceSession.create(p.model,{executionProviders:['wasm']})}
    catch(e){issue('WASM session failed to load model: '+(e?.message||e))}
    let closed=false;
    return{
     async run(feeds,request={}){
      if(closed)issue('Session closed');
      const actual={};
      for(const meta of p.inputs){
       const data=feeds[meta.name];if(!data)issue('Missing input '+meta.name);
       actual[meta.name]=new lib.Tensor(data.type,data.data,data.dims);
      }
      let out;
      try{out=await session.run(actual)}catch(e){issue('Real ONNX forward pass failed: '+(e?.message||e))}
      const requested=request.outputs||p.outputs.map(x=>x.name),result={};
      for(const name of requested){
       const v=out[name];if(!v)issue('Real inference did not return requested output '+name);
       // Binary tensor remains untouched; no fake intermediate values.
       result[name]={type:v.type,dims:Array.from(v.dims),data:v.data};
      }
      return result;
     },
     async dispose(){if(!closed){closed=true;await session.release()}},
     isClosed:()=>closed
    };
   }
  });
 }
 return Object.freeze({make});
});
