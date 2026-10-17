'use strict';
/* SCEPT U5: deterministic, fail-closed, plugin-defined intervention engine.
 * No model parameters are changed. No anatomical/clinical validity implied.
 * Same interface for CNN / ViT / other compatible frozen ONNX graphs. */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_INTERVENTION=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0-u5';
 const ERROR=(code,message)=>{const e=Error('SCEPT U5: '+message);e.code=code;throw e};
 const plain=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const finite=(x,label)=>{if(typeof x!=='number'||!Number.isFinite(x))ERROR('INVALID_PARAMETER',label+' must be finite');return x};
 const integer=(x,label)=>{if(!Number.isSafeInteger(x))ERROR('INVALID_PARAMETER',label+' must be an integer');return x};
 const own=(o,k)=>Object.prototype.hasOwnProperty.call(o,k);
 const clone=t=>({type:t.type,dims:t.dims.slice(),data:new t.data.constructor(t.data)});
 function copyFeeds(feeds){const next={};for(const [k,t]of Object.entries(feeds))next[k]=clone(t);return next}
 function nameOf(desc,config){const name=config.inputName||desc.inputs[0]?.name;
   if(!desc.inputs.some(x=>x.name===name))ERROR('INPUT_NOT_FOUND','Unknown model input '+name);return name}
 function tensorOf(feeds,desc,config){
   const name=nameOf(desc,config),t=feeds[name];
   if(!t)ERROR('INPUT_NOT_FOUND','No supplied input '+name);
   if(t.type!=='float32'&&t.type!=='float64')ERROR('UNSUPPORTED_DTYPE','Intervention supports only explicit floating point tensors: '+name);
   return{name,t};
 }
 function regionIndices(dims,region){
  if(!plain(region)||!Array.isArray(region.bounds)||region.bounds.length!==dims.length)
   ERROR('INVALID_REGION','ROI requires bounds [start,end] for every actual tensor axis');
  const bounds=region.bounds.map((r,i)=>{
   if(!Array.isArray(r)||r.length!==2)ERROR('INVALID_REGION','Bounds must be [start,end)');
   const [start,end]=r;integer(start,'region start');integer(end,'region end');
   if(start<0||end>dims[i]||end<=start)ERROR('INVALID_REGION','Region outside tensor dimension '+i);
   return [start,end];
  });
  const count=bounds.reduce((n,[a,b])=>n*(b-a),1);
  if(!Number.isSafeInteger(count)||count>2e6)ERROR('RESOURCE_LIMIT','More than 2 million selected elements');
  const strides=[];let stride=1;
  for(let i=dims.length-1;i>=0;i--){strides[i]=stride;stride*=dims[i]}
  const indices=new Array(count);
  for(let j=0;j<count;j++){
   let left=j,offset=0;
   for(let axis=dims.length-1;axis>=0;axis--){
    const width=bounds[axis][1]-bounds[axis][0],local=left%width;
    left=Math.floor(left/width);offset+=(bounds[axis][0]+local)*strides[axis];
   }
   indices[j]=offset;
  }
  return{indices,bounds,count};
 }
 function sampler(seed){
  integer(seed,'random seed');if(seed<0||seed>0xffffffff)ERROR('INVALID_PARAMETER','Seed must be uint32');
  let x=seed>>>0||0x6d2b79f5;
  return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296};
 }
 function normal(random){const u=Math.max(random(),1e-12),v=random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
 function changes(edited,original,indices,makeValue){
  let changed=0;for(const i of indices){
   const value=makeValue(original.data[i],i);
   finite(value,'intervened value');
   edited.data[i]=value;
   if(!Number.isFinite(edited.data[i]))ERROR('NONFINITE_EDIT','Edited value overflowed target dtype');
   if(edited.data[i]!==original.data[i])changed++;
  }return changed;
 }
 function fullOrRegion(t,region){
  return region?regionIndices(t.dims,region):(()=>{
   if(t.data.length>2e6)ERROR('RESOURCE_LIMIT','Whole-array operation capped at 2 million values; select an ROI');
   return{indices:Array.from({length:t.data.length},(_,i)=>i),bounds:null,count:t.data.length};
  })();
 }
 function requireControl(feeds,reference,description,dataAdapter){
  if(!reference)ERROR('REFERENCE_REQUIRED','Region replacement requires an explicit reference tensor input');
  return dataAdapter.prepare(description,reference);
 }
 function checkShapeSignature(a,b){return a.type===b.type&&a.dims.length===b.dims.length&&a.dims.every((d,i)=>d===b.dims[i])}
 function registerBuiltins(registry){
  const adapters=[
   {id:'u5.noise',kind:'intervention',capabilities:['input.intervention'],apply(ctx){
     const {description,baseline,config}=ctx,{name,t}=tensorOf(baseline,description,config);
     const sigma=finite(config.sigma??.1,'sigma'),seed=config.seed??1;
     if(sigma<0||sigma>1e6)ERROR('INVALID_PARAMETER','sigma must be between 0 and 1e6');
     const scope=fullOrRegion(t,config.region),random=sampler(seed),edited=copyFeeds(baseline);
     const changed=changes(edited[name],t,scope.indices,a=>a+sigma*normal(random));
     return{baseline,edited,metadata:{kind:'noise',inputName:name,distribution:'gaussian-box-muller',
      seed,sigma,region:scope.bounds,selected:scope.count,changed}};
   }},
   {id:'u5.occlusion',kind:'intervention',capabilities:['input.intervention'],apply(ctx){
     const {description,baseline,config}=ctx,{name,t}=tensorOf(baseline,description,config);
     const scope=regionIndices(t.dims,config.region),fill=finite(config.fill??0,'occlusion fill'),edited=copyFeeds(baseline);
     const changed=changes(edited[name],t,scope.indices,()=>fill);
     return{baseline,edited,metadata:{kind:'occlusion',inputName:name,fill,region:scope.bounds,selected:scope.count,changed}};
   }},
   {id:'u5.replace',kind:'intervention',capabilities:['input.intervention'],apply(ctx){
     const {description,baseline,config}=ctx,{name,t}=tensorOf(baseline,description,config);
     const reference=requireControl(baseline,ctx.referenceFeeds,description,ctx.dataAdapter);
     if(!checkShapeSignature(t,reference[name]))ERROR('SHAPE_MISMATCH','Replacement tensor must exactly match source shape/type');
     const scope=regionIndices(t.dims,config.region),edited=copyFeeds(baseline);
     const changed=changes(edited[name],t,scope.indices,(_,i)=>reference[name].data[i]);
     return{baseline,edited,metadata:{kind:'region-replacement',inputName:name,source:'explicit-reference-feeds',
      region:scope.bounds,selected:scope.count,changed}};
   }},
   {id:'u5.matched-control',kind:'intervention',capabilities:['input.intervention'],apply(ctx){
     const {description,baseline,config}=ctx,{name,t}=tensorOf(baseline,description,config);
     const main=regionIndices(t.dims,config.region),control=regionIndices(t.dims,config.controlRegion);
     if(main.count!==control.count)ERROR('MATCH_FAILED','Control must have exactly the same number of elements');
     const occupied=new Set(main.indices);
     if(control.indices.some(i=>occupied.has(i)))ERROR('MATCH_FAILED','Control and primary region must not overlap');
     const fill=finite(config.fill??0,'fill');
     // "Matched" is earned by an explicit, reproducible numeric covariate check.
     const tolerance=finite(config.meanTolerance,'meanTolerance');
     if(tolerance<0)ERROR('INVALID_PARAMETER','meanTolerance must be nonnegative');
     const mean=indices=>indices.reduce((s,i)=>s+Number(t.data[i]),0)/indices.length;
     const meanOriginal=mean(main.indices),meanControl=mean(control.indices),difference=Math.abs(meanOriginal-meanControl);
     if(difference>tolerance)ERROR('MATCH_FAILED','Control baseline mean differs by '+difference+' > '+tolerance);
     const edited=copyFeeds(baseline),controlFeeds=copyFeeds(baseline);
     const changed=changes(edited[name],t,main.indices,()=>fill);
     const controlChanged=changes(controlFeeds[name],t,control.indices,()=>fill);
     return{baseline,edited,control:controlFeeds,metadata:{kind:'matched-control-occlusion',inputName:name,fill,
      primaryRegion:main.bounds,controlRegion:control.bounds,primaryCount:main.count,controlCount:control.count,selected:main.count,
      meanTolerance:tolerance,baselineMeanPrimary:meanOriginal,baselineMeanControl:meanControl,
      baselineMeanDifference:difference,changed,controlChanged,
      matching:'equal element count, non-overlapping regions and pre-intervention mean within tolerance only'}};
   }},
   {id:'u5.internal-mask',kind:'intervention',capabilities:['hidden.intervention'],apply(ctx){
     const {description,baseline,config}=ctx,maskInput=config.maskInput||'scept_mask';
     const declared=(description.internalMasks||[]).find(x=>x.input===maskInput);
     if(!description.capabilities?.includes('hidden.intervention')||!declared)
      ERROR('UNSUPPORTED_CAPABILITY','Model has no verified in-graph activation mask; export an instrumented ONNX graph');
     const t=baseline[maskInput];
     if(!t||t.type!=='float32'||!checkShapeSignature(t,{type:'float32',dims:declared.shape}))
      ERROR('MASK_SIGNATURE_MISMATCH','Internal mask input must match verified target shape');
     if(!Array.isArray(config.indices)||!config.indices.length||config.indices.length>2e6)
      ERROR('INVALID_PARAMETER','Explicit nonempty mask indices required (<=2 million)');
     const indices=new Set();
     for(const i of config.indices){integer(i,'mask index');if(i<0||i>=t.data.length)ERROR('INVALID_PARAMETER','Mask index out of bounds');indices.add(i)}
     const value=finite(config.value??0,'mask value');
     if(value<0||value>1)ERROR('INVALID_PARAMETER','Mask multiplier must be within [0,1]');
     const normalized=copyFeeds(baseline),edited=copyFeeds(baseline);
     normalized[maskInput].data.fill(1);edited[maskInput].data.fill(1);
     for(const i of indices)edited[maskInput].data[i]=value;
     return{baseline:normalized,edited,metadata:{kind:'instrumented-internal-state-mask',
      inputName:maskInput,targetTensor:declared.target,graphNode:declared.node,
      multiplier:value,selected:indices.size,changed:value===1?0:indices.size,
      indices:[...indices].sort((a,b)=>a-b),source:'verified-existing-Mul-node-in-the-actual-ONNX-graph',
      baseline:'all-one mask',weightsModified:false}};
   }}
  ];
  adapters.forEach(x=>registry.register('intervention',x));
  return adapters.map(x=>({id:x.id,capabilities:x.capabilities.slice()}));
 }
 function make(registry,core,dataAdapter){
  if(!registry||!core?.validateTensor||!dataAdapter?.prepare)ERROR('INVALID_ENGINE','Registry/Core/U3 data adapter required');
  async function execute({pluginId,description,feeds,config={},referenceFeeds=null}){
   if(!plain(config))ERROR('INVALID_PARAMETER','Plugin parameters must be an object');
   const plugin=registry.requireAdapter('intervention',pluginId);
   const cap=pluginId==='u5.internal-mask'?'hidden.intervention':'input.intervention';
   registry.requireCapability(plugin,cap);
   if(typeof plugin.apply!=='function')ERROR('INVALID_PLUGIN','Plugin requires apply(context)');
   const baseline=dataAdapter.prepare(description,feeds);
   const untouched=dataAdapter.prepare(description,baseline);
   // Give the plugin its own copy. Never expose the caller's source arrays.
   const outcome=await plugin.apply({description,baseline:dataAdapter.prepare(description,baseline),config,referenceFeeds,dataAdapter});
   if(!plain(outcome)||!outcome.baseline||!outcome.edited||!plain(outcome.metadata))
    ERROR('INVALID_PLUGIN','Plugin must return {baseline,edited,metadata}');
   const primary=dataAdapter.prepare(description,outcome.baseline);
   const edited=dataAdapter.prepare(description,outcome.edited);
   const control=outcome.control?dataAdapter.prepare(description,outcome.control):null;
   // Only a verified, graph-instrumented mask plugin can legally alter the baseline
   // mask feed to all-ones; every other plugin must preserve baseline elementwise.
   if(pluginId!=='u5.internal-mask')for(const name of Object.keys(untouched)){
    const a=untouched[name].data,b=primary[name].data;
    if(a.length!==b.length||a.some((v,i)=>v!==b[i]))ERROR('BASELINE_MODIFIED','Intervention plugin changed baseline input');
   }
   if(!plain(outcome.metadata)||outcome.metadata.selected===0)ERROR('INVALID_PLUGIN','No selected region');
   return{baseline:primary,edited,control,intervention:{
     schema:'scept.u5.intervention.v1',plugin:pluginId,capability:cap,
     ...outcome.metadata,pairedInputsUnchangedExceptTarget:true,
     scientificClaim:'numeric computational intervention only; no automatic medical/causal interpretation'}};
  }
  return Object.freeze({VERSION,registerBuiltins:()=>registerBuiltins(registry),execute,
   available:()=>registry.list('intervention'),regionIndices});
 }
 return Object.freeze({VERSION,make,regionIndices});
});