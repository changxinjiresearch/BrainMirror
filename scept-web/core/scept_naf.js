'use strict';
/* V6.2 true one-run NAF: State, same-forward certified Update, native channel
 * Pattern. No A/B, synthetic activity, saliency or patient-level diagnosis.
 */
(function(root,factory){const a=factory();if(typeof module==='object'&&module.exports)module.exports=a;if(root)root.SCEPT_NAF=a})(typeof window!=='undefined'?window:null,function(){
 const VERSION='6.2.0',SCHEMA='scept.naf.v1';
 const issue=(code,m)=>{const e=Error('SCEPT NAF: '+m);e.code=code;throw e};
 function assertCapture(cap){
  if(!cap||cap.record?.schema!=='scept.run-record.v1'||typeof cap.getTensor!=='function')issue('INVALID_CAPTURE','V6.1 actual capture required');
 }
 function units(record,profile,probe){
  const obs=record.observations.find(v=>v.tensor===probe);
  if(!obs||obs.status!=='measured'||obs.activity?.source!=='actual-runtime-output')
   return {status:'not_observed',tensor:probe,reason:obs?.reason||'probe_not_measured',units:[]};
  const a=obs.activity,role=profile==='vit-token-channel'?'token':
    profile==='cnn-nchw'||profile==='cnn-nhwc'?'channel':null;
  const assignedRole=role&&Array.isArray(a.axisSummaries[role])?role:
    (Array.isArray(a.axisSummaries.feature)?'feature':null);
  const rows=a.axisSummaries[assignedRole];
  if(!assignedRole||!Array.isArray(rows)||!rows.length) return{status:'unsupported',tensor:probe,
    reason:'no_native_units_for_profile_or_resource_cap',units:[]};
  return{status:'measured',tensor:probe,architecture:a.architecture,profile,role:assignedRole,
    shape:a.shape,units:rows.map(u=>({index:u.index,rms:u.rms,mean:u.mean,
     positiveFraction:u.positiveFraction,negativeFraction:u.negativeFraction,
     count:u.count})),scientificMeaning:'numeric internal state magnitude, not importance'};
 }
 function getPattern(cap,probe,index,profile,{maxValues=65536}={}){
  assertCapture(cap);
  const t=cap.getTensor(probe);if(!t)issue('UNOBSERVED','Raw tensor not retained: '+probe);
  if(t.dims.length===2&&t.dims[0]===1&&Number.isInteger(index)&&index>=0&&index<t.dims[1])
   return{schema:'scept.naf.pattern.v1',tensor:probe,profile,unit:index,shape:t.dims.slice(),
    coordinates:{feature:index},values:[Number(t.data[index])],
    provenance:'verified_same_forward_real_output_only',interpretation:'model output feature, NOT internal hidden unit'};
  if(!Number.isInteger(index)||index<0)issue('BAD_UNIT','Unit must be an integer');
  let stride,start,coordinates;
  if(profile==='vit-token-channel'){
   if(t.dims.length!==3||t.dims[0]!==1||index>=t.dims[1])issue('BAD_SHAPE','Expected [1,tokens,channels]');
   stride=t.dims[2];start=index*stride;coordinates={token:index,featureChannels:stride};
  }else if(profile==='cnn-nchw'){
   if(t.dims.length!==4||t.dims[0]!==1||index>=t.dims[1])issue('BAD_SHAPE','Expected [1,channel,h,w]');
   stride=t.dims[2]*t.dims[3];start=index*stride;coordinates={channel:index,height:t.dims[2],width:t.dims[3]};
  }else if(profile==='cnn-nhwc'){
   if(t.dims.length!==4||t.dims[0]!==1||index>=t.dims[3])issue('BAD_SHAPE','Expected [1,h,w,channel]');
   stride=t.dims[1]*t.dims[2];start=-1;coordinates={channel:index,height:t.dims[1],width:t.dims[2]};
  }else issue('UNSUPPORTED','Pattern profile '+profile+' unsupported');
  if(stride>maxValues)issue('RESOURCE_LIMIT','Unit pattern exceeds safe detailed render');
  const values=[];
  for(let j=0;j<stride;j++)values.push(Number(t.data[start<0?j*t.dims[3]+index:start+j]));
  if(!values.every(Number.isFinite))issue('NONFINITE','Raw activation invalid');
  return{schema:'scept.naf.pattern.v1',tensor:probe,profile,unit:index,shape:t.dims.slice(),
   coordinates,values,provenance:'verified_same_forward_private_runtime_tensor',
   interpretation:'raw channels / spatial feature values only; no semantic labels'};
 }
 function connected(graph,input,output){
  if(!Array.isArray(graph?.nodes)||!graph.nodes.length)return false;
  const consumers=new Map();
  for(const n of graph.nodes)for(const s of n.inputs||[]){
   if(!consumers.has(s))consumers.set(s,[]);
   consumers.get(s).push(n);
  }
  let frontier=[input],seen=new Set(),steps=0;
  while(frontier.length&&steps<50000){
   const cur=frontier.shift();steps++;
   if(cur===output)return true;
   if(seen.has(cur))continue;seen.add(cur);
   for(const n of consumers.get(cur)||[])for(const next of n.outputs||[])if(!seen.has(next))frontier.push(next);
  }
  return false;
 }
 function update(cap,{inputTensor,outputTensor,profile,graph,coordinateSemantics,declaredBy,verifiedIdentity}={}){
  assertCapture(cap);
  const not=(reason)=>({schema:'scept.naf.update.v1',inputTensor,outputTensor,status:'not_evaluable',reason,units:[]});
  if(!inputTensor||!outputTensor||!profile)return not('no_declared_module_input_output');
  if(coordinateSemantics!=='same-residual-stream'||verifiedIdentity!==true||
   typeof declaredBy!=='string'||!declaredBy.trim())
   return not('missing_verified_same_coordinate_semantics');
  const a=cap.getTensor(inputTensor),b=cap.getTensor(outputTensor);
  if(!a||!b)return not('input_or_output_not_observed');
  if(a.type!==b.type||a.type!=='float32'||a.dims.join()!==b.dims.join())
   return not('shape_or_dtype_not_identical');
  if(!connected(graph,inputTensor,outputTensor))return not('no_real_computation_path');
  const role=profile==='vit-token-channel'&&a.dims.length===3&&a.dims[0]===1?'token':null;
  if(!role)return not('update_axis_not_compatible');
  const stride=a.dims[2],units=[];
  for(let u=0;u<a.dims[1];u++){
   let ss=0,signed=0;
   for(let j=0;j<stride;j++){
    const d=Number(b.data[u*stride+j])-Number(a.data[u*stride+j]);
    ss+=d*d;signed+=d;
   }
   units.push({index:u,rmsUpdate:Math.sqrt(ss/stride),meanSignedUpdate:signed/stride});
  }
  return {schema:'scept.naf.update.v1',status:'measured',
   inputTensor,outputTensor,profile,role,units,declaredBy,
   proof:'same-coordinate-adapter-declaration-and-real_onnx_dependency_path',
   limitation:'representation update, not FLOPs, importance, or a causal path'};
 }
 function create(cap,{profile,probes,updates=[]}={}){
  assertCapture(cap);if(!Array.isArray(probes)||!probes.length||new Set(probes).size!==probes.length)issue('INVALID_PROBES','Real distinct probes required');
  const fingerprint={schema:SCHEMA,version:VERSION,runId:cap.record.runId,
    provenance:JSON.parse(JSON.stringify(cap.record.provenance)),
    profile,groups:probes.map(p=>units(cap.record,profile,p)),
    updates:updates.map(up=>update(cap,{...up,profile})),
    science:{status:'single_run_observation_only',importance:'NOT_ESTABLISHED',stability:'NOT_EVALUATED',
      causality:'NOT_ESTABLISHED',clinical:'NOT_ESTABLISHED'}};
  return fingerprint;
 }
 return Object.freeze({VERSION,SCHEMA,units,getPattern,update,create,connected});
});
