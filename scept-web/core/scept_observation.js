'use strict';
/* U4: observations use genuine runtime arrays, axes from explicit profiles.
   197-token/192-channel assumption exists ONLY in the legacy ViT profile. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.SCEPT_OBSERVATION=api;})(typeof window!=='undefined'?window:null,function(){
 const fail=(code,s)=>{const e=Error('SCEPT observation: '+s);e.code=code;throw e};
 const profiles=Object.freeze({
  generic:{architecture:'unknown',layouts:{}},
  'vit-token-channel':{architecture:'vision-transformer',layouts:{
   2:[['batch',0],['feature',1]],3:[['batch',0],['token',1],['channel',2]]}},
  'cnn-nchw':{architecture:'convolutional',layouts:{
   2:[['batch',0],['feature',1]],4:[['batch',0],['channel',1],['height',2],['width',3]]}},
  'cnn-nhwc':{architecture:'convolutional',layouts:{
   2:[['batch',0],['feature',1]],4:[['batch',0],['height',1],['width',2],['channel',3]]}},
  'legacy-vit-197x192':{architecture:'legacy-vit-only',layouts:{
   3:[['batch',0],['token',1],['channel',2]]},exact:[1,197,192]}
 });
 function selectProfile(id){if(!Object.hasOwn(profiles,id))fail('PROFILE_UNSUPPORTED','Explicit model profile not found: '+id);return profiles[id]}
 function check(t,m,name){
  if(!t||!Array.isArray(t.dims)||!t.dims.length||t.dims.length>8||!t.dims.every(x=>Number.isSafeInteger(x)&&x>0))fail('BAD_ACTIVATION','Invalid dimensions '+name);
  if(!['float32','float64','int32','uint8','bool'].includes(t.type))fail('BAD_ACTIVATION','Unsupported numerical observation dtype '+name);
  const n=t.dims.reduce((a,b)=>a*b,1);
  if(!Number.isSafeInteger(n)||n>1e8||!t.data||t.data.length!==n)fail('BAD_ACTIVATION','Runtime length mismatch '+name);
  if(m&&(t.type!==m.type||t.dims.length!==m.shape.length||
   t.dims.some((d,i)=>typeof m.shape[i]==='number'&&d!==m.shape[i])))fail('SIGNATURE_MISMATCH','Runtime tensor contradicts ONNX signature '+name);
  return n;
 }
 function observeTensor(name,a,b,meta,profileId,source,limit=4096){
  const profile=selectProfile(profileId),n=check(a,meta,name);
  if(check(b,meta,name)!==n||a.type!==b.type||a.dims.some((x,i)=>x!==b.dims[i]))fail('PAIRED_SHAPE_MISMATCH','A/B shape mismatch '+name);
  if(!Number.isInteger(limit)||limit<0||limit>16384)fail('INVALID_LIMIT','Axis output limit');
  const dims=a.dims;
  if(profile.exact&&dims.length===3&&profile.exact.some((d,i)=>dims[i]!==d))
   fail('LEGACY_LAYOUT_MISMATCH','197 tokens / 192 channels are permitted solely in exact legacy ViT adapter');
  const axes=(profile.layouts[dims.length]||[]).map(([role,index])=>({role,index,size:dims[index]}));
  const stride=[];let x=1;for(let i=dims.length-1;i>=0;i--){stride[i]=x;x*=dims[i]}
  const groups=axes.map(axis=>axis.size<=limit?Array.from({length:axis.size},()=>({sum:0,sq:0,changed:0,n:0})):null);
  let sum=0,sq=0,signed=0,max=0,changed=0;const first12=[];
  for(let i=0;i<n;i++){
   const va=Number(a.data[i]),vb=Number(b.data[i]);if(!Number.isFinite(va)||!Number.isFinite(vb))fail('NONFINITE_ACTIVATION','Nonfinite output '+name);
   const d=vb-va,ad=Math.abs(d);if(!Number.isFinite(d))fail('NONFINITE_ACTIVATION','Nonfinite numerical difference '+name);sum+=ad;sq+=d*d;signed+=d;max=Math.max(max,ad);
   if(ad>1e-7)changed++;if(i<12)first12.push({index:i,A:va,B:vb,delta:d});
   for(let j=0;j<axes.length;j++){const group=groups[j];if(!group)continue;
    const ax=axes[j],g=group[Math.floor(i/stride[ax.index])%ax.size];g.sum+=ad;g.sq+=d*d;g.n++;if(ad>1e-7)g.changed++;
   }
  }
  const axisSummaries={};axes.forEach((ax,j)=>{if(groups[j])axisSummaries[ax.role]=groups[j].map((g,index)=>({
   index,elements:g.n,changed:g.changed,mean_abs_delta:g.sum/g.n,rms_delta:Math.sqrt(g.sq/g.n)}))});
  return {tensor:name,type:a.type,shape:dims.slice(),source,profile:profileId,architecture:profile.architecture,
   axes,axisSummaries,total:n,changed,mean_abs_delta:sum/n,max_abs_delta:max,rms_delta:Math.sqrt(sq/n),
   mean_signed_delta:signed/n,first12,evidence:'actual-onnx-runtime-graph-outputs',
   caveat:'Numeric dependencies are not biological neurons, anatomical correspondences or scientific explanations'};
 }
 function observePair({description,baseline,edited,probes=[],profileId='generic',maxAxisUnits=4096}){
  selectProfile(profileId);
  if(!Array.isArray(description?.outputs)||!baseline||!edited||!Array.isArray(probes))fail('INVALID_CONTRACT','Pair/signature required');
  const names=description.outputs.map(m=>m.name);
  if(probes.some(name=>!names.includes(name)))fail('PROBE_NOT_EXPOSED','Probe not exported as genuine ONNX output');
  if(Object.keys(baseline).length!==names.length||Object.keys(edited).length!==names.length||names.some(n=>!Object.hasOwn(baseline,n)||!Object.hasOwn(edited,n)))
   fail('OUTPUT_MISMATCH','Inference pair does not match full declared output signature');
  return description.outputs.map(m=>observeTensor(m.name,baseline[m.name],edited[m.name],m,profileId,
    probes.includes(m.name)?'selected_real_intermediate':'actual_graph_output',maxAxisUnits));
 }
 return Object.freeze({VERSION:'1.0.0-u4',profiles,selectProfile,observeTensor,observePair});
});