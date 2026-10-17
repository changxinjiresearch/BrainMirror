'use strict';
/* U8 read-only research-visualization adapter.
 * Every numerical point is derived from U4 real runtime observations or actual
 * paired model feeds; layout coordinates are presentation, not model anatomy. */
(function(root,factory){
 const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_VISUAL_DATA=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0-u8',SCHEMA='scept.u8.scene.v1';
 const fail=(code,msg)=>{const e=Error('SCEPT U8: '+msg);e.code=code;throw e};
 const finite=n=>typeof n==='number'&&Number.isFinite(n);
 const record=o=>o!==null&&typeof o==='object'&&!Array.isArray(o);
 function validateObservation(o){
  if(!record(o)||typeof o.tensor!=='string'||o.evidence!=='actual-onnx-runtime-graph-outputs')
   fail('UNTRUSTED_OBSERVATION','View requires genuine ONNX Runtime U4 observation');
  if(!Array.isArray(o.shape)||!o.shape.length||!o.shape.every(d=>Number.isSafeInteger(d)&&d>0))
   fail('INVALID_OBSERVATION','Actual tensor shape required');
  if(!finite(o.mean_abs_delta)||!finite(o.mean_abs_A)||!finite(o.mean_abs_B)||!finite(o.rms_delta)||
   !Number.isSafeInteger(o.changed)||o.changed<0||o.changed>o.total)
   fail('INVALID_OBSERVATION','Finite actual A/B/delta and changed count required');
  if(!record(o.axisSummaries)||!Array.isArray(o.axes))fail('INVALID_OBSERVATION','U4 axis metadata absent');
  const axes=o.axes.filter(a=>['token','channel','feature'].includes(a.role));
  for(const a of axes){
   if(o.shape[a.index]!==a.size)fail('AXIS_MISMATCH','Axis size contradicts actual tensor shape');
   const data=o.axisSummaries[a.role];
   if(data!==undefined&&(data.length!==a.size||data.some(t=>!finite(t.mean_abs_A)||
     !finite(t.mean_abs_B)||!finite(t.mean_abs_delta)||!finite(t.rms_delta))))
    fail('AXIS_MISMATCH','Cannot invent values for missing U4 axis');
  }
  return o;
 }
 function tensorView(o){
  validateObservation(o);
  const axes=o.axes.filter(a=>['token','channel','feature'].includes(a.role)&&
   Array.isArray(o.axisSummaries[a.role]));
  const preferred=axes.find(a=>a.role==='token')||axes.find(a=>a.role==='channel')||axes.find(a=>a.role==='feature');
  return{tensor:o.tensor,shape:o.shape.slice(),source:o.source,evidence:o.evidence,
   profile:o.profile,architecture:o.architecture,axes:o.axes.map(a=>({...a})),
   role:preferred?.role||null,
   units:preferred?o.axisSummaries[preferred.role].map(t=>({index:t.index,
    A:t.mean_abs_A,B:t.mean_abs_B,delta:t.mean_abs_delta,rms:t.rms_delta,changed:t.changed})):[],
   summary:{A:o.mean_abs_A,B:o.mean_abs_B,delta:o.mean_abs_delta,rms:o.rms_delta,
    changed:o.changed,total:o.total},exactMapping:false,
   note:preferred?.role==='token'?'Token 编号是张量索引；空间位置仅用于可视布局，不能推断 MRI Patch 映射':
    preferred?.role==='channel'?'通道索引为真实张量轴；空间点位并非真实神经元位置':
    '仅展示已导出的实际 ONNX 张量统计；无可声明语义的轴不生成节点'};
 }
 function graphView(desc,observations,maxNodes=240){
  const g=desc?.graph;
  if(!g||!Array.isArray(g.operators)||!Array.isArray(g.dependencies))return{status:'unavailable',reason:'模型未导出可检查的真实计算图',nodes:[],edges:[],originalCount:0};
  const all=g.operators;if(all.length>10000||!Number.isSafeInteger(maxNodes)||maxNodes<5||maxNodes>400)
   fail('RESOURCE_LIMIT','Graph view exceeds safe node budget');
  const selected=new Set();
  for(let i=0;i<Math.min(maxNodes,all.length);i++){
   const id=Math.min(all.length-1,Math.floor((i+.5)*all.length/Math.min(maxNodes,all.length)));
   selected.add(id);
  }
  const producers=new Map();
  all.forEach((n,i)=>{for(const t of n.outputs||[])if(t)producers.set(t,i)});
  const measured=new Map();
  for(const o of observations){
   const node=producers.get(o.tensor);
   if(node!==undefined){measured.set(node,{tensor:o.tensor,A:o.mean_abs_A,B:o.mean_abs_B,
    delta:o.mean_abs_delta});selected.add(node)}
  }
  // A capped display, never a fabricated edge or full-graph representation.
  const ids=[...selected].sort((a,b)=>a-b);
  if(ids.length>420)fail('RESOURCE_LIMIT','Observed graph selection exceeds safe budget');
  const selectedIds=new Set(ids);
  const nodes=ids.map(id=>{const n=all[id],v=measured.get(id);
   return{id,op:n.op||'Unknown',name:n.name||'op_'+id,observed:v||null,
    outputs:(n.outputs||[]).slice(0,4)}});
  const edges=g.dependencies.filter(e=>selectedIds.has(e.from)&&selectedIds.has(e.to))
   .slice(0,1500).map(e=>({from:e.from,to:e.to,tensor:e.tensor}));
  return{status:'available',nodes,edges,originalCount:all.length,originalEdges:g.dependencies.length,
   renderedNodes:nodes.length,renderedEdges:edges.length,sampling:'deterministic subset plus measured producers',
   edgeSemantics:'actual producer-to-consumer ONNX tensor dependence only',
   unobservedSemantics:'neutral; no inferred activation'};
 }
 function createScene({modelSha256,description,observations,taskId='T001',
  provenance={},experiment=null}){
  if(typeof modelSha256!=='string'||!/^[a-f0-9]{64}$/i.test(modelSha256))fail('INVALID_MODEL','Verified model SHA-256 required');
  if(!Array.isArray(observations)||observations.length===0)fail('EMPTY_OBSERVATION','Real ONNX observation required');
  if(!Array.isArray(description?.outputs))fail('INVALID_SIGNATURE','Actual ONNX output signature required');
  const declared=new Set(description.outputs.map(o=>o.name));
  if(observations.some(o=>!declared.has(o.tensor)))fail('UNEXPOSED_TENSOR','Visualization cannot invent ONNX observations');
  const views=observations.map(tensorView);
  const active=views.filter(x=>x.source==='selected_real_intermediate').length;
  const graph=graphView(description,observations);
  return{schema:SCHEMA,version:VERSION,taskId,modelSha256,
   provenance:{modelSha256,engine:provenance.engine||'real-onnx-runtime',
    inputSource:provenance.inputSource||'unknown',sourceSchema:provenance.sourceSchema||null,
    experimentId:experiment?.id||null,
    warnings:['Coordinates in 3D graph/token views are synthetic visual layouts, not physical positions',
      'No clinical, anatomical or causal inference from numeric changes']},
   graph,views,observedCount:views.length,observedIntermediateCount:active,
   scientificValidation:'NOT_PERFORMED'};
 }
 function fromAudit({audit,description,modelSha256}){
  if(!record(audit)||!Array.isArray(audit.outputs)||audit.originalModelSHA256!==modelSha256)
   fail('AUDIT_MISMATCH','Actual A/B audit and loaded model SHA must match');
  return createScene({modelSha256,description,observations:audit.outputs,
   provenance:{sourceSchema:audit.schema,inputSource:audit.inputSource}});
 }
 function fromQueue({snapshot,taskId,description,modelSha256}){
  if(!record(snapshot)||snapshot.modelSha256!==modelSha256)fail('AUDIT_MISMATCH','Queue model SHA mismatch');
  const t=snapshot.tasks?.find(x=>x.id===taskId);
  if(!t||t.status!=='completed'||!t.result?.genuineInference)
   fail('TASK_UNAVAILABLE','Select a completed task with actual ONNX inference');
  return createScene({modelSha256,description,observations:t.result.outputs,taskId,
   provenance:{sourceSchema:t.result.schema,engine:'U6 true ONNX session'}});
 }
 function fromEvidence({evidence,taskId,description,modelSha256}){
  if(!record(evidence)||evidence.schema!=='scept.u7.evidence.v1'||
     evidence.provenance?.modelSha256!==modelSha256)
   fail('AUDIT_MISMATCH','U7 evidence model SHA mismatch');
  const task=evidence.tasks?.find(x=>x.id===taskId);
  if(!task||task.status!=='completed')fail('TASK_UNAVAILABLE','U7 task is not completed');
  return createScene({modelSha256,description,observations:task.observations,taskId,
   experiment:evidence.experiment,
   provenance:{sourceSchema:evidence.schema,engine:'U7 genuine ONNX evidence'}});
 }
 function verifyPaired(a,b){
  if(!a||!b||a.type!==b.type||!['float32','float64','int32','uint8'].includes(a.type)||
   !Array.isArray(a.dims)||a.dims.length!==b.dims?.length||
   a.dims.some((d,i)=>d!==b.dims[i]||!Number.isSafeInteger(d)||d<1))
   fail('INPUT_MISMATCH','Volume A/B typed tensors must have identical concrete shapes');
  const n=a.dims.reduce((p,d)=>p*d,1);
  if(!Number.isSafeInteger(n)||a.data?.length!==n||b.data?.length!==n)
   fail('INPUT_MISMATCH','Actual typed input data length mismatch');
 }
 function captureInput({baseline,edited,inputName,layout,source='model-input',taskId='T001',maxSide=72}){
  if(!record(baseline)||!record(edited)||typeof inputName!=='string')fail('INPUT_MISSING','Paired ONNX input feeds required');
  const a=baseline[inputName],b=edited[inputName];verifyPaired(a,b);
  if(!['NCHW','NHWC','NCDHW','NDHWC'].includes(layout))fail('LAYOUT_REQUIRED','Explicit tensor layout required');
  const three=layout==='NCDHW'||layout==='NDHWC';
  if(a.dims.length!==(three?5:4))fail('LAYOUT_MISMATCH','Explicit layout incompatible with input dimension rank');
  if(!Number.isInteger(maxSide)||maxSide<8||maxSide>96)fail('RESOURCE_LIMIT','Max side must be 8..96');
  const [batch,...shape]=a.dims;if(batch<1)fail('INPUT_MISMATCH','Zero batch');
  const axes=layout==='NCHW'?[a.dims[2],a.dims[3]]:
   layout==='NHWC'?[a.dims[1],a.dims[2]]:
   layout==='NCDHW'?[a.dims[2],a.dims[3],a.dims[4]]:
   [a.dims[1],a.dims[2],a.dims[3]];
  const sampled=axes.map(n=>Math.min(n,maxSide));
  if(sampled.reduce((p,v)=>p*v,1)>884736)fail('RESOURCE_LIMIT','Visual input sample too large');
  const length=sampled.reduce((p,v)=>p*v,1),A=new Float32Array(length),B=new Float32Array(length);
  const coordToIndex=(coord)=>{
   const c=0,n=0;
   if(layout==='NCHW')return ((n*a.dims[1]+c)*a.dims[2]+coord[0])*a.dims[3]+coord[1];
   if(layout==='NHWC')return ((n*a.dims[1]+coord[0])*a.dims[2]+coord[1])*a.dims[3]+c;
   if(layout==='NCDHW')return (((n*a.dims[1]+c)*a.dims[2]+coord[0])*a.dims[3]+coord[1])*a.dims[4]+coord[2];
   return (((n*a.dims[1]+coord[0])*a.dims[2]+coord[1])*a.dims[3]+coord[2])*a.dims[4]+c;
  };
  let changed=0,maxAbsDelta=0;
  for(let i=0;i<length;i++){
   let j=i,position=[];
   for(let k=sampled.length-1;k>=0;k--){position[k]=j%sampled[k];j=Math.floor(j/sampled[k])}
   const at=position.map((v,k)=>Math.min(axes[k]-1,Math.floor((v+.5)*axes[k]/sampled[k])));
   const rawIndex=coordToIndex(at),av=Number(a.data[rawIndex]),bv=Number(b.data[rawIndex]);
   if(!finite(av)||!finite(bv))fail('NONFINITE_INPUT','Cannot draw NaN/Infinity MRI/volume inputs');
   A[i]=av;B[i]=bv;
   if(!finite(A[i])||!finite(B[i]))fail('NONFINITE_INPUT','Float32 visual sample overflow');
   const d=Math.abs(bv-av);if(d>1e-7)changed++;maxAbsDelta=Math.max(maxAbsDelta,d);
  }
  return{schema:'scept.u8.inputview.v1',taskId,inputName,source,layout,shape:a.dims.slice(),
   axes:axes.slice(),sampled:sampled.slice(),resampled:axes.some((d,k)=>sampled[k]!==d),
   spatialDims:three?3:2,channelIndex:0,batchIndex:0,
   A,B,changedSampled:changed,maxAbsDelta,kind:three?'model-input-volume':'model-input-image',
   anatomicalStatus:'UNVERIFIED',warning:'Tensor-space input; MRI status, anatomy, geometry and registration are not inferred'};
 }
 function plane(v,orientation='axial',slice=null){
  if(!v||v.schema!=='scept.u8.inputview.v1')fail('INPUT_MISSING','No captured actual model input');
  if(!['axial','coronal','sagittal'].includes(orientation))fail('INVALID_PLANE','Unknown orientation');
  const dims=v.sampled,three=v.spatialDims===3;
  if(!three&&orientation!=='axial')return null;
  const axis=orientation==='axial'?0:orientation==='coronal'?1:2;
  const planeAxis=three?axis:-1;
  const where=three?(slice===null?Math.floor(dims[planeAxis]/2):slice):0;
  if(three&&(!Number.isInteger(where)||where<0||where>=dims[axis]))fail('INVALID_SLICE','Slice outside sampled tensor');
  const rows=three?(axis===0?dims[1]:dims[0]):dims[0];
  const cols=three?(axis===2?dims[1]:dims[2]):dims[1];
  if(three&&axis===1){/* D by W */ }
  const n=rows*cols,A=new Float32Array(n),B=new Float32Array(n),D=new Float32Array(n);
  const index=(a,b,c)=>a*dims[1]*dims[2]+b*dims[2]+c;
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
   let k;if(!three)k=r*dims[1]+c;
   else if(axis===0)k=index(where,r,c);
   else if(axis===1)k=index(r,where,c);
   else k=index(r,c,where);
   const idx=r*cols+c;A[idx]=v.A[k];B[idx]=v.B[k];D[idx]=B[idx]-A[idx];
  }
  return{orientation,axis:planeAxis,index:where,rows,cols,A,B,delta:D,
   metadata:{source:v.source,layout:v.layout,shape:v.shape,sampled:v.sampled,
    anatomicalStatus:'UNVERIFIED',resampled:v.resampled}};
 }
 return Object.freeze({VERSION,SCHEMA,validateObservation,tensorView,graphView,createScene,fromAudit,fromQueue,fromEvidence,
  captureInput,plane});
});