'use strict';
/* SCEPT U10 / Scientific Explanation Mechanism v1.0.
 * Standalone trusted SDK extension. Every numerical response MUST be derived
 * from U7 actual ONNX forward passes A/B/C, not synthetic saliency.
 * This algorithm is an intervention-evidence TEST, not causal identification.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_U10_MECHANISM=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0-u10',SCHEMA='scept.u10.explanation.v1',INTR='scept.u10.intervention.v1';
 const IDs=Object.freeze({intervention:'scept.u10.paired-evidence-intervention',
  explanation:'scept.u10.method.evidential-dependence',
  perTask:'scept.u10.eval.paired-evidence',
  suite:'scept.u10.eval.preregistered-gates'});
 const fail=(code,msg)=>{const e=new Error('SCEPT U10: '+msg);e.code=code;throw e};
 const obj=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const finite=(n,key)=>{if(typeof n!=='number'||!Number.isFinite(n))fail('NONFINITE',key+' requires finite number');return n};
 const int=(n,key)=>{if(!Number.isSafeInteger(n))fail('INVALID_PARAMETER',key+' requires integer');return n};
 const uint=n=>int(n,'nuisanceSeed')>=0&&n<=4294967295;
 const bounded=(v,key,lo,hi)=>{finite(v,key);if(v<lo||v>hi)fail('INVALID_PARAMETER',key+' outside ['+lo+','+hi+']');return v};
 const median=values=>{
  if(!values.length)fail('INVALID_DATA','Cannot calculate median from empty samples');
  const v=values.slice().sort((a,b)=>a-b),m=v.length>>1;
  return v.length%2?v[m]:(v[m-1]+v[m])/2;
 };
 const verdict=ok=>ok?'criterion_met':'criterion_not_met';
 const fixed=n=>Math.abs(n)<1e-12?0:n;
 function mask(config,side,t){
  if(!obj(config)||typeof config.sourceId!=='string'||!config.sourceId.trim()||config.sourceId.length>160||
   config.coordinateSpace!=='model-input-flat-indices'||config.inputName!==side.inputName||
   !Array.isArray(config.shape)||config.shape.length!==t.dims.length||
   config.shape.some((x,i)=>x!==t.dims[i])||
   !Array.isArray(config.indices)||config.indices.length<1||config.indices.length>100000||
   typeof config.indicesSha256!=='string'||!/^[a-f0-9]{64}$/i.test(config.indicesSha256))
   fail('MASK_CONTRACT','Explicit tensor-space ROI source, dims, bounded sorted indices and SHA256 required');
  let prev=-1;
  for(const x of config.indices){
   int(x,'mask index');
   if(x<=prev||x>=t.data.length||x<0)fail('MASK_CONTRACT','Mask indices must be sorted, unique and in real input bounds');
   prev=x;
  }
  return config.indices;
 }
 function stats(t,indices){
  let sum=0,sumSq=0;
  for(const i of indices){const v=Number(t.data[i]);finite(v,'source tensor value');sum+=v;sumSq+=v*v}
  const mean=sum/indices.length,variance=Math.max(0,sumSq/indices.length-mean*mean);
  return{mean,std:Math.sqrt(variance)};
 }
 function rng(seed){
  let s=seed>>>0;
  return()=>{s=(s+0x6d2b79f5)>>>0;let t=s;t=Math.imul(t^(t>>>15),t|1);
   t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296};
 }
 function gaussian(r){
  const u=Math.max(r(),Number.MIN_VALUE),v=r();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
 }
 function copy(feeds){
  const out={};for(const [name,t]of Object.entries(feeds))out[name]={type:t.type,dims:t.dims.slice(),
   data:new t.data.constructor(t.data)};return out;
 }
 function plan({inputName,shape,targetMask,controlMask,fill=0,strengths=[0,.5,1],
  nuisanceSeeds=[109,211,307],nuisanceSigma=.001,meanTolerance=.02,stdTolerance=.02,
  baseSeed=240216,profileId='generic',targetTensor,targetIndex=0,modelSha256,
  minTargetEffect=.01,minSpecificity=.005,minPositiveFraction=1,
  maxSpread=.02,minimumSignAgreement=1,doseTolerance=.005,nullTolerance=.00001,
  outputScale=1,expectedSign=null}={}){
  if(!/^[a-f0-9]{64}$/i.test(modelSha256||'')||typeof targetTensor!=='string'||!targetTensor)
   fail('INVALID_PLAN','Real frozen model SHA256 and explicit output required');
  if(!Array.isArray(strengths)||strengths.length<3||strengths.length>10||
   !strengths.every(x=>typeof x==='number'&&Number.isFinite(x)&&x>=0&&x<=1)||
   new Set(strengths).size!==strengths.length||strengths[0]!==0||strengths[strengths.length-1]!==1||
   strengths.some((x,i)=>i>0&&x<=strengths[i-1]))
   fail('INVALID_PLAN','Pre-register sorted distinct strengths (3..10), including 0 and 1');
  if(!Array.isArray(nuisanceSeeds)||nuisanceSeeds.length<3||nuisanceSeeds.length>12||
   nuisanceSeeds.some(x=>!uint(x))||new Set(nuisanceSeeds).size!==nuisanceSeeds.length)
   fail('INVALID_PLAN','3..12 unique uint32 nuisance seeds required');
  if(strengths.length*nuisanceSeeds.length>128)fail('RESOURCE_LIMIT','Task count exceeds scheduler 128');
  if(!Array.isArray(shape)||shape.length<1||shape.some(x=>!Number.isSafeInteger(x)||x<1))
   fail('INVALID_PLAN','Explicit actual input dims required');
  if(!obj(targetMask)||!obj(controlMask))fail('INVALID_PLAN','Both externally sourced ROI masks are required');
  if(!Number.isSafeInteger(targetIndex)||targetIndex<0)fail('INVALID_PLAN','Explicit target flat index required');
  for(const [k,v,lo,hi]of [
   ['nuisanceSigma',nuisanceSigma,0,.1],['meanTolerance',meanTolerance,0,100],
   ['stdTolerance',stdTolerance,0,100],['minTargetEffect',minTargetEffect,0,100000],
   ['minSpecificity',minSpecificity,0,100000],['minPositiveFraction',minPositiveFraction,0,1],
   ['maxSpread',maxSpread,0,100000],['minimumSignAgreement',minimumSignAgreement,0,1],
   ['doseTolerance',doseTolerance,0,100000],['nullTolerance',nullTolerance,0,100000],
   ['outputScale',outputScale,1e-12,1e12]])bounded(v,k,lo,hi);
  if(expectedSign!==null&&![1,-1].includes(expectedSign))
   fail('INVALID_PLAN','expectedSign is predeclared +1/-1 or null (sign-agnostic)');
  const controls=[];
  for(const seed of nuisanceSeeds)for(const strength of strengths){
   controls.push({id:'seed-'+seed+'-dose-'+strength,config:{inputName,fill,strength,
    nuisanceSeed:seed,nuisanceSigma,meanTolerance,stdTolerance,
    targetMask,controlMask}});
  }
  const suite={minReplicates:nuisanceSeeds.length,minTargetEffect,minSpecificity,
   minPositiveFraction,maxSpread,minimumSignAgreement,doseTolerance,nullTolerance,
   expectedStrengths:strengths,expectedNuisanceSeeds:nuisanceSeeds,expectedSign,outputScale};
  return{schema:'scept.u7.experiment.v1',id:'scept-u10-evidence-'+modelSha256.slice(0,12),
   version:VERSION,model:{sha256:modelSha256,requiredCapabilities:['inference']},
   target:{adapterId:'u7.target.tensor-element',tensor:targetTensor,index:targetIndex},
   intervention:{mode:'batch',pluginId:IDs.intervention,baseSeed,config:{},controls},
   observation:{profileId,tensors:[targetTensor]},
   method:{adapterId:IDs.explanation,params:{outputScale}},
   evaluations:[
    {adapterId:IDs.perTask,params:{minSpecificity}},
    {adapterId:IDs.suite,params:suite}
   ],
   hypotheses:{mechanism:'pre-registered paired evidence dependence, nuisance-stability and graded intervention',
    anatomy:'UNVERIFIED',causal:'NOT_ESTABLISHED',clinical:'NOT_ESTABLISHED'}};
 }
 function runIntervention(core,description,baseline,config){
  const name=config.inputName;
  if(typeof name!=='string'||!Object.hasOwn(baseline,name))fail('INPUT_MISSING','Exact declared input required');
  const t=baseline[name],m={inputName:name};
  if(!['float32','float64'].includes(t.type))fail('INVALID_INPUT','Only floating point ONNX input intervention supported');
  if(!Array.isArray(description.inputs)||!description.inputs.some(i=>i.name===name))
   fail('INPUT_MISSING','Input not declared in inspected model');
  const target=mask(config.targetMask,m,t),control=mask(config.controlMask,m,t);
  if(target.length!==control.length)fail('CONTROL_UNMATCHED','Control must have same count as target');
  const s=new Set(target);
  if(control.some(i=>s.has(i)))fail('CONTROL_OVERLAP','Matched control and target cannot overlap');
  const fill=finite(config.fill,'fill'),strength=bounded(config.strength,'strength',0,1);
  const sigma=bounded(config.nuisanceSigma,'nuisanceSigma',0,.1);
  const seed=config.nuisanceSeed;
  if(!uint(seed))fail('INVALID_PARAMETER','Unique uint32 nuisance seed required');
  const tolMean=bounded(config.meanTolerance,'meanTolerance',0,100),
   tolStd=bounded(config.stdTolerance,'stdTolerance',0,100);
  return Promise.all([
    core.sha256(Uint32Array.from(target)),core.sha256(Uint32Array.from(control))
  ]).then(([targetHash,controlHash])=>{
   if(targetHash.toLowerCase()!==config.targetMask.indicesSha256.toLowerCase()||
      controlHash.toLowerCase()!==config.controlMask.indicesSha256.toLowerCase())
    fail('MASK_HASH_MISMATCH','External ROI/matched control mask checksum mismatch');
   // U5 guarantees a completely unchanged A feed for trusted plugin safety.
   // Apply an IDENTICAL nuisance realization only to the B/C intervention arms.
   // At dose=0, B==C numerically, so their controlled difference is exactly null.
   // This measures robustness of a B-vs-C contrast under shared nuisance, NOT
   // baseline model robustness across independent altered A inputs.
   const A=copy(baseline),B=copy(baseline),C=copy(baseline),r=rng(seed);
   for(const key of Object.keys(A)){
    const a=A[key],b=B[key],c=C[key];
    for(let i=0;i<a.data.length;i++){
     if(!['float32','float64'].includes(a.type))continue;
     const x=Number(a.data[i])+sigma*gaussian(r);
     if(!Number.isFinite(x))fail('NONFINITE_INPUT','Nuisance perturbation overflow');
     b.data[i]=x;c.data[i]=x;
     if(!Number.isFinite(b.data[i]))fail('NONFINITE_INPUT','Typed nuisance overflow');
    }
   }
   const tA=A[name],tB=B[name],tC=C[name],ts=stats(tB,target),cs=stats(tC,control);
   const meanDiff=Math.abs(ts.mean-cs.mean),stdDiff=Math.abs(ts.std-cs.std);
   if(meanDiff>tolMean||stdDiff>tolStd)
    fail('CONTROL_UNMATCHED','Pre-intervention target/control numeric mean or std outside predeclared tolerance');
   for(const [idxs,buf]of [[target,tB.data],[control,tC.data]]){
    for(const i of idxs){
     const original=Number(buf[i]),next=original+strength*(fill-original);
     if(!Number.isFinite(next))fail('NONFINITE_INPUT','Target/control occlusion overflow');
     buf[i]=next;if(!Number.isFinite(buf[i]))fail('NONFINITE_INPUT','Typed input conversion overflow');
    }
   }
   return{baseline:A,edited:B,control:C,metadata:{
    schema:INTR,kind:'paired-controlled-numeric-occlusion',inputName:name,
    strength,nuisanceSeed:seed,nuisanceSigma:sigma,fill,
    selected:target.length,targetCount:target.length,controlCount:control.length,
    targetMaskSha256:targetHash,controlMaskSha256:controlHash,
    targetSourceId:config.targetMask.sourceId,controlSourceId:config.controlMask.sourceId,
    coordinateSpace:'model-input-flat-indices',meanDiff,stdDiff,meanTolerance:tolMean,stdTolerance:tolStd,
    matching:'equal element counts; disjoint ROI; pre-intervention numeric mean/std checks; no anatomical matching proof',
    anatomy:'NOT_VERIFIED',segmentation:'NOT_VERIFIED',clinical:'NOT_ESTABLISHED',
    nuisance:'seeded Gaussian applied identically to B/C only (A unchanged); distinct seeds form correlated controlled-contrast replicates'}};
  });
 }
 function method({target,observations,matchedControlObservations,intervention,config={}}){
  if(!obj(intervention)||intervention.schema!==INTR||target.controlDelta===null)
   fail('CONTROL_REQUIRED','Actual A/B/C from U10 paired-intervention required');
  if(!Array.isArray(observations)||!observations.length||
   !Array.isArray(matchedControlObservations)||matchedControlObservations.length!==observations.length)
   fail('OBSERVATIONS_REQUIRED','A/B and A/C real ONNX observations required');
  const scale=bounded(config.outputScale,'outputScale',1e-12,1e12),
    dt=finite(target.delta,'target.delta'),dc=finite(target.controlDelta,'target.controlDelta');
  const t=Math.abs(dt)/scale,c=Math.abs(dc)/scale,margin=t-c,
    controlled=(dt-dc)/scale;
  const layers=[];
  for(let i=0;i<observations.length;i++){
   const x=observations[i],y=matchedControlObservations[i];
   if(x.evidence!=='actual-onnx-runtime-graph-outputs'||y.evidence!=='actual-onnx-runtime-graph-outputs'||
     x.tensor!==y.tensor||JSON.stringify(x.shape)!==JSON.stringify(y.shape))
    fail('UNTRUSTED_OBSERVATION','Mismatched or fabricated ONNX observation');
   layers.push({tensor:x.tensor,shape:x.shape,
    targetedMeanAbsoluteChange:finite(x.mean_abs_delta,'layer target response'),
    matchedControlMeanAbsoluteChange:finite(y.mean_abs_delta,'layer control response'),
    matchedDifference:x.mean_abs_delta-y.mean_abs_delta});
  }
  return{schema:SCHEMA,method:'SCEPT-Evidential-Dependence-v1',
   outputType:'paired-controlled-target-dependence',
   score:fixed(margin),targetedEffect:t,matchedControlEffect:c,
   controlledSignedDelta:controlled,controlledMagnitude:Math.abs(controlled),
   targetSignedDelta:dt/scale,controlSignedDelta:dc/scale,
   nuisanceSeed:intervention.nuisanceSeed,nuisanceSigma:intervention.nuisanceSigma,
   strength:intervention.strength,controlMatching:{targetCount:intervention.targetCount,
    controlCount:intervention.controlCount,meanDiff:intervention.meanDiff,stdDiff:intervention.stdDiff,
    meanTolerance:intervention.meanTolerance,stdTolerance:intervention.stdTolerance},
   maskHashes:{target:intervention.targetMaskSha256,control:intervention.controlMaskSha256},
   layerResponses:layers,
   semantics:'specificity margin = (|B-A|-|C-A|)/outputScale; controlled contrast = (B-C)/outputScale; nuisance applied equally to B/C but NOT A',
   limits:['Numerical evidence dependence requires preregistered controls and cohort gates',
    'Controlled tensor intervention is not proof of causal lesion dependence or clinical interpretability']};
 }
 function perTask({target,method:result,intervention,config={}}){
  if(!result||result.schema!==SCHEMA||intervention?.schema!==INTR)
   fail('INVALID_METHOD','U10 method and matched intervention required');
  const minimum=bounded(config.minSpecificity??0,'minSpecificity',0,1e5);
  if(result.strength===0)return{schema:'scept.u10.test.v1',test:'paired-specificity',
   verdict:'not_evaluable',reason:'Zero-dose is an explicit null calibration cell, not an explanation',
   strength:0,scientificStatus:'NUMERIC_CONTROL_CALIBRATION_ONLY'};
  return{schema:'scept.u10.test.v1',test:'paired-specificity',
   verdict:verdict(result.score>=minimum&&result.targetedEffect>0&&result.controlledMagnitude>0),
   observed:result.score,threshold:minimum,controlEffect:result.matchedControlEffect,
   strength:result.strength,scientificStatus:'NUMERIC_PREDECLARED_CONTROL_ONLY'};
 }
 function suite({tasks,config}){
  const base={schema:'scept.u10.gates.v1',test:'SCEPT-Evidential-Dependence-v1',
   scientificStatus:'NUMERIC_PREDECLARED_COMPUTATIONAL_CRITERIA_ONLY',
   clinical:'NOT_ESTABLISHED',causal:'NOT_ESTABLISHED'};
  const no=reason=>({...base,verdict:'not_evaluable',reason,gates:[],nTasks:tasks?.length||0});
  if(!Array.isArray(tasks)||tasks.length<9||tasks.some(t=>t.status!=='completed'))
   return no('At least 3 dose levels × 3 completed independent nuisance seeds required');
  const q=config;
  if(!obj(q)||!Array.isArray(q.expectedStrengths)||!Array.isArray(q.expectedNuisanceSeeds))
   return no('Pre-registered strengths and nuisance seeds required');
  let strengths=q.expectedStrengths,seeds=q.expectedNuisanceSeeds;
  if(strengths.length<3||new Set(strengths).size!==strengths.length||
   strengths[0]!==0||strengths[strengths.length-1]!==1||
   seeds.length<3||new Set(seeds).size!==seeds.length)
   return no('Invalid or incomplete preregistered nuisance-dose study');
  const seen=new Map(),notes=[],sign=t=>t>0?1:t<0?-1:0;
  for(const task of tasks){
   const m=task.result?.analysis?.method,ctrl=task.result?.intervention,tar=task.result?.analysis?.target;
   if(!m||m.schema!==SCHEMA||ctrl?.schema!==INTR||tar?.controlDelta===null||
    !task.result?.genuineInference)return no('Tasks lack real paired triple ONNX forward evidence');
   const key=ctrl.nuisanceSeed+':'+ctrl.strength;
   if(seen.has(key))return no('Duplicate (nuisanceSeed,strength) pair; no independent replicate');
   if(!seeds.includes(ctrl.nuisanceSeed)||!strengths.includes(ctrl.strength))
    return no('Actual executed task outside preregistered seed or strength');
   if(ctrl.targetCount!==ctrl.controlCount||ctrl.meanDiff>ctrl.meanTolerance||
      ctrl.stdDiff>ctrl.stdTolerance||
      ctrl.targetMaskSha256!==tasks[0].result.intervention.targetMaskSha256||
      ctrl.controlMaskSha256!==tasks[0].result.intervention.controlMaskSha256||
      ctrl.nuisanceSigma!==tasks[0].result.intervention.nuisanceSigma)
    return no('Mismatched control count, ROI hashes, nuisance law or control diagnostics');
   if(!Number.isFinite(m.score)||!Number.isFinite(m.targetedEffect)||!Number.isFinite(m.matchedControlEffect))
    return no('Nonfinite scientific statistic');
   seen.set(key,m);
  }
  if(seen.size!==seeds.length*strengths.length)return no('Not all preregistered seed×dose cells executed');
  for(const seed of seeds)for(const dose of strengths)
   if(!seen.has(seed+':'+dose))return no('Missing dose '+dose+' for nuisance seed '+seed);
  const get=(s,q)=>seen.get(s+':'+q),max=seeds.map(s=>get(s,1)),
   controlled=max.map(m=>m.controlledSignedDelta),
   d=max.map(m=>m.targetSignedDelta),abs=max.map(m=>m.targetedEffect),
   scores=max.map(m=>m.score);
  const medianControlled=median(max.map(m=>m.controlledMagnitude)),
   medianTarget=median(abs),medianSpecificity=median(scores),
   positiveFraction=scores.filter(x=>x>=q.minSpecificity).length/scores.length,
   medianSigned=median(controlled),effectSpread=Math.max(...controlled.map(x=>Math.abs(x-medianSigned))),
   direction=q.expectedSign===null?sign(medianSigned):q.expectedSign,
   signAgreement=controlled.filter(x=>sign(x)===direction).length/d.length;
  const doseProblems=[],nullCells=[],maxControl=[];
  for(const seed of seeds){
   const baseline=get(seed,0);
   nullCells.push(baseline.controlledMagnitude);
   maxControl.push(get(seed,1).matchedControlEffect);
   for(let k=1;k<strengths.length;k++){
    const prev=get(seed,strengths[k-1]).controlledMagnitude,
     cur=get(seed,strengths[k]).controlledMagnitude;
    if(cur+q.doseTolerance<prev)doseProblems.push({seed,from:strengths[k-1],to:strengths[k],decrease:prev-cur});
   }
  }
  const gates=[
   {name:'paired-control-specificity',value:medianSpecificity,threshold:q.minSpecificity,
    verdict:verdict(medianSpecificity>=q.minSpecificity&&medianTarget>0)},
   {name:'controlled-intervention-effect',value:medianControlled,threshold:q.minTargetEffect,
    verdict:verdict(medianControlled>=q.minTargetEffect&&medianControlled>0)},
   {name:'replicate-specificity',value:positiveFraction,threshold:q.minPositiveFraction,
    verdict:verdict(positiveFraction>=q.minPositiveFraction)},
   {name:'nuisance-effect-stability',value:effectSpread,threshold:q.maxSpread,
    verdict:verdict(effectSpread<=q.maxSpread)},
   {name:'nuisance-sign-agreement',value:signAgreement,threshold:q.minimumSignAgreement,
    verdict:verdict(direction!==0&&signAgreement>=q.minimumSignAgreement)},
   {name:'graded-sensitivity',value:doseProblems.length,threshold:q.doseTolerance,
    verdict:verdict(doseProblems.length===0)},
   {name:'zero-dose-null',value:Math.max(...nullCells),threshold:q.nullTolerance,
    verdict:verdict(Math.max(...nullCells)<=q.nullTolerance)}
  ];
  return{...base,verdict:gates.every(g=>g.verdict==='criterion_met')?'criterion_met':'criterion_not_met',
   nTasks:tasks.length,nReplicates:seeds.length,strengths:strengths.slice(),seeds:seeds.slice(),
   gates,metrics:{medianTarget,medianControlled,medianSpecificity,positiveFraction,effectSpread,
    medianSigned,signAgreement,doseViolations:doseProblems.length,
    maxZeroDose:Math.max(...nullCells),medianControlAtFullDose:median(maxControl),
    outputScale:q.outputScale},
   doseProblems: doseProblems.slice(0,15),
   notes:['Stability is response consistency under seeded nuisance input shifts, not proof of causal evidence',
    'Matched control checks only pre-intervention numeric ROI count/mean/std, not anatomy/semantics',
    'A is unchanged; stability tests B-C under shared nuisance, not independent baseline robustness',
    'Correlated dose repeats within a seed are not independent patient replicates',
    'Thresholds are user-preregistered engineering gates, not calibrated statistical confidence or p-values']};
 }
 function install({sdk,registry,core}){
  if(!sdk?.install||!registry?.register||!core?.sha256)
   fail('INVALID_ENGINE','Trusted U9 SDK, U1 registry and SHA256 required');
  const plugin=[
   {id:IDs.intervention,kind:'intervention',version:VERSION,capabilities:['input.intervention'],
    async apply({description,baseline,config}){return runIntervention(core,description,baseline,config)}},
   {id:IDs.explanation,kind:'explanation',version:VERSION,capabilities:['method.explain'],
    calculate:method},
   {id:IDs.perTask,kind:'evaluation',version:VERSION,capabilities:['evaluation.score'],
    evaluate:perTask},
   {id:IDs.suite,kind:'evaluation',version:VERSION,capabilities:['evaluation.score'],
    scope:'suite',evaluateSuite:suite}
  ];
  return plugin.map(adapter=>sdk.install({core,registry,adapter}));
 }
 return Object.freeze({VERSION,SCHEMA,IDs,install,plan,method,perTask,suite,median});
});