'use strict';
/* SCEPT V5 / falsifiable numeric neural fingerprint science.
 * Raw source must be genuine same-frozen-model ONNX tensors, never drawn pixels.
 * NDF = descriptive paired change; CRC = repeated-counterfactual concordance;
 * ICM = actual runtime patch intervention, separate from mere difference.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_V5_SCIENCE=api;
})(typeof window==='undefined'?null:window,function(){
 const VERSION='5.0.0-preview',NDF='scept.v5.ndf.v1',CRC='scept.v5.crc.v1',ICM='scept.v5.icm.v1';
 const err=(msg)=>{throw Error('SCEPT V5: '+msg)};
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 const med=a=>{let x=a.slice().sort((u,v)=>u-v);return x.length%2?x[(x.length-1)/2]:(x[x.length/2-1]+x[x.length/2])/2};
 const mean=x=>x.reduce((a,b)=>a+b,0)/x.length;
 const isArray=x=>Array.isArray(x)||ArrayBuffer.isView(x);
 function validateTensor(t,name){
  if(!t||!isArray(t.dims)||!isArray(t.data)||!t.dims.length||
   !Array.from(t.dims).every(d=>Number.isSafeInteger(d)&&d>0))err('invalid measured tensor '+name);
  let count=Array.from(t.dims).reduce((a,b)=>a*b,1);
  if(count>3000000||!Number.isSafeInteger(count)||count!==t.data.length)err('unbounded or inconsistent tensor '+name);
  if(!Array.from(t.data).every(finite))err('nonfinite tensor '+name);
  return count;
 }
 function unitsFor(dims,profile){
  if(profile==='vit-token-channel'){
   if(dims.length!==3||dims[0]!==1||dims[1]<2||dims[2]<1)err('ViT requires real [1,tokens,channels]');
   return{units:dims[1],stride:dims[2],semantics:'actual-vit-token-index'};
  }
  if(profile==='cnn-nchw'){
   if(dims.length!==4||dims[0]!==1)err('CNN requires NCHW');
   return{units:dims[1],stride:dims[2]*dims[3],semantics:'actual-cnn-channel-index'};
  }
  if(profile==='generic'){
   if(dims[0]!==1)err('generic profile requires batch=1');
   return{units:dims.length===1?dims[0]:dims[1],stride:dims.length===1?1:dims.slice(2).reduce((a,b)=>a*b,1),semantics:'declared-output-feature-index'};
  }
  err('unknown observation profile');
 }
 function ndf({modelSha256,baselineHash,counterfactualHash,layers,profile='vit-token-channel',source='real-onnx-runtime',label='unnamed'}){
  if(!/^[0-9a-f]{64}$/i.test(modelSha256||'')||!Array.isArray(layers)||!layers.length)
   err('model SHA-256 and actual layer data required');
  if(!/^[0-9a-f]{64}$/i.test(baselineHash||'')||!/^[0-9a-f]{64}$/i.test(counterfactualHash||''))
   err('raw input checksums required');
  const identicalInputs=baselineHash===counterfactualHash;
  if(source!=='real-onnx-runtime')err('cannot build science from mock or drawn values');
  const seen=new Set(),items=layers.map(l=>{
   if(typeof l.name!=='string'||!l.name||seen.has(l.name))err('duplicate or unnamed real tensor');
   seen.add(l.name);
   const a=l.a,b=l.b,n=validateTensor(a,l.name),n2=validateTensor(b,l.name);
   if(n!==n2||a.dims.some((d,i)=>b.dims[i]!==d)||a.dims.length!==b.dims.length)
    err('A/B tensor mismatch '+l.name);
   const {units,stride,semantics}=unitsFor(a.dims,profile);
   if(units*stride!==n)err('invalid feature axis in '+l.name);
   let sumAbs=0,sq=0,max=0,nonzero=0;
   const values=[];
   for(let u=0;u<units;u++){
    let ss=0,sa=0,signed=0,absa=0,absb=0,changed=0;
    for(let j=0;j<stride;j++){
     const i=u*stride+j,aa=Number(a.data[i]),bb=Number(b.data[i]),d=bb-aa,abs=Math.abs(d);
     ss+=d*d;sa+=abs;signed+=d;absa+=Math.abs(aa);absb+=Math.abs(bb);
     sumAbs+=abs;sq+=d*d;max=Math.max(max,abs);if(abs>1e-7){changed++;nonzero++}
    }
    values.push({index:u,rms:Math.sqrt(ss/stride),mae:sa/stride,
      signedMean:signed/stride,meanAbsA:absa/stride,meanAbsB:absb/stride,changed});
   }
   return{name:l.name,layer:Number.isInteger(l.layer)?l.layer:null,shape:Array.from(a.dims),
    semantics,units,stride,values,meanAbsDelta:sumAbs/n,rmsDelta:Math.sqrt(sq/n),
    maxAbsDelta:max,changed:nonzero,total:n};
  });
  return{schema:NDF,version:VERSION,modelSha256,baselineHash,counterfactualHash,
    source,profile,label,rows:items,identicalInputs,scientificStatus:identicalInputs?'zero_effect_control':'descriptive_internal_response_only',
    limitations:['A/B difference is not evidence of a causal path','A/B alone does not establish stability',
      'Display scales and activation magnitudes are not calibrated across different network layers']};
 }
 function ranking(values){
  return values.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v||a.i-b.i).map(x=>x.i);
 }
 function topSet(x,k){return new Set(ranking(x).slice(0,k))}
 function spearman(a,b){
  if(a.length!==b.length||a.length<2)return null;
  const ranks=x=>{const sorted=x.map((v,i)=>({v,i})).sort((u,v)=>u.v-v.v),out=new Array(x.length);
   for(let i=0;i<sorted.length;){let j=i+1;while(j<sorted.length&&sorted[j].v===sorted[i].v)j++;
    const rank=(i+j-1)/2;for(let p=i;p<j;p++)out[sorted[p].i]=rank;i=j;}return out};
  const x=ranks(a),y=ranks(b),mx=mean(x),my=mean(y);
  let num=0,sx=0,sy=0;for(let i=0;i<x.length;i++){let dx=x[i]-mx,dy=y[i]-my;num+=dx*dy;sx+=dx*dx;sy+=dy*dy}
  return sx>1e-12&&sy>1e-12?num/Math.sqrt(sx*sy):null;
 }
 function crc(repetitions,{topFraction=.1,minimum=3,zeroTolerance=1e-10}={}){
  if(!Array.isArray(repetitions)||repetitions.length<minimum||minimum<3||
   !finite(topFraction)||topFraction<=0||topFraction>=.5)err('at least 3 measured counterfactuals and valid top fraction required');
  const first=repetitions[0];
  if(!first||first.schema!==NDF||!first.rows.length)err('real NDF required');
  if(new Set(repetitions.map(r=>r.counterfactualHash)).size!==repetitions.length)
   err('duplicate counterfactual hashes are not independent realizations');
  for(const r of repetitions){
   if(r.schema!==NDF||r.modelSha256!==first.modelSha256||r.baselineHash!==first.baselineHash||
    r.profile!==first.profile||r.rows.length!==first.rows.length||
    r.rows.some((x,i)=>x.name!==first.rows[i].name||x.units!==first.rows[i].units||
      x.shape.join(',')!==first.rows[i].shape.join(',')))
    err('repetitions require same frozen model, same A and identical real probe signatures');
  }
  const rows=first.rows.map((row,i)=>{
   const series=repetitions.map(r=>r.rows[i].values.map(x=>x.rms));
   const maxResponse=Math.max(...series.flat());
   if(maxResponse<=zeroTolerance)
    return{name:row.name,layer:row.layer,verdict:'not_evaluable',reason:'all responses effectively zero',
      topKOverlap:null,rankCorrelation:null,medianUnitDispersion:null,units:row.units};
   const k=Math.max(1,Math.ceil(row.units*topFraction)),jaccard=[],spears=[],dispersions=[];
   for(let p=0;p<series.length;p++)for(let q=p+1;q<series.length;q++){
    const x=topSet(series[p],k),y=topSet(series[q],k),inter=[...x].filter(t=>y.has(t)).length;
    jaccard.push(inter/(2*k-inter));
    const sc=spearman(series[p],series[q]);if(sc!==null)spears.push(sc);
   }
   for(let u=0;u<row.units;u++){
    const mags=series.map(s=>s[u]),center=med(mags);
    dispersions.push(med(mags.map(v=>Math.abs(v-center))));
   }
   return{name:row.name,layer:row.layer,verdict:spears.length?'measured':'not_evaluable',
    reason:spears.length?null:'rank correlation undefined for constant response vectors',
    topKOverlap:med(jaccard),rankCorrelation:spears.length?med(spears):null,
    medianUnitDispersion:med(dispersions),units:row.units,topK:k};
  });
  return{schema:CRC,version:VERSION,nRealizations:repetitions.length,modelSha256:first.modelSha256,
    baselineHash:first.baselineHash,topFraction,rows,verdict:rows.every(r=>r.verdict==='measured')?'measured':'not_evaluable',
    scientificStatus:'repeatability_not_pathology_specificity',
    limitations:['Counterfactual variants of one patient are not independent patients',
     'No user-selected criterion_met without preregistered control distribution and thresholds',
     'High agreement alone does not establish causal tumor dependence']};
 }
 // Strictly gated ICM numeric result: downstream model execution must be real,
 // perturb a declared single real tensor using validated runtime patching.
 function icm({modelSha256,layerName,unit,sA,sB,sPatchedAB,sPatchedBA,offA,offB,
   nonceSource='real-onnx-patch-runtime',tolerance=1e-4}){
  if(nonceSource!=='real-onnx-patch-runtime'||!/^[a-f0-9]{64}$/i.test(modelSha256||''))
   err('ICM requires genuinely executed, provenance-bound patchable ONNX model');
  if(!layerName||!Number.isInteger(unit)||unit<0||![sA,sB,sPatchedAB,sPatchedBA,offA,offB,tolerance].every(finite)||
   tolerance<=0)err('invalid real ICM values');
  if(Math.abs(sA-offA)>tolerance||Math.abs(sB-offB)>tolerance)
   err('no-op patched graph does NOT reproduce the original model');
  const denom=sA-sB;
  if(Math.abs(denom)<=tolerance)return{schema:ICM,verdict:'not_evaluable',
    reason:'original target score gap below preregistered noise tolerance',layerName,unit};
  // sPatchedAB runs B input with unit activation A; sPatchedBA runs A with unit B.
  return{schema:ICM,version:VERSION,modelSha256,layerName,unit,verdict:'measured',
    reverseFraction:(sA-sPatchedBA)/denom,recoveryFraction:(sPatchedAB-sB)/denom,
    values:{sA,sB,sPatchedAB,sPatchedBA,offA,offB},tolerance,
    scientificStatus:'actual_internal_patch_output_response_not_tumor_causal_identification',
    limitations:['Patching can create off-manifold internal states','Unit effects need not add up',
      'Recovery ratio is neither a probability nor a calibrated importance score']};
 }
 return Object.freeze({VERSION,NDF,CRC,ICM,ndf,crc,icm,unitsFor,spearman});
});
