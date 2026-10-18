'use strict';
/* V6.3 fixed raw color transforms and provenance-bound empirical reference.
 * No adaptive autoscale in raw mode. No fake population clinical percentiles.
 */
(function(root,factory){const a=factory();if(typeof module==='object'&&module.exports)module.exports=a;if(root)root.SCEPT_NAF_CALIBRATION=a})(typeof window!=='undefined'?window:null,function(){
 const VERSION='6.3.0';
 function fail(m){throw Error('SCEPT NAF calibration: '+m)}
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 function metric(source){if(!source||source.schema!=='scept.naf.v1')fail('Genuine NAF required');return source}
 const key=(group,index)=>group.tensor+'\u0001'+index;
 function buildReference(samples,{label,domain,kind='within-volume',minSamples=5}={}){
  if(!['within-volume','independent-cohort'].includes(kind)||!label||!domain||!Array.isArray(samples)||
   samples.length<Math.max(5,minSamples))fail('>=5 real observations and explicit provenance domain/kind needed');
  const first=metric(samples[0]),orig=first.provenance,points=new Map();
  const inputs=new Set();
  for(const s of samples){
   metric(s);
   if(s.profile!==first.profile||s.provenance.model.sha256!==orig.model.sha256||
    s.provenance.preprocessing.sha256!==orig.preprocessing.sha256||
    JSON.stringify(s.groups.map(g=>({tensor:g.tensor,shape:g.shape,role:g.role})))!==
    JSON.stringify(first.groups.map(g=>({tensor:g.tensor,shape:g.shape,role:g.role}))))
    fail('Different model, preprocessing, layer or unit shape not reference-compatible');
   const signatures=JSON.stringify(s.provenance.inputs);
   if(inputs.has(signatures))fail('Duplicated identical input cannot count as a reference sample');
   inputs.add(signatures);
   for(const g of s.groups){
    if(g.status!=='measured')fail('Cannot calibrate an unobserved layer');
    for(const u of g.units){
     if(!finite(u.rms)||u.rms<0)fail('Invalid RMS in real reference');
     const k=key(g,u.index);if(!points.has(k))points.set(k,[]);
     points.get(k).push(u.rms);
    }
   }
  }
  const distributions={};for(const [k,vals] of points)distributions[k]=vals.sort((a,b)=>a-b);
  return{schema:'scept.naf.reference.v1',version:VERSION,modelSha256:orig.model.sha256,
   preprocessSha256:orig.preprocessing.sha256,profile:first.profile,label,domain,kind,nSamples:samples.length,
   distributions,scientificMeaning:kind==='within-volume'?
    'descriptive within-volume local slice rank; NOT population reference and NOT clinical abnormality':
    'empirical rank versus predeclared independent comparison samples, NOT significance or pathology probability'};
 }
 function empirical(x,sorted){
  if(!sorted||sorted.length<5||!finite(x))return null;
  // midrank for exact ties; no invented normality or p-values
  let smaller=0,equal=0;for(const v of sorted){if(v<x)smaller++;if(v===x)equal++}
  return (smaller+equal*.5)/sorted.length;
 }
 function calibrated(naf,reference){
  metric(naf);
  if(!reference||reference.schema!=='scept.naf.reference.v1'||reference.profile!==naf.profile||
   reference.modelSha256!==naf.provenance.model.sha256||
   reference.preprocessSha256!==naf.provenance.preprocessing.sha256)
   fail('Reference is incompatible with exact model, preprocessing or profile');
  return naf.groups.map(g=>({...g,units:g.units.map(u=>({...u,
   empiricalRank:empirical(u.rms,reference.distributions[key(g,u.index)])}))}));
 }
 // Immutable scientific mapping: raw data magnitude uses externally chosen raw max
 // and saturates above fixed bound; neither max nor reference mutates per image.
 function scale(value,{mode='raw',rawMax=2,referenceRank=null}={}){
  if(!finite(value)||value<0)fail('Measured RMS must be finite and nonnegative');
  if(mode==='raw'){if(!finite(rawMax)||rawMax<=0)fail('Positive fixed raw max required');
   return Math.min(1,value/rawMax);}
  if(mode==='reference'){if(!finite(referenceRank)||referenceRank<0||referenceRank>1)
   fail('Actual reference rank required');return referenceRank}
  fail('Within-run pseudo-calibration is not a fixed scientific scale');
 }
 return Object.freeze({VERSION,buildReference,calibrated,scale,empirical});
});
