'use strict';
/* SCEPT Universal Core V1 — independently packaged TRUSTED method SDK template.
 * Review code before loading; JSON manifests NEVER evaluate JavaScript.
 * Replace the example with a scientifically justified algorithm and tests.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_TEMPLATE_EXTENSION=api;})(typeof window!=='undefined'?window:null,function(){
 return Object.freeze({
  id:'research.template-method-bundle',version:'1.0.0',
  install({sdk,core,registry}){
   const method={id:'research.example-numeric-response',kind:'explanation',version:'1.0.0',
    capabilities:['method.explain'],
    calculate({target,observations,config={}}){
     if(!Number.isFinite(target?.delta)||!Array.isArray(observations)||
        observations.some(o=>o.evidence!=='actual-onnx-runtime-graph-outputs'))
      throw Error('Real ONNX target and U4 observations required');
     const scale=config.scale??1;
     if(!Number.isFinite(scale)||scale<0||scale>10)throw Error('Invalid scale');
     return{schema:'research.example.numeric.v1',method:'example-numeric-response',
      score:Math.abs(target.delta)*scale,outputType:'numeric-example',
      scientificStatus:'UNVALIDATED SDK example; NOT an established explanation'};
    }};
   const evaluator={id:'research.example-numeric-test',kind:'evaluation',version:'1.0.0',
    capabilities:['evaluation.score'],
    evaluate({method,config={}}){
     const bound=config.minimum??0;
     if(!Number.isFinite(bound)||bound<0)throw Error('Invalid minimum');
     return{schema:'research.example.test.v1',test:'numeric-minimum',
      verdict:method.score>=bound?'criterion_met':'criterion_not_met',
      observed:method.score,threshold:bound,scientificStatus:'NUMERIC_ONLY'};
    }};
   return[sdk.install({registry,core,adapter:method}),
    sdk.install({registry,core,adapter:evaluator})];
  }
 });
});