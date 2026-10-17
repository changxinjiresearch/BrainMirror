'use strict';
/* Third-party-style U7 extension example. This file is NOT imported by the
 * SCEPT main program; it can be loaded as a trusted addon independently. */
(function(root,factory){
 const extension=factory();if(typeof module==='object'&&module.exports)module.exports=extension;
 if(root)root.SCEPT_RESEARCH_EXAMPLE=extension;
})(typeof window!=='undefined'?window:null,function(){
 return Object.freeze({
  register(registry){
   registry.register('explanation',{id:'research.method.signed-double',kind:'explanation',
    capabilities:['method.explain'],calculate({target}){
     const v=target.delta*2;
     if(!Number.isFinite(v))throw Error('External XAI method has nonfinite result');
     return{schema:'research.method.v1',method:'signed-double',score:v,
      claim:'Example numeric scalar transform only, not a validated XAI algorithm'};
    }});
   registry.register('evaluation',{id:'research.eval.signed-double-nonzero',kind:'evaluation',
    capabilities:['evaluation.score'],evaluate({method}){
     return{schema:'research.test.v1',test:'signed-double-nonzero',
      verdict:method.score===0?'criterion_not_met':'criterion_met',
      scientificStatus:'example code extension; no medical validation'};
    }});
  }
 });
});