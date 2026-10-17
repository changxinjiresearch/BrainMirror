'use strict';
/* U9 SDK example: a genuinely executable intervention from independent file.
 * This is typed numeric scaling, NOT medical lesion deletion or causal XAI.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_INTERVENTION_EXTENSION_TEMPLATE=api;})(typeof window!=='undefined'?window:null,function(){
 return Object.freeze({
  id:'research.input-scale-probe',version:'1.0.0',
  install({sdk,core,registry}){
   return sdk.install({core,registry,adapter:{
    kind:'intervention',id:'research.input-scale-probe',version:'1.0.0',
    capabilities:['input.intervention'],
    apply({description,baseline,config,dataAdapter}){
     const name=config.inputName||description.inputs[0]?.name;
     const tensor=baseline[name];
     if(!tensor||!['float32','float64'].includes(tensor.type))
      throw Error('Research plugin requires a declared floating-point input');
     if(!Array.isArray(config.indices)||!config.indices.length||config.indices.length>8192)
      throw Error('Provide an explicit bounded numeric input index list');
     const factor=config.factor;
     if(typeof factor!=='number'||!Number.isFinite(factor)||factor<0||factor>2)
      throw Error('Scaling factor must be finite and between 0 and 2');
     const indices=new Set();
     for(const i of config.indices){
      if(!Number.isSafeInteger(i)||i<0||i>=tensor.data.length||indices.has(i))
       throw Error('Indices must be unique and in actual tensor bounds');
      indices.add(i);
     }
     const edited=dataAdapter.prepare(description,baseline);
     let changed=0;
     for(const i of indices){
      const next=tensor.data[i]*factor;
      if(!Number.isFinite(next))throw Error('Numerical overflow');
      edited[name].data[i]=next;
      if(!Number.isFinite(edited[name].data[i]))throw Error('Float conversion overflow');
      if(edited[name].data[i]!==tensor.data[i])changed++;
     }
     return{baseline,edited,metadata:{
      kind:'research-numeric-input-scale',inputName:name,selected:indices.size,
      changed,factor,
      scientificStatus:'ENGINEERING ONLY; no validated counterfactual or biological mapping'
     }};
    }
   }});
  }
 });
});