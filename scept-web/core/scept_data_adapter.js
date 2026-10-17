'use strict';
/* U3 explicit typed input adapter; never guesses biomedical preprocessing. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.SCEPT_DATA_ADAPTER=api;})(typeof window!=='undefined'?window:null,function(){
 const types={float32:Float32Array,float64:Float64Array,int32:Int32Array,int64:BigInt64Array,uint8:Uint8Array,bool:Uint8Array};
 const fail=s=>{throw Error('SCEPT data adapter: '+s)};
 function prepare(description,feeds,core){
  if(!description?.inputs?.length||!feeds||typeof feeds!=='object'||Array.isArray(feeds)||Object.keys(feeds).length!==description.inputs.length)fail('Named input signature mismatch');
  const result={};
  for(const m of description.inputs){
   const t=feeds[m.name],C=types[m.type];if(!t||!C||t.type!==m.type||!(t.data instanceof C))fail('Expected '+m.type+' typed input for '+m.name);
   core.validateTensor(t,m);
   result[m.name]={type:t.type,dims:t.dims.slice(),data:new C(t.data)};
  }
  if(Object.keys(feeds).some(name=>!Object.hasOwn(result,name)))fail('Undeclared input');
  return result;
 }
 function offset(prepared,description,{inputName=null,delta=0,fraction=.01}={}){
  if(!Number.isFinite(delta)||!Number.isFinite(fraction)||fraction<=0||fraction>1||!description?.inputs?.length)fail('Invalid intervention');
  const name=inputName||description.inputs[0].name;
  if(!Object.hasOwn(prepared,name))fail('Unknown input '+name);
  const edited={};for(const [key,t]of Object.entries(prepared))edited[key]={type:t.type,dims:t.dims.slice(),data:new t.data.constructor(t.data)};
  const t=edited[name];if(!['float32','float64'].includes(t.type))fail('Numeric offset requires floating point input');
  const elementCount=Math.max(1,Math.floor(t.data.length*fraction));
  for(let i=0;i<elementCount;i++){const next=t.data[i]+delta;if(!Number.isFinite(next))fail('Nonfinite edit');t.data[i]=next}
  return {edited,intervention:{kind:'engineering-numeric-offset',inputName:name,delta,elementCount,semantics:'Not a medically valid MRI perturbation'}};
 }
 function make(core){if(typeof core?.validateTensor!=='function')fail('SCEPT_CORE required');
  return Object.freeze({kind:'data',id:'typed-tensor-v1',capabilities:['data.preprocess'],prepare:(d,f)=>prepare(d,f,core),offset})}
 return Object.freeze({make,prepare,offset});
});