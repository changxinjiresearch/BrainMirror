'use strict';
// Independent actual installed U10 JS SDK intervention parity fixture.
// Node writes genuine Uint32-mask SHA to ensure Python U11 is not silently
// benchmarking a different Gaussian nuisance and ROI fill operator.
const CORE=require('../scept-web/core/scept_core.js');
const SDK=require('../scept-web/core/scept_sdk.js');
const U10=require('../scept-web/plugins/u10_evidential_dependence.js');
(async()=>{
 const reg=CORE.createRegistry();
 U10.install({sdk:SDK,registry:reg,core:CORE});
 const method=reg.requireAdapter('intervention',U10.IDs.intervention);
 const a=Float32Array.from([1,1,1,1,1,1,1,1]);
 const mk=async(indices,sourceId)=>({
  inputName:'x',shape:[1,8],indices,
  indicesSha256:await CORE.sha256(Uint32Array.from(indices)),
  coordinateSpace:'model-input-flat-indices',sourceId
 });
 const targetMask=await mk([0,1],'numeric-target-not-anatomy');
 const controlMask=await mk([6,7],'numeric-control-not-anatomy');
 const desc={inputs:[{name:'x',type:'float32',shape:[1,8]}]};
 const baseline={x:{type:'float32',dims:[1,8],data:a}};
 const results=[];
 for(const dose of [0,.5,1]){
  const config={inputName:'x',fill:0,strength:dose,nuisanceSeed:109,
   nuisanceSigma:.001,meanTolerance:.2,stdTolerance:.2,targetMask,controlMask};
  const out=await method.apply({description:desc,baseline,config});
  results.push({dose,A:Array.from(out.baseline.x.data),
   B:Array.from(out.edited.x.data),C:Array.from(out.control.x.data),
   meanDiff:out.metadata.meanDiff,stdDiff:out.metadata.stdDiff,
   targetMaskSha256:out.metadata.targetMaskSha256,
   controlMaskSha256:out.metadata.controlMaskSha256});
 }
 if(a.some(x=>x!==1))throw Error('U10 extension mutated original user input');
 process.stdout.write(JSON.stringify({schema:'scept.u11.python-js-parity.v1',
   original:[...a],nuisanceSeed:109,sigma:.001,results})+'\n');
})().catch(e=>{console.error(e);process.exitCode=1});
