'use strict';
/* U10 controlled experiment workbench. Creates a real U7 JSON manifest from
 * validated user-specified numeric masks; never infers clinical ROI semantics.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_U10_CONSOLE=api;})(typeof window!=='undefined'?window:null,function(){
 function make({core,getSession,getFeeds}){
  const $=id=>document.getElementById(id);
  const status=(s,type='pending')=>{const el=$('u10Status');el.textContent=s;el.dataset.state=type};
  const field=$('u10Protocol'),report=$('u10Report');
  function refresh(){
   $('u10Generate').disabled=!getSession();
  }
  async function manifest(){
   const session=getSession();if(!session)throw Error('请先加载真实 ONNX 模型');
   const raw=JSON.parse(field.value||'{}');
   if(!raw||Array.isArray(raw)||typeof raw!=='object')throw Error('必须提供实验配置对象');
   const feeds=getFeeds().feeds,inputName=raw.inputName||session.description.inputs[0]?.name;
   const t=feeds[inputName],out=raw.targetTensor;
   if(!t||!session.description.inputs.some(i=>i.name===inputName))throw Error('请选择实际模型输入张量');
   if(typeof out!=='string'||!session.description.outputs.some(t=>t.name===out))
    throw Error('必须明确选择模型真实导出的目标输出名称');
   const getMask=async(side)=>{
    const indices=raw[side+'Indices'];
    if(!Array.isArray(indices)||!indices.length||indices.length>100000||
     indices.some((x,i)=>!Number.isSafeInteger(x)||x<0||x>=t.data.length||(i>0&&x<=indices[i-1])))
     throw Error(side+' ROI 必须是排序、无重复、在真实模型输入范围内的非空索引');
    const sourceId=raw[side+'SourceId'];
    if(typeof sourceId!=='string'||!sourceId.trim())throw Error(side+' ROI 必须声明来源');
    return{inputName,shape:t.dims.slice(),indices:indices.slice(),
     indicesSha256:await core.sha256(Uint32Array.from(indices)),
     sourceId,coordinateSpace:'model-input-flat-indices'};
   };
   const targetMask=await getMask('target'),controlMask=await getMask('control');
   const args={...raw,inputName,shape:t.dims.slice(),targetMask,controlMask,
    targetTensor:out,modelSha256:session.sha256,
    profileId:document.getElementById('architectureProfile').value};
   return SCEPT_U10_MECHANISM.plan(args);
  }
  $('u10Generate').onclick=async()=>{
   $('u10Generate').disabled=true;
   try{
    const m=await manifest();
    $('u7Manifest').value=JSON.stringify(m,null,2);
    $('u7Manifest').dispatchEvent(new Event('input'));
    const tests=m.evaluations[1].params;
    report.textContent='U10 已在 U7 编辑框中生成可复现预注册的 '+m.intervention.controls.length+
     ' 项 A/B/C 任务。\n模型 SHA-256: '+m.model.sha256+
     '\n噪声种子: '+tests.expectedNuisanceSeeds.join(',')+
     '\n强度: '+tests.expectedStrengths.join(',')+
     '\n解释算法: '+m.method.adapterId+
     '\n7 个科学判别门槛均需满足才能得到 criterion_met；仍不代表医学/因果有效性。';
    status('U10 实验方案已生成。请在 U7 点击「注册插件 → 验证并调度 → 执行 XAI 实验」。','success');
   }catch(e){status('生成失败：'+e.message,'error');report.textContent=e.message}
   finally{refresh()}
  };
  function render(evidence){
   const result=evidence?.suiteTests?.find(t=>t.plugin===SCEPT_U10_MECHANISM.IDs.suite);
   if(!result)return;
   const lines=['SCEPT U10 · 预注册受控证据判别',
    '计算判别：'+result.verdict+'；这不是临床验证或因果证明。',
    '真实实验任务：'+(result.nTasks||0)+'；重复扰动种子：'+(result.nReplicates||0)];
   if(result.reason)lines.push('不可评价原因：'+result.reason);
   for(const gate of result.gates||[])lines.push(gate.verdict+' · '+gate.name+
    '：'+Number(gate.value).toPrecision(5)+' / 门槛 '+gate.threshold);
   lines.push('证据来源：已执行 ONNX A/B/C 输出；分数仅表示预设的数值差异与对照判别。');
   report.textContent=lines.join('\n');
   status('真实 U10 任务完成；总体判别：'+result.verdict,
    result.verdict==='criterion_met'?'success':result.verdict==='not_evaluable'?'pending':'error');
  }
  refresh();
  return Object.freeze({manifest,refresh,render});
 }
 return Object.freeze({make});
});