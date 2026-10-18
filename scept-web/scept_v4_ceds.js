'use strict';
/* SCEPT Observatory v4: real-data CEDS, not a simulated or clinical explanation.
 * Builds real U10/U7 triple-forward studies on the same frozen ONNX session as
 * the v3 MRI/Token observatory. The CEDS chart uses only U7 evidence records.
 */
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root){root.SCEPT_CEDS=api;if(root.document)root.document.addEventListener('DOMContentLoaded',()=>api.mount(root));}
})(typeof window==='undefined'?null:window,function(){
  const DOSES=[0,.5,1],SEEDS=[109,211,307];
  const PROTOCOL=Object.freeze({strengths:DOSES,nuisanceSeeds:SEEDS,nuisanceSigma:.001,
    meanTolerance:.2,stdTolerance:.2,minTargetEffect:.01,minSpecificity:.005,
    minPositiveFraction:1,maxSpread:.02,minimumSignAgreement:1,doseTolerance:.005,
    nullTolerance:.00001,fill:0,outputScale:1});
  const GATE_LABELS={'paired-control-specificity':'目标相对对照特异性',
    'controlled-intervention-effect':'真实受控效应','replicate-specificity':'重复实验特异性',
    'nuisance-effect-stability':'干扰下效应稳定性','nuisance-sign-agreement':'效应方向一致性',
    'graded-sensitivity':'剂量响应单调性','zero-dose-null':'零剂量对照'};
  const avg=a=>a.reduce((x,y)=>x+y,0)/a.length;
  const med=a=>{const b=a.slice().sort((x,y)=>x-y),n=b.length;return n%2?b[(n-1)/2]:(b[n/2-1]+b[n/2])/2};
  const finite=x=>typeof x==='number'&&Number.isFinite(x);
  const fmt=x=>finite(x)?Number(x).toPrecision(4):'—';
  function summary(evidence){
    if(!evidence||evidence.schema!=='scept.u7.evidence.v1'||!Array.isArray(evidence.tasks))
      throw Error('仅接受 SCEPT U7 真实证据文件');
    const suite=evidence.suiteTests?.find(t=>t.plugin==='scept.u10.eval.preregistered-gates');
    if(!suite)throw Error('缺少 U10 注册判别结果');
    const tasks=evidence.tasks.filter(t=>t.status==='completed'&&t.explanation?.schema==='scept.u10.explanation.v1');
    if(tasks.length!==evidence.tasks.length||tasks.length<9)
      throw Error('U10 A/B/C 数据不完整，不能绘制完整证据谱');
    const doses=[...new Set(tasks.map(t=>t.explanation.strength))].sort((a,b)=>a-b);
    const seeds=[...new Set(tasks.map(t=>t.explanation.nuisanceSeed))].sort((a,b)=>a-b);
    if(doses.length<3||seeds.length<3||tasks.length!==doses.length*seeds.length)
      throw Error('独立种子×剂量矩阵不完整');
    const points=doses.map(q=>{
      const group=tasks.filter(t=>t.explanation.strength===q);
      if(group.length!==seeds.length||new Set(group.map(t=>t.explanation.nuisanceSeed)).size!==seeds.length)
        throw Error('缺少受控对照种子');
      const a=group.map(t=>t.explanation.targetedEffect),b=group.map(t=>t.explanation.matchedControlEffect),
        e=group.map(t=>t.explanation.controlledMagnitude),s=group.map(t=>t.explanation.score);
      if([...a,...b,...e,...s].some(x=>!finite(x)))throw Error('实测分数含无效数值');
      return{dose:q,p:med(a),q:med(b),e:med(e),specificity:med(s),
        pMin:Math.min(...a),pMax:Math.max(...a),qMin:Math.min(...b),qMax:Math.max(...b),
        raw:group.map(t=>({seed:t.explanation.nuisanceSeed,p:t.explanation.targetedEffect,
          q:t.explanation.matchedControlEffect,e:t.explanation.controlledMagnitude,
          specificity:t.explanation.score}))};
    });
    return{schema:'scept.v4.ceds.v1',scientificVerdict:suite.verdict,
      gates:suite.gates||[],reason:suite.reason||null,points,seeds,modelSha256:evidence.provenance?.modelSha256||null,
      caseCount:tasks.length,claims:evidence.claims||{},rawEvidence:evidence};
  }
  function selectedPatch(baseline,edited){
    if(baseline.length!==3*224*224||edited.length!==baseline.length)
      throw Error('仅能为当前 ViT 的真实 [1,3,224,224] 输入建立自动区域');
    const N=224*224,blocks=[],step=4;
    for(let y=0;y<224;y+=step)for(let x=0;x<224;x+=step){
      const ids=[],vals=[],modified=[];
      for(let c=0;c<3;c++)for(let j=0;j<4;j++)for(let i=0;i<4;i++){
        const k=c*N+(y+j)*224+x+i;
        ids.push(k);vals.push(baseline[k]);modified.push(Math.abs(edited[k]-baseline[k])>1e-7);
      }
      const mean=avg(vals),std=Math.sqrt(avg(vals.map(t=>(t-mean)*(t-mean))));
      blocks.push({x,y,ids,count:modified.filter(Boolean).length,mean,std});
    }
    const candidates=blocks.filter(b=>b.count>=3).sort((a,b)=>b.count-a.count||a.y-b.y||a.x-b.x);
    if(!candidates.length)throw Error('选定 MRI ROI 没有进入模型当前二维输入切片；请调整 MRI 切片或区域');
    const p=candidates[0];
    let best=null,bestError=Infinity;
    for(const q of blocks){
      if(q.count||Math.abs(q.x-p.x)<12&&Math.abs(q.y-p.y)<12)continue;
      const mu=Math.abs(p.mean-q.mean),sd=Math.abs(p.std-q.std);
      if(mu>.18||sd>.18)continue; // reserve margin for shared sigma=.001 nuisance
      const error=mu+sd+.000001*(Math.abs(p.x-q.x)+Math.abs(p.y-q.y));
      if(error<bestError){best=q;bestError=error}
    }
    if(!best)throw Error('找不到数值均值与标准差足够匹配的真实对照 Q；本次不能运行 U10');
    return{p,q:best,source:'actual-ROI-difference-selected-4x4-RGB-input-block',
      limits:'P 是实际受修改影响的模型输入 4×4 块；Q 仅通过数字强度/方差匹配，并非解剖学对照'};
  }
  function plot(canvas,data){
    if(!canvas||!data?.points?.length)return;
    const ratio=Math.min(2,globalThis.devicePixelRatio||1);
    const width=Math.max(360,Math.round(canvas.getBoundingClientRect().width||780)),height=280;
    canvas.width=Math.round(width*ratio);canvas.height=height*ratio;
    const g=canvas.getContext('2d');if(!g)return;
    g.setTransform(ratio,0,0,ratio,0,0);g.fillStyle='#071923';g.fillRect(0,0,width,height);
    const L=62,R=width-22,T=38,B=height-44,ys=data.points.flatMap(p=>[p.pMax,p.qMax,p.e]);
    const max=Math.max(.000001,...ys)*1.15;
    const X=x=>L+(R-L)*x,Y=y=>B-(B-T)*y/max;
    g.font='11px system-ui';g.textAlign='left';g.fillStyle='#a4c5cd';
    g.fillText('实际目标输出 |Δ| / 预注册 S₀',14,19);
    for(let i=0;i<=4;i++){const v=max*i/4,y=Y(v);g.strokeStyle='#284652';g.lineWidth=1;
      g.beginPath();g.moveTo(L,y);g.lineTo(R,y);g.stroke();g.fillStyle='#abc7ce';
      g.fillText(fmt(v),5,y+4)}
    function line(points,key,color){
      g.strokeStyle=color;g.lineWidth=2.8;g.beginPath();
      points.forEach((p,i)=>{const x=X(p.dose),y=Y(p[key]);i?g.lineTo(x,y):g.moveTo(x,y)});
      g.stroke();
      for(const p of points){const lo=Y(p[key+'Max']),hi=Y(p[key+'Min']),x=X(p.dose);
        g.strokeStyle=color;g.globalAlpha=.55;g.lineWidth=1.6;g.beginPath();g.moveTo(x,lo);g.lineTo(x,hi);g.stroke();
        g.beginPath();g.moveTo(x-4,lo);g.lineTo(x+4,lo);g.moveTo(x-4,hi);g.lineTo(x+4,hi);g.stroke();
        g.globalAlpha=1;g.fillStyle=color;g.beginPath();g.arc(x,Y(p[key]),4,0,Math.PI*2);g.fill();}
    }
    line(data.points,'p','#65ddc3');line(data.points,'q','#efae68');
    for(const p of data.points){g.fillStyle='#b3d3d6';g.textAlign='center';
      g.fillText(String(p.dose),X(p.dose),B+17)}
    g.fillStyle='#b3d3d6';g.fillText('干预剂量 q',width/2,height-8);
    g.textAlign='left';g.fillStyle='#65ddc3';g.fillText('● 目标 P',L+10,30);
    g.fillStyle='#efae68';g.fillText('● 对照 Q',L+90,30);
    g.fillStyle='#9db5c0';g.fillText('线=中位数 竖线=种子范围',Math.min(width-183,L+184),30);
    g.setTransform(1,0,0,1,0,0);
  }
  function mount(root){
    const $=id=>root.document.getElementById(id),btn=$('v4RunEvidence');
    if(!btn)return;
    let current=null,busy=false,engine=null,roi=null;
    const status=(s,error=false)=>{const el=$('v4Status');el.textContent=s;el.dataset.status=error?'error':'info'};
    function show(data){
      current=data;plot($('v4Signature'),data);
      const gateRoot=$('v4Gates');gateRoot.replaceChildren();
      const dict={criterion_met:'满足',criterion_not_met:'未满足',not_evaluable:'不可评价'};
      for(const gate of data.gates){
        const row=root.document.createElement('div');row.className='v4-gate';
        const name=root.document.createElement('span');name.textContent=GATE_LABELS[gate.name]||gate.name;
        const v=root.document.createElement('strong');v.dataset.verdict=gate.verdict;
        v.textContent=(dict[gate.verdict]||'未评价')+' · '+fmt(gate.value);
        row.append(name,v);gateRoot.append(row);
      }
      const lroot=$('v4LayerEvidence');lroot.replaceChildren();
      const full=data.rawEvidence.tasks.filter(t=>t.status==='completed'&&t.explanation?.strength===1);
      const representative=full[0]?.observations||[];
      for(const obs of representative){
        if(obs.tensor===data.rawEvidence.experiment?.manifest?.target?.tensor)continue;
        const a=full.map(t=>t.observations?.find(o=>o.tensor===obs.tensor)?.mean_abs_delta).filter(finite);
        const c=full.map(t=>t.controlObservations?.find(o=>o.tensor===obs.tensor)?.mean_abs_delta).filter(finite);
        if(a.length!==full.length||c.length!==full.length)continue;
        const row=root.document.createElement('div');row.className='v4-gate';
        const label=root.document.createElement('span');label.textContent=obs.tensor.split('/').slice(-2).join('/')||obs.tensor;
        const value=root.document.createElement('strong');value.textContent='P '+fmt(med(a))+' / Q '+fmt(med(c));
        value.style.color='#9ce5d8';row.append(label,value);lroot.append(row);
      }
      const verdict=data.scientificVerdict;
      $('v4Verdict').textContent=verdict==='criterion_met'?'计算判别条件满足（不等于医学因果证明）':
        verdict==='criterion_not_met'?'至少一项预注册条件未满足':'无法评价：'+(data.reason||'证据不足');
      $('v4Verdict').dataset.verdict=verdict;
      $('v4Metrics').textContent='真实 ONNX A/B/C 任务 '+data.caseCount+' 项 · 干扰种子 '+data.seeds.length+
        ' 个 · 指定目标输出分量 · 模型 SHA-256 '+(data.modelSha256||'未知').slice(0,16)+'…';
      const last=data.points[data.points.length-1];
      $('v4Current').textContent='满剂量中位数：目标 '+fmt(last.p)+' · 对照 '+fmt(last.q)+
        ' · 特异性 '+fmt(last.specificity)+' · 受控效应 '+fmt(last.e);
      $('v4ExportEvidence').disabled=false;
    }
    $('v4ShowPlanes').addEventListener('click',()=>root.document.body.classList.toggle('show-planes'));
    btn.addEventListener('click',async()=>{
      if(busy)return;
      busy=true;btn.disabled=true;$('v4ExportEvidence').disabled=true;
      $('v4Gates').replaceChildren();$('v4Verdict').textContent='未评价';
      current=null;roi=null;
      try{
        if($('modelMode').value!=='vit')throw Error('当前单击式科学解释仅支持已验证接入的 ViT；MRI CNN 暂无内部观测与匹配方案');
        const ctx=root.SCEPT_V3?.getScientificContext?.();
        if(!ctx?.session||!root.SCEPT_MRI?.state?.loaded)throw Error('请先确认公开预训练模型与真实 MRI 已加载');
        const baseline=root.SCEPT_MRI.makeSlice(false,2),edited=root.SCEPT_MRI.makeSlice(true,2);
        roi=selectedPatch(baseline,edited);
        $('v4Roi').textContent='P: 输入块 ('+roi.p.x+','+roi.p.y+') / Q: 匹配块 ('+
          roi.q.x+','+roi.q.y+') · 每块 4×4×3 个模型输入数值；仅数值匹配，不代表解剖对照';
        status('正在从相同冻结 ViT 获得真实基线输出…');
        const dims=[1,3,224,224],input='pixel_values';
        const forward=async feeds=>{
          const m={};
          for(const [name,t]of Object.entries(feeds))m[name]=new root.ort.Tensor(t.type,t.data,t.dims);
          return ctx.session.run(m);
        };
        const a=await forward({[input]:{type:'float32',dims,data:baseline}});
        const outputs=Object.keys(a).map(name=>({name,type:a[name].type,shape:[...a[name].dims]}));
        const logits=outputs.find(t=>t.name==='logits')||outputs.find(t=>t.name.toLowerCase().includes('logit'));
        if(!logits||!a[logits.name]?.data?.length)
          throw Error('该 ViT 没有公开可确认的 logits 输出；不能猜测模型目标');
        const outputValues=a[logits.name].data;
        let targetIndex=0;
        for(let i=1;i<outputValues.length;i++)if(outputValues[i]>outputValues[targetIndex])targetIndex=i;
        $('v4Target').textContent='目标：ViT 原始 logits['+targetIndex+']（模型最大输出索引；不是 MRI 诊断类别）';
        const hash=async ids=>root.SCEPT_CORE.sha256(Uint32Array.from(ids));
        const mask=async(block,id)=>({inputName:input,shape:dims,
          indices:block.ids,indicesSha256:await hash(block.ids),
          sourceId:'v4-numeric-matched-'+id+'-'+root.SCEPT_MRI.state.revision,
          coordinateSpace:'model-input-flat-indices'});
        const p=await mask(roi.p,'target'),q=await mask(roi.q,'control');
        const registry=root.SCEPT_CORE.createRegistry();
        const data=root.SCEPT_DATA_ADAPTER.make(root.SCEPT_CORE);
        const intervention=root.SCEPT_INTERVENTION.make(registry,root.SCEPT_CORE,data);
        intervention.registerBuiltins();
        root.SCEPT_XAI.registerBuiltins(registry,root.SCEPT_CORE);
        root.SCEPT_U10_MECHANISM.install({sdk:root.SCEPT_SDK,registry,core:root.SCEPT_CORE});
        const observation=root.SCEPT_OBSERVATION.make();
        const session={sha256:ctx.sha256,manifest:{modelId:'v4-pretrained-vit',probes:ctx.probes.map(t=>t.name)},
          description:{inputs:[{name:input,type:'float32',shape:dims}],outputs,
            capabilities:['inference','observations.output','observations.intermediate']},
          run:forward};
        engine=root.SCEPT_XAI.make({registry,session,interventionEngine:intervention,
          observationAdapter:observation,dataAdapter:data,core:root.SCEPT_CORE,
          schedulerFactory:root.SCEPT_SCHEDULER,onProgress:s=>{
            const done=s.completed||0,total=s.total||0,failed=s.failed||0;
            status('U10 受控真实推理：'+done+'/'+total+' 完成'+(failed?' · '+failed+' 失败':''));
          }});
        const manifest=root.SCEPT_U10_MECHANISM.plan({...PROTOCOL,
          targetMask:p,controlMask:q,inputName:input,shape:dims,
          targetTensor:logits.name,targetIndex,modelSha256:ctx.sha256,profileId:'generic'});
        manifest.observation.tensors=[logits.name,...ctx.probes.map(t=>t.name).filter(n=>n!==logits.name)].slice(0,8);
        engine.registerExperiment(manifest);
        await engine.prepare(manifest.id,{feeds:{[input]:{type:'float32',dims,data:baseline}}});
        const raw=await engine.run();
        if(root.SCEPT_MRI.state.revision!==ctx.mriRevision)
          throw Error('运行中 MRI 被修改；已完成旧版输入实验但不绑定到新 MRI');
        show(summary(raw));
        status('真实受控证据谱已生成。三维点云的普通 A/B 视图与本次 A/B/C 解释实验属于不同实验，请勿混同。');
      }catch(e){status('无法完成本次科学检验：'+e.message,true);$('v4Verdict').textContent='不可评价';$('v4Verdict').dataset.verdict='not_evaluable'}
      finally{busy=false;btn.disabled=false}
    });
    $('v4ExportEvidence').addEventListener('click',()=>{
      if(!current)return;
      const text=JSON.stringify({ceds:{schema:current.schema,points:current.points,gates:current.gates,
        scientificVerdict:current.scientificVerdict,caseCount:current.caseCount,seeds:current.seeds},
        roi:{target:{x:roi.p.x,y:roi.p.y},control:{x:roi.q.x,y:roi.q.y},source:roi.source,limits:roi.limits},
        source:current.rawEvidence,interpretation:'Numeric, cross-domain nonclinical SCEPT U10 evidence; not medical or causal validation'},null,2);
      const u=URL.createObjectURL(new Blob([text],{type:'application/json'})),link=root.document.createElement('a');
      link.href=u;link.download='scept-v4-ceds-real-evidence.json';link.click();
      setTimeout(()=>URL.revokeObjectURL(u),1500);
    });
    root.addEventListener('resize',()=>{if(current)plot($('v4Signature'),current)});
    $('v4Verdict').textContent='尚未检验';
    status('选择 MRI 干预区域，然后点击「检验 SCEPT 解释」。');
  }
  return Object.freeze({summary,selectedPatch,plot,mount,PROTOCOL});
});