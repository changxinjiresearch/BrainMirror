'use strict';
/* V6.5 practical research workspace. All interactivity is functional:
 * model route -> genuine V6 one-run engine, history -> real Run Records;
 * protocol cards navigate to existing, explicitly un-migrated U10/V5 lab.
 */
(function(root){
 if(!root?.document)return;
 root.document.addEventListener('DOMContentLoaded',()=>{
  const $=id=>document.getElementById(id);
  if(!$('v65Workspace')||!root.SCEPT_V6||!root.SCEPT_WORKSPACE_JOURNAL)return;
  const journal=root.SCEPT_WORKSPACE_JOURNAL.make({maxRuns:32});
  let selected=null;
  const fmt=s=>String(s||'');
  const put=(id,value)=>{const el=$(id);if(el)el.textContent=fmt(value)};
  const status=(s,error=false)=>{put('v65Status',s);$('v65Status').dataset.error=String(error)};
  const short=s=>typeof s==='string'?s.slice(0,13):'—';
  const anchor=(href,label,desc,state)=>{
   const a=document.createElement('a');a.href=href;a.className='v65-route';
   const title=document.createElement('strong');title.textContent=label;
   const p=document.createElement('span');p.textContent=desc;
   const b=document.createElement('em');b.textContent=state;
   a.append(title,p,b);return a;
  };
  const catalog=[
   {name:'单次观测 · NAF',description:'当前 V6.3 引擎：一次冻结模型前向运行生成独立活动记录',href:'#v6Studio',state:'已接入单次运行'},
   {name:'模型与数据 · MRI 输入',description:'本页当前预置的真实 NIfTI / ViT / CNN 适配器',href:'#v65Input',state:'已验证两个预训练架构'},
   {name:'任意 ONNX 接入',description:'现有 Universal Core 研究控制台：ONNX 导入、真实探针和受控研究',href:'./universal.html',state:'高级旧接口；未接入 V6.1 单次记录'},
   {name:'受控输入干预 · NDF / CRC',description:'V5 MRI A/B 与多反事实证据工作台',href:'./v5.html',state:'独立旧协议；V6.4 统一入口未完成'},
   {name:'CEDS / U10 方法',description:'受控目标 P 与匹配对照 Q 的证据依赖实验',href:'./v4.html',state:'独立旧协议；未迁移 V6.4'},
   {name:'实验室 / 模型适配',description:'读取真实 ONNX 算子、适配说明和协议调试',href:'./lab.html',state:'高级工程入口'}
  ];
  const routes=$('v65Protocols');routes.replaceChildren();
  catalog.forEach(c=>routes.append(anchor(c.href,c.name,c.description,c.state)));
  const nav=$('v65Nav');if(nav)nav.onclick=e=>{
   const a=e.target.closest('a[href^="#"]');if(!a)return;
   const element=document.querySelector(a.getAttribute('href'));if(element)element.scrollIntoView({behavior:'smooth',block:'start'});
  };
  function render(){
   const items=journal.list();
   put('v65RunCount',items.length+' 份真实或已标明外部来源的运行摘要（最多32份，本页会话内存）');
   const host=$('v65Runs');host.replaceChildren();
   if(!items.length){
    const p=document.createElement('p');p.className='v65-muted';p.textContent='暂无记录。点击上方「单次运行并观察」生成你的第一份真实 NAF。';host.append(p);
    put('v65Details','请先生成或导入一份记录。');return;
   }
   if(!selected||!journal.get(selected))selected=items[items.length-1].runId;
   for(const [i,item] of items.entries()){
    const button=document.createElement('button');button.type='button';button.className='v65-run-row';
    button.setAttribute('aria-pressed',String(selected===item.runId));
    const main=document.createElement('strong');main.textContent=
      (i+1)+'. '+item.profile+' · '+item.measuredUnits+' units';
    const meta=document.createElement('span');meta.textContent=
      item.source==='imported_unverified'?'外部导入 · 未验证真实性':'本页真实采集 · 仅摘要保留';
    const sub=document.createElement('small');sub.textContent='Model '+short(item.modelSha256)+' · input '+short(item.inputSha256);
    button.append(main,meta,sub);
    button.onclick=()=>{selected=item.runId;render()};
    host.append(button);
   }
   const entry=journal.get(selected),summary=items.find(r=>r.runId===selected);
   if(!entry||!summary)return;
   const measured=entry.naf.groups.filter(g=>g.status==='measured');
   put('v65Details',
    'Run ID: '+entry.runId+'\n来源: '+(summary.source==='live_runtime'?
      '本页实际运行产生的摘要；原始张量仅保留最新一次':'外部导入的用户自述记录；无法据此验证源模型真的执行')+
    '\n模型: '+summary.model+'\n模型 SHA-256: '+summary.modelSha256+
    '\n输入 SHA-256: '+Object.values(entry.record.provenance.inputs).join(', ')+
    '\n任务/单位: '+summary.profile+'\n实测张量: '+measured.length+
    ' · 记录单位: '+summary.measuredUnits+
    '\n输出：'+entry.record.outputs.map(v=>v.name+' ['+v.shape.join(',')+']').join('; ')+
    '\n更新观测：'+summary.updateStatus.join(', ')+
    '\n科学声明：仅活动观测。因果性、稳定性及诊断效度均未建立。');
   $('v65Note').value=summary.note;
   $('v65SelectedId').textContent=short(entry.runId);
   $('v65ExportSelected').disabled=false;
   $('v65SaveNote').disabled=false;
   $('v65Remove').disabled=false;
  }
  function download(manifest,name){
   const file=JSON.stringify(manifest,null,2);
   const url=URL.createObjectURL(new Blob([file],{type:'application/json;charset=utf-8'}));
   const a=document.createElement('a');a.href=url;a.download=name;
   document.body.append(a);a.click();a.remove();
   root.setTimeout(()=>URL.revokeObjectURL(url),2500);
  }
  root.SCEPT_V6.onRunCapture(({record,naf})=>{
   try{const summary=journal.addLive(record,naf);selected=summary.runId;render();
    status('已把真实单次运行追加至本地研究记录。建议导出 JSON 备份；页面关闭后摘要会丢失。')}
   catch(e){status('真实 NAF 已生成，但研究记录未能保存：'+e.message,true)}
  });
  $('v65Project').onchange=e=>{
   try{journal.setProject(e.target.value);status('项目名称已更新。运行记录仍只保存在本页内存。')}
   catch(err){status(err.message,true)}
  };
  $('v65ExportAll').onclick=()=>{
   try{download(journal.exportManifest(),'scept-v65-workspace-journal.json');
    status('已导出当前科学运行摘要 JSON（不包含原始 MRI 与张量）。')}
   catch(e){status('导出失败：'+e.message,true)}
  };
  $('v65ExportSelected').onclick=()=>{
   try{download(journal.exportManifest([selected]),'scept-v65-single-run.json');
    status('已导出选中的一份运行记录。')}
   catch(e){status('导出失败：'+e.message,true)}
  };
  $('v65Import').onchange=async e=>{
   const file=e.target.files?.[0];if(!file)return;
   try{
    if(file.size>8*1024*1024)throw Error('导入 JSON 最大 8 MB');
    const text=await file.text(),data=JSON.parse(text),count=journal.importManifest(data);
    selected=journal.list().at(-1).runId;render();
    status('已导入 '+count+' 份摘要，全部标为「外部导入、真实性未验证」。不会运行模型或自动恢复原始张量。')
   }catch(err){status('导入拒绝：'+err.message,true)}
   finally{e.target.value=''}
  };
  $('v65SaveNote').onclick=()=>{
   try{journal.setNote(selected,$('v65Note').value);render();status('研究笔记已更新；导出工作区 JSON 才能永久备份。')}
   catch(e){status(e.message,true)}
  };
  $('v65Remove').onclick=()=>{
   if(!selected)return;
   try{journal.remove(selected);selected=null;render();status('已从本页研究记录中删除所选摘要，不影响模型或已下载文件。')}
   catch(e){status(e.message,true)}
  };
  $('v65Clear').onclick=()=>{
   if(!root.confirm('清空本页全部研究运行摘要？尚未导出的内容将无法恢复。'))return;
   journal.clear();selected=null;render();status('本页运行摘要已清空。')
  };
  $('v65OpenInput').onclick=()=>document.querySelector('#v65Input').scrollIntoView({behavior:'smooth'});
  // Keep both current native adapters visible; a third model is not falsely
  // described as captured by captureOnce until future generic adapter work.
  const adapter=$('v65Adapter');
  adapter.onchange=()=>{
   if(adapter.value==='generic'){status('任意 ONNX 目前需要使用已存在的高级研究控制台；V6.4 单次记录统一尚未完成。');
    const a=$('v65Generic');a.hidden=false;
   }else{
    $('v65Generic').hidden=true;$('v6Model').value=adapter.value;
    status('已选择真实模型适配器 '+adapter.value+'；下方运行 NAF 不要求 A/B。');
   }
  };
  $('v6Model').addEventListener('change',()=>{if(adapter.value!=='generic')adapter.value=$('v6Model').value});
  root.SCEPT_V65={getSnapshot:()=>({runs:journal.list(),project:journal.projectName(),
   selected,protocolCount:catalog.length,unifiedComparisonReady:false}),
   getManifest:()=>journal.list().length?journal.exportManifest():null};
  render();status('通用研究工作台就绪。单次观察可立即使用；NDF/CRC/ICM 与 U10 仍保留独立旧入口。');
 });
})(typeof window==='undefined'?null:window);
