'use strict';
/* Browser-local U9 SDK/audit/replay UX; never loads or evals untrusted JS.
 * Requires explicit re-upload of model and original typed data after reload.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_M3_CONSOLE=api;})(typeof window!=='undefined'?window:null,function(){
 function make({core,registry,xai,schedulerFactory,interventionEngine,observationAdapter,
  dataAdapter,getSession,getFeeds,getU6,getU7,getU6Definition,getU7Definition,getReference}){
  const byId=id=>document.getElementById(id);
  let imported=null,lastReport=null,working=false;
  const status=(s,type='pending')=>{byId('u9Status').textContent=s;byId('u9Status').dataset.state=type};
  const show=data=>{byId('u9Details').textContent=JSON.stringify(data,null,2).slice(0,10000)};
  function download(obj,filename){
   const content=JSON.stringify(obj,null,2);
   const blob=new Blob([content],{type:'application/json'});
   const url=URL.createObjectURL(blob),a=document.createElement('a');
   a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();
   setTimeout(()=>URL.revokeObjectURL(url),10000);
  }
  function reference(definition){
   const uses=definition?.intervention?.pluginId==='u5.replace'||
    definition?.pluginId==='u5.replace'||
    definition?.controls?.some(t=>t.pluginId==='u5.replace')||
    definition?.intervention?.controls?.some(t=>t.pluginId==='u5.replace');
   return uses?getReference():null;
  }
  function refresh(){
   const session=getSession(),available=!!session&&!working;
   for(const id of ['u9Compatibility','u9GenerateTemplate','u9Benchmark'])
    byId(id).disabled=!available;
   const readyU6=!!getU6()&&getU6().snapshot()?.status==='completed';
   const readyU7=!!getU7()&&getU7().snapshot()?.status==='completed';
   byId('u9ExportU6').disabled=!available||!readyU6||!getU6Definition();
   byId('u9ExportU7').disabled=!available||!readyU7||!getU7Definition();
   byId('u9Verify').disabled=!available||!imported;
   byId('u9Replay').disabled=!available||!imported;
  }
  async function wrap(label,fn){
   if(working)return;working=true;refresh();
   status(label);
   try{lastReport=await fn();show(lastReport);status(label+'：完成','success');return lastReport}
   catch(e){lastReport={code:e.code||'U9_ERROR',message:e.message};show(lastReport);status(label+'：'+e.message,'error');return null}
   finally{working=false;refresh()}
  }
  byId('u9Compatibility').onclick=()=>wrap('兼容性预检',async()=>{
   const m=JSON.parse(byId('u7Manifest').value);
   const result=SCEPT_SDK.compatibility({registry,core,xai,session:getSession(),manifest:m});
   return result;
  });
  byId('u9GenerateTemplate').onclick=()=>wrap('生成模型对应插件模板',async()=>{
   const s=getSession(),out=s.description.outputs[0],input=s.description.inputs[0],
    manifest=SCEPT_SDK.exampleManifest({modelSha256:s.sha256,modelId:s.manifest?.modelId,
      inputName:input?.name,targetName:out?.name,
      probeName:s.manifest?.probes?.[0]||out.name});
   byId('u7Manifest').value=JSON.stringify(manifest,null,2);
   download(manifest,'scept-u9-experiment-template.json');
   return{status:'generated',modelSha256:s.sha256,experimentId:manifest.id,
    warning:'Template requires real valid model feeds, intervention config, and scientifically justified methods'};
  });
  byId('u9Benchmark').onclick=()=>wrap('真实推理性能测试',async()=>{
   const feeds=getFeeds().feeds,report=await SCEPT_PERFORMANCE.benchmark({
    session:getSession(),feeds,runs:3,warmup:1,maxInputMiB:128});
   return report;
  });
  byId('u9ExportU6').onclick=()=>wrap('生成 U6 可核查审计',async()=>{
   const engine=getU6();
   const record=engine.exportAudit(),definition=getU6Definition();
   const bundle=await SCEPT_AUDIT.create({core,record,definition});
   download(bundle,'scept-u9-u6-audit.json');
   byId('u9Definition').value=JSON.stringify(definition,null,2);
   return{schema:bundle.schema,kind:bundle.kind,contentSha256:bundle.contentSha256,
    modelSha256:bundle.modelSha256,definitionSha256:bundle.definitionSha256,
    tasks:bundle.record.tasks.length};
  });
  byId('u9ExportU7').onclick=()=>wrap('生成 U7 可核查审计',async()=>{
   const engine=getU7();
   const record=engine.exportEvidence(),definition=getU7Definition();
   const bundle=await SCEPT_AUDIT.create({core,record,definition});
   download(bundle,'scept-u9-u7-audit.json');
   byId('u9Definition').value=JSON.stringify(definition,null,2);
   return{schema:bundle.schema,kind:bundle.kind,contentSha256:bundle.contentSha256,
    modelSha256:bundle.modelSha256,definitionSha256:bundle.definitionSha256,
    tasks:bundle.record.tasks.length};
  });
  byId('u9BundleFile').onchange=async()=>{
   try{
    const file=byId('u9BundleFile').files?.[0];if(!file)return;
    if(file.size>6000000)throw Error('Audit file exceeds 6 MB bound');
    imported=JSON.parse(await file.text());
    // Never treat the imported content as executable code.
    SCEPT_AUDIT.normalize(imported);
    if(imported.schema!=='scept.u9.audit.v1')throw Error('Unsupported audit schema');
    status('已本地导入审计。提供原始实验方案并重新导入相同模型与输入后验证。','success');
   }catch(e){imported=null;status('审计文件拒绝：'+e.message,'error')}
   refresh();
  };
  const verify=()=>SCEPT_AUDIT.verify({core,bundle:imported,
   modelSha256:getSession()?.sha256,feeds:getFeeds().feeds,
   referenceFeeds:reference(JSON.parse(byId('u9Definition').value)),
   definition:JSON.parse(byId('u9Definition').value)});
  byId('u9Verify').onclick=()=>wrap('哈希/模型/输入完整性核验',verify);
  byId('u9Replay').onclick=()=>wrap('真实 ONNX 重放对照',async()=>{
   const session=getSession(),def=JSON.parse(byId('u9Definition').value),
    feeds=getFeeds().feeds,referenceFeeds=reference(def);
   const runner=imported.kind==='u7'?
    xai.make({registry,session,interventionEngine,observationAdapter,
     dataAdapter,core,schedulerFactory}):
    schedulerFactory.make({session,interventionEngine,observationAdapter,
     dataAdapter,core});
   const report=await SCEPT_AUDIT.replay({core,bundle:imported,session,feeds,
    referenceFeeds,definition:def,engine:runner});
   return report;
  });
  refresh();
  return Object.freeze({refresh,getSnapshot:()=>({imported:!!imported,kind:imported?.kind||null,
   working,lastReport})});
 }
 return Object.freeze({VERSION:'1.0.0',make});
});