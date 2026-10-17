'use strict';
/* SCEPT U6: deterministic, serial, browser-local experiment task scheduler.
 * Executes real U5 interventions and U4 observations on the same frozen session.
 * No simulated results, remote service, model training, or medical conclusions.
 */
(function(root,factory){
 const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SCEPT_SCHEDULER=api;
})(typeof window!=='undefined'?window:null,function(){
 const VERSION='1.0.0-u6',MAX_TASKS=128;
 const fail=(code,msg)=>{const e=Error('SCEPT U6: '+msg);e.code=code;throw e};
 const plain=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const uint32=(x,label)=>{if(!Number.isInteger(x)||x<0||x>0xffffffff)fail('INVALID_PLAN',label+' must be a uint32');return x};
 const cloneJson=x=>{try{const text=JSON.stringify(x);if(!text||text.length>1000000)fail('RESOURCE_LIMIT','Plan JSON too large');const y=JSON.parse(text);return y}
  catch(e){if(e.code)throw e;fail('INVALID_PLAN','Plan must contain finite JSON-serializable data')}};
 function checkJson(x){const scan=(v,depth)=>{
   if(depth>20)fail('INVALID_PLAN','Plan nesting too deep');
   if(typeof v==='number'&&!Number.isFinite(v))fail('INVALID_PLAN','NaN or Infinity is not a valid parameter');
   if(v===undefined||typeof v==='function'||typeof v==='bigint')fail('INVALID_PLAN','Non-JSON plan parameter');
   if(Array.isArray(v))v.forEach(z=>scan(z,depth+1));
   else if(plain(v))Object.values(v).forEach(z=>scan(z,depth+1));
  };scan(x,0)}
 function stableSeed(seed,index){
  // Deterministic, independent of execution order and failure/retries.
  let x=(seed+(index+1)*0x9e3779b9)>>>0;
  x^=x>>>16;x=Math.imul(x,0x85ebca6b);x^=x>>>13;x=Math.imul(x,0xc2b2ae35);x^=x>>>16;
  return x>>>0;
 }
 function task(id,variant,pluginId,config,seed,intensity=null){
  const c=cloneJson(config);
  if(pluginId==='u5.noise')c.seed=seed;
  return{id,variant,pluginId,config:c,seed, intensity,status:'pending',attempts:0,
    error:null,result:null,startedAt:null,finishedAt:null};
 }
 function plan(raw){
  if(!plain(raw))fail('INVALID_PLAN','Experiment plan object required');
  checkJson(raw);
  const mode=raw.mode||'ab',pluginId=raw.pluginId;
  if(!['ab','sweep','batch'].includes(mode))fail('INVALID_PLAN','mode must be ab / sweep / batch');
  if(typeof pluginId!=='string'||!pluginId.trim())fail('INVALID_PLAN','pluginId required');
  const config=cloneJson(raw.config||{}),baseSeed=uint32(raw.baseSeed??240216,'baseSeed');
  if(!plain(config))fail('INVALID_PLAN','config must be an object');
  const jobs=[];
  if(mode==='ab')jobs.push(task('T001','ab',pluginId,config,stableSeed(baseSeed,0)));
  if(mode==='sweep'){
   if(!['sigma','fill','value'].includes(raw.parameter))fail('INVALID_PLAN','Sweep parameter must explicitly name sigma / fill / value');
   if(!Array.isArray(raw.intensities)||!raw.intensities.length)fail('INVALID_PLAN','Nonempty intensity array required');
   raw.intensities.forEach((n,i)=>{
    if(typeof n!=='number'||!Number.isFinite(n))fail('INVALID_PLAN','Intensity must be finite');
    jobs.push(task('T'+String(i+1).padStart(3,'0'),'strength-'+i,pluginId,
      {...config,[raw.parameter]:n},stableSeed(baseSeed,i),n));
   });
  }
  if(mode==='batch'){
   if(!Array.isArray(raw.controls)||!raw.controls.length)fail('INVALID_PLAN','Explicit controls array required');
   raw.controls.forEach((v,i)=>{
    if(!plain(v)||!plain(v.config))fail('INVALID_PLAN','Each control must provide {config}');
    const id=v.id||'control-'+i;
    if(typeof id!=='string'||id.length>64||!id)fail('INVALID_PLAN','Invalid control id');
    jobs.push(task('T'+String(i+1).padStart(3,'0'),id,v.pluginId||pluginId,
      {...config,...v.config},stableSeed(baseSeed,i)));
   });
  }
  if(jobs.length>MAX_TASKS)fail('RESOURCE_LIMIT','Maximum 128 scheduled tasks per experiment');
  return{schema:'scept.u6.plan.v1',mode,pluginId,parameter:raw.parameter||null,
   baseSeed,tasks:jobs,
   notes:'Each task has its own deterministic seed; numeric controls are not guaranteed to be scientifically matched.'};
 }
 const iso=()=>new Date().toISOString();
 function scrubError(error){return{code:String(error?.code||'TASK_FAILED').slice(0,80),
  message:String(error?.message||error||'Unknown failure').slice(0,500)}}
 function pack(t){return{id:t.id,variant:t.variant,pluginId:t.pluginId,config:cloneJson(t.config),seed:t.seed,
  intensity:t.intensity,status:t.status,attempts:t.attempts,error:t.error,
  result:t.result,startedAt:t.startedAt,finishedAt:t.finishedAt}}
 function make({session,interventionEngine,observationAdapter,dataAdapter,core,onProgress=null,analysisAdapter=null}){
  if(!session||typeof session.run!=='function'||!session.description||!session.sha256||
    !interventionEngine?.execute||!observationAdapter?.observePair||!dataAdapter?.prepare||!core?.sha256)
    fail('INVALID_ENGINE','Requires genuine opened model session and U3/U4/U5 engines');
  if(onProgress!==null&&typeof onProgress!=='function')fail('INVALID_ENGINE','onProgress must be callable');
  if(analysisAdapter!==null&&typeof analysisAdapter.evaluateTask!=='function')fail('INVALID_ENGINE','analysisAdapter must declare evaluateTask()');
  let schedule=null,context=null,working=false,cancelRequested=false,disposed=false;
  function emit(){if(onProgress)try{onProgress(snapshot())}catch(_){/* UI callbacks cannot corrupt experiments */}}
  function snapshot(){
   if(!schedule)return{schema:'scept.u6.status.v1',status:'idle',tasks:[],total:0,completed:0,failed:0,cancelled:0,pending:0,running:0};
   const tasks=schedule.tasks.map(pack),counts={completed:0,failed:0,cancelled:0,pending:0,running:0};
   tasks.forEach(t=>counts[t.status]++);
   return{schema:'scept.u6.status.v1',status:working?'running':
    counts.pending+counts.cancelled+counts.failed===0?'completed':
    cancelRequested?'cancelled':counts.failed?'failed':'ready',
    total:tasks.length,...counts,modelId:context.modelId,modelSha256:context.modelSha256,
    inputHashes:context.inputHashes,referenceHashes:context.referenceHashes,
    profileId:context.profileId,plan:{mode:schedule.mode,pluginId:schedule.pluginId,
     parameter:schedule.parameter,baseSeed:schedule.baseSeed},tasks};
  }
  function ensureReady(){if(disposed)fail('DISPOSED','Scheduler has been disposed');if(working)fail('ALREADY_RUNNING','Scheduler is already running')}
  async function tensorHashes(feeds){const out={};for(const name of Object.keys(feeds).sort()){
    const t=feeds[name],hash=await core.sha256(t.data);
    out[name]={sha256:hash,type:t.type,dims:t.dims.slice()};
   }return out}
  async function create(spec,{feeds,referenceFeeds=null,profileId='generic'}={}){
   ensureReady();if(!session.description.inputs?.length)fail('INVALID_MODEL','No model input signature');
   const source=dataAdapter.prepare(session.description,feeds);
   const reference=referenceFeeds?dataAdapter.prepare(session.description,referenceFeeds):null;
   const planned=plan(spec);
   const hashes=await tensorHashes(source),referenceHashes=reference?await tensorHashes(reference):null;
   // Never persist patient tensors or the model checkpoint in the U6 task log.
   schedule=planned;context={feeds:source,referenceFeeds:reference,inputHashes:hashes,referenceHashes,
    modelId:session.manifest?.modelId||'loaded-onnx',modelSha256:session.sha256,
    profileId,probes:session.manifest?.probes||[],description:session.description};
   cancelRequested=false;emit();return snapshot();
  }
  async function run({maxTasks=MAX_TASKS}={}){
   ensureReady();if(!schedule||!context)fail('EMPTY_QUEUE','Create a plan before running');
   if(!Number.isInteger(maxTasks)||maxTasks<1||maxTasks>MAX_TASKS)fail('INVALID_PARAMETER','maxTasks must be integer [1,128]');
   cancelRequested=false;working=true;emit();let attempted=0;
   try{
    for(const t of schedule.tasks){
     if(cancelRequested||attempted>=maxTasks)break;
     if(t.status!=='pending')continue;
     t.status='running';t.startedAt=iso();t.finishedAt=null;t.attempts++;t.error=null;t.result=null;emit();
     try{
      const change=await interventionEngine.execute({pluginId:t.pluginId,description:context.description,
       feeds:context.feeds,referenceFeeds:context.referenceFeeds,config:t.config});
      if(cancelRequested){t.status='cancelled';continue}
      const a=await session.run(change.baseline);
      if(cancelRequested){t.status='cancelled';continue}
      const b=await session.run(change.edited);
      if(cancelRequested){t.status='cancelled';continue}
      const c=change.control?await session.run(change.control):null;
      if(cancelRequested){t.status='cancelled';continue}
      const args={description:context.description,baseline:a,edited:b,probes:context.probes,
       profileId:context.profileId,maxAxisUnits:128};
      const outputs=observationAdapter.observePair(args);
      const controls=c?observationAdapter.observePair({...args,edited:c}):null;
      const analysis=analysisAdapter?await analysisAdapter.evaluateTask({
       description:context.description,task:{id:t.id,variant:t.variant,seed:t.seed,config:cloneJson(t.config)},
       intervention:change.intervention,baseline:a,edited:b,control:c,
       observations:outputs,matchedControlObservations:controls,
       modelSha256:context.modelSha256,inputHashes:context.inputHashes}):null;
      if(analysis!==null){checkJson(analysis);if(!plain(analysis))fail('INVALID_ANALYSIS','Analysis must be a JSON object');
       const serialized=JSON.stringify(analysis);
       if(serialized.length>1000000)fail('RESOURCE_LIMIT','Per-task analysis exceeds 1 MB');}
      t.result={schema:'scept.u6.task.result.v1',modelSha256:context.modelSha256,
       seed:t.seed,intervention:change.intervention,outputs,matchedControlOutputs:controls,
       genuineInference:true,numberOfForwards:c?3:2,...(analysis===null?{}:{analysis:cloneJson(analysis)})};
      t.status='completed';
     }catch(error){
      t.status='failed';t.error=scrubError(error);
     }finally{
      t.finishedAt=iso();attempted++;emit();
     }
    }
   }finally{working=false;emit()}
   return snapshot();
  }
  function cancel(){
   if(disposed)fail('DISPOSED','Scheduler disposed');
   cancelRequested=true;emit();return snapshot();
  }
  function retryFailed(ids=null){
   ensureReady();if(!schedule)fail('EMPTY_QUEUE','Nothing to retry');
   if(ids!==null&&(!Array.isArray(ids)||ids.some(id=>typeof id!=='string'||!schedule.tasks.some(t=>t.id===id))))
    fail('INVALID_PARAMETER','Retry IDs must exist');
   for(const t of schedule.tasks){
    if(t.status==='failed'&&(ids===null||ids.includes(t.id))){
     t.status='pending';t.error=null;t.result=null;t.startedAt=null;t.finishedAt=null;
    }
   }
   cancelRequested=false;emit();return snapshot();
  }
  function resume(){
   ensureReady();if(!schedule)fail('EMPTY_QUEUE','No schedule');
   for(const t of schedule.tasks)if(t.status==='cancelled')t.status='pending';
   cancelRequested=false;emit();return snapshot();
  }
  function exportAudit(){return JSON.parse(JSON.stringify(snapshot()))}
  function exportCSV(){
   const s=snapshot(),esc=x=>'"'+String(x??'').replace(/"/g,'""')+'"';
   const lines=[['task_id','mode','variant','status','plugin','seed','intensity',
    'attempts','model_sha256','input_hashes','observed_tensor','mean_abs_delta',
    'matched_control_mean_abs_delta','error_code','error_message'].map(esc).join(',')];
   for(const t of s.tasks){
    const output=t.result?.outputs||[],control=t.result?.matchedControlOutputs||[];
    const tensorRows=output.length?output:[null];
    for(const o of tensorRows){
     const paired=control.find(x=>x.tensor===o?.tensor);
     lines.push([t.id,s.plan.mode,t.variant,t.status,t.pluginId,t.seed,t.intensity,
      t.attempts,s.modelSha256,JSON.stringify(s.inputHashes),o?.tensor,o?.mean_abs_delta,
      paired?.mean_abs_delta,t.error?.code,t.error?.message].map(esc).join(','));
    }
   }return lines.join('\n');
  }
  async function dispose(){ensureReady();disposed=true;context=null;schedule=null}
  return Object.freeze({VERSION,create,run,cancel,retryFailed,resume,snapshot,exportAudit,exportCSV,dispose});
 }
 return Object.freeze({VERSION,MAX_TASKS,plan,stableSeed,make});
});