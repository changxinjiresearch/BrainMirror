/* SCEPT live web integration: real model, no simulated inference. */
(function(){
'use strict';
const panel=document.createElement('section');panel.id='livePanel';panel.className='panel live-panel';
panel.innerHTML=[
'<div class="live-heading"><div><div class="panel-kicker">00 / LIVE COMPUTATION · NO TRAINING</div><h2>真实 MRI → SCEPT 机制实验</h2><p>上传已配准的四模态 MRI，后台自动加载预训练 SwinUNETR。</p></div><span class="live-pill" id="liveStatus">正在检查服务器</span></div>',
'<form id="liveForm"><div class="modality-grid">',
'<label>FLAIR <span>肿瘤边界</span><input type="file" name="flair" accept=".nii,.nii.gz" required></label>',
'<label>T1ce <span>增强 T1</span><input type="file" name="t1ce" accept=".nii,.nii.gz" required></label>',
'<label>T1 <span>原始 T1</span><input type="file" name="t1" accept=".nii,.nii.gz" required></label>',
'<label>T2 <span>原始 T2</span><input type="file" name="t2" accept=".nii,.nii.gz" required></label>',
'</div><div class="live-bottom"><div class="private-input"><label>访问密钥 <input id="liveKey" type="password" autocomplete="off" placeholder="计算服务接入后填写"></label><small>仅当前页面内存使用，不写入实验报告。</small></div>',
'<button class="primary" id="liveStart" type="submit" disabled>执行真实 SCEPT 实验</button></div>',
'<div id="liveProgress" class="live-progress" role="status" aria-live="polite">等待独立计算服务器接入。网站不会伪造新预测。</div></form>',
'<p class="help live-disclaimer">需要四种已经对齐的 3D MRI，每个文件不超过 90 MiB。当前只测 64³ 脑中心局部区域。实验效应不等于独立临床诊断结论。</p>'
].join('');
const target=document.querySelector('#page-observatory .workbench');
if(target)target.parentNode.insertBefore(panel,target);
const status=document.getElementById('liveStatus'),info=document.getElementById('liveProgress'),start=document.getElementById('liveStart'),form=document.getElementById('liveForm');
let origin=null,running=false;
const phases={queued:'等待计算任务',loading_verified_pretrained_weights:'自动校验并加载 SwinUNETR',
reading_four_modalities:'读取并验证四种 MRI',real_swinunetr_prediction:'运行真实 MRI 推理',
screening_actual_attention_messages:'筛选真实 Attention 消息',
real_attention_disruption_and_restoration:'执行破坏与恢复干预',
complete:'完成；已将结果更新到研究工作台',failed:'计算失败（未生成有效新实验）'};
function state(label,ready){status.textContent=label;status.classList.toggle('online',ready);start.disabled=!ready||running}
async function configure(){
 try{
  const r=await fetch('./scept-config.json',{cache:'no-store'});if(!r.ok)throw Error('no config');
  const cfg=await r.json();
  if(typeof cfg.api_url!=='string'||!/^https:\/\/[a-z0-9.-]+(?::\d+)?$/i.test(cfg.api_url))throw Error('no backend');
  origin=cfg.api_url;
  const h=await fetch(origin+'/health',{signal:AbortSignal.timeout(10000)});
  if(!h.ok)throw Error('backend not healthy');
  const health=await h.json();if(health.status!=='ready'||health.mode!=='real_pretrained_model')throw Error('not configured');
  state(health.cuda?'GPU 可用':'CPU 可用（速度较慢）',true);
  info.textContent='真实计算服务已连接。选择四种 MRI，填写私人密钥后开始。';
 }catch(e){state('计算服务器尚未连接',false);info.textContent='网页可以导入 MRI / 回放实验；新的真实推理仍须部署独立计算服务，不会使用 Kaggle 或伪造结果。'}
}
async function awaitResult(id,key){
 for(let i=0;i<600;i++){
  await new Promise(ok=>setTimeout(ok,3000));
  const r=await fetch(origin+'/api/v1/runs/'+encodeURIComponent(id),{headers:{Authorization:'Bearer '+key},signal:AbortSignal.timeout(20000)});
  if(!r.ok)throw Error('结果查询失败：HTTP '+r.status);
  const s=await r.json();
  info.textContent=phases[s.step]||('后台：'+s.step);
  if(s.status==='complete'){if(s.result?.schema!=='scept/live-4modality/v1')throw Error('结果结构未通过校验');return s.result}
  if(s.status==='failed')throw Error('真实推理失败：'+s.error_type);
 }
 throw Error('等待超时；任务结果尚未确认');
}
form.addEventListener('submit',async function(ev){
 ev.preventDefault();if(running||!origin)return;
 const data=new FormData(form);
 for(const name of ['flair','t1ce','t1','t2']){
  const f=data.get(name);
  if(!(f instanceof File)||!f.size||!/\.nii(\.gz)?$/i.test(f.name)||f.size>90*1024*1024){info.textContent='请正确选择 '+name+' 的 .nii / .nii.gz 文件（≤90 MiB）';return}
 }
 const key=document.getElementById('liveKey').value.trim();if(!key){info.textContent='请输入私人 SCEPT 访问密钥';return}
 running=true;state('真实计算中',false);
 try{
  info.textContent='正在将 MRI 安全传输至私有计算服务…';
  const flair=data.get('flair');
  const r=await fetch(origin+'/api/v1/runs',{method:'POST',headers:{Authorization:'Bearer '+key},body:data});
  if(!r.ok){const e=await r.json().catch(()=>({detail:'unknown'}));throw Error(e.detail||('HTTP '+r.status))}
  const job=await r.json();if(!job.job_id)throw Error('无效任务编号');
  if(typeof importMRI==='function')await importMRI(flair,false);
  const measured=await awaitResult(job.job_id,key);
  S.record=measured;S.recordQuality='live';S.graphMode='path';S.selectedPath=0;
  document.getElementById('pathMode').classList.add('active');
  document.getElementById('architectureMode').classList.remove('active');
  updateRecord();
  if(typeof event==='function')event('SCEPT 实时计算完成','四模态 MRI · 实测 Attention','Private model worker');
  if(typeof setView==='function')setView('observatory');
  info.textContent='真实模型实验完成。右侧可检查被干预原子，下方是实际测量的预测效应。';
 }catch(ex){info.textContent='实验未完成：'+ex.message}
 finally{running=false;start.disabled=!origin;status.textContent=origin?'计算服务已连接':'计算服务未连接'}
});
configure();
})();
