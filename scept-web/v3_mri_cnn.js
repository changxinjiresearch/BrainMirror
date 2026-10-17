'use strict';
/* Optional MRI-trained 2D EfficientNetB3 demonstration.
   IMPORTANT model card reports dataset train/test leakage. Do not interpret
   probabilities as medically reliable. All processing and inference local.
*/
(function(){
const URL='https://huggingface.co/Udit013/brain-tumor-efficientnetb3/resolve/main/efficientnetb3.onnx';
const SHA='320b1ef648f3da5216f6ee0374349324b3cb00a76bebf7dbd1c3796e6155991c';
const $=id=>document.getElementById(id),s={session:null,loading:false,running:false,graph:null,audit:null};
const hash=async b=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b))).map(x=>x.toString(16).padStart(2,'0')).join('');
const fmt=x=>Number.isFinite(x)?x.toFixed(6):'—';
function state(msg,kind='pending'){$('readyLabel').textContent=msg;$('readyLabel').dataset.state=kind;$('runProgress').textContent=msg}
function ready(){const okay=!!s.session&&!!window.SCEPT_MRI?.state?.loaded;$('runExperiment').disabled=!okay;$('runExperiment').textContent=okay?(s.audit?'重新计算':'开始实验'):'准备 MRI 专用模型…';return okay}
async function load(){
 if(s.session||s.loading)return;
 s.loading=true;$('runExperiment').disabled=true;
 try{
  state('正在下载 MRI 训练模型（46 MB）…');
  const res=await fetch(URL,{mode:'cors',cache:'force-cache'});if(!res.ok)throw Error('模型下载 HTTP '+res.status);
  const bin=await res.arrayBuffer(),h=await hash(bin);
  if(h!==SHA)throw Error('MRI 模型数字指纹校验失败');
  $('sourceModelHash').textContent=h;
  $('instrumentedHash').textContent=h;
  s.graph=window.SCEPT_INSPECT_ONNX(bin);
  state('加载 MRI 专用 CNN…');
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  ort.env.wasm.numThreads=1;
  s.session=await ort.InferenceSession.create(new Uint8Array(bin),{executionProviders:['wasm']});
  if(!s.session.inputNames.includes('input')||!s.session.outputNames.includes('dense_2'))throw Error('MRI 模型接口与验证版本不一致');
  state('MRI 训练模型已就绪','ready');ready();
 }catch(e){state('MRI 研究模型不可用','error');$('warning').textContent='MRI 训练模型加载失败：'+e.message;s.session=null;$('runExperiment').textContent='重新加载'}
 finally{s.loading=false}
}
function drawNetwork(values){
 const canvas=$('networkCanvas'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
 ctx.fillStyle='#081923';ctx.fillRect(0,0,w,h);
 ctx.fillStyle='#9fc7c7';ctx.font='13px sans-serif';ctx.fillText('MRI 训练 EfficientNetB3 · 真实 ONNX 模型的算子与分类输出',18,24);
 const ops=s.graph?.nodes??[],N=12,bin=Math.max(1,Math.floor(ops.length/N));
 for(let j=0;j<N;j++){
  const x=35+j*(w-70)/(N-1);
  if(j<N-1){ctx.strokeStyle='#30525b';ctx.beginPath();ctx.moveTo(x+10,175);ctx.lineTo(x+(w-70)/(N-1)-8,175);ctx.stroke()}
  for(let k=0;k<5;k++){ctx.beginPath();ctx.arc(x,100+k*34,4.4,0,Math.PI*2);ctx.fillStyle='#7598a0';ctx.fill()}
  ctx.fillStyle='#95b8bc';ctx.font='10px sans-serif';ctx.textAlign='center';ctx.fillText('G'+j,x,287);
 }
 if(values){
  const d=values.after.map((n,i)=>Math.abs(n-values.before[i])),max=Math.max(...d,1e-12);
  for(let i=0;i<4;i++){
   const x=w/2-90+i*60,y=330;
   ctx.fillStyle=`rgba(${Math.floor(70+95*d[i]/max)},${Math.floor(135+90*d[i]/max)},153,1)`;
   ctx.beginPath();ctx.arc(x,y,6+5*d[i]/max,0,Math.PI*2);ctx.fill();
  }
 }
 ctx.textAlign='left';ctx.fillStyle='#83abad';ctx.font='11px sans-serif';ctx.fillText('灰色 = 未暴露内部激活 · 下方彩色点 = 实测分类输出差异（不是层内神经元）',17,355);
}
function plot(values){
 const canvas=$('responseChart'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
 ctx.fillStyle='#081923';ctx.fillRect(0,0,w,h);
 const peak=Math.max(1e-9,...values.after.map((n,i)=>Math.abs(n-values.before[i])));
 for(let i=0;i<4;i++){const x=100+i*165,d=Math.abs(values.after[i]-values.before[i]);ctx.fillStyle='#65d8bb';ctx.fillRect(x,154-Math.max(1,d/peak*112),65,Math.max(1,d/peak*112));ctx.fillStyle='#aac8c6';ctx.font='12px sans-serif';ctx.fillText('输出 '+i,x,173)}
 ctx.fillStyle='#a3c8c2';ctx.fillText('MRI 训练模型：四个真实输出差异（不可作为医学分类结论）',14,20);
}
async function run(){
 if(s.running)return;
 if(!ready()){await load();return}
 s.running=true;$('runExperiment').disabled=true;$('runExperiment').textContent='真实 MRI 模型计算中…';
 try{
  const base=SCEPT_MRI.makeCnnInput(false),edited=SCEPT_MRI.makeCnnInput(true);
  state('MRI CNN 推理：原始切片 1/2');
  const a=await s.session.run({input:new ort.Tensor('float32',base,[1,256,256,3])});
  state('MRI CNN 推理：修改后切片 2/2');
  const b=await s.session.run({input:new ort.Tensor('float32',edited,[1,256,256,3])});
  const before=Array.from(a.dense_2?.data||[]),after=Array.from(b.dense_2?.data||[]);
  if(before.length!==4||after.length!==4)throw Error('MRI 研究模型未返回四个数值');
  const deltas=before.map((v,i)=>after[i]-v),abs=deltas.map(Math.abs),mean=abs.reduce((a,b)=>a+b,0)/4,changed=abs.filter(v=>v>1e-7).length;
  const volume=window.SCEPT_MRI.state;
  s.audit={schema:'scept.v3.mri_cnn',at:new Date().toISOString(),model_sha256:SHA,model_name:'Udit013 brain-tumor EfficientNetB3',model_warning:'Author discloses train/test leakage. No diagnostic validity, do not use as medical evidence.',mri:volume,mri_sha256:await SCEPT_MRI.sourceHash(),input_base_sha256:await hash(base.buffer),input_edit_sha256:await hash(edited.buffer),preprocessing:'native i-j z=center slice, nearest-neighbor 256x256, 2-98% clipped minmax -> [0,255] grayscale-to-RGB, NHWC; not verified medical preprocessor parity',observations:{output:'dense_2',before,after,delta:deltas},evidence:'Engineering paired frozen-model response, no causal attribution'};
  $('metricLayers').textContent='输出层 1';
  $('metricChanged').textContent=changed+' / 4';$('metricMean').textContent=fmt(mean);
  $('layerSelect').replaceChildren();const opt=document.createElement('option');opt.textContent='四个类别输出';opt.value='0';$('layerSelect').add(opt);
  const table=$('layerTable');table.replaceChildren();
  for(let i=0;i<4;i++){const row=document.createElement('div');row.className='layer-row';const key=document.createElement('span'),v=document.createElement('span'),delta=document.createElement('strong');key.textContent='类别索引 '+i;v.textContent=fmt(before[i])+' → '+fmt(after[i]);delta.textContent=(deltas[i]>=0?'+':'')+fmt(deltas[i]);row.append(key,v,delta);table.append(row)}
  drawNetwork({before,after});plot({before,after});
  $('networkState').textContent='CNN 真实输出已更新';
  $('interpretation').textContent='MRI 专用预训练 CNN 确实对 MRI 图像强度干预产生了四个实际输出的数值变化。但模型作者披露训练/测试集泄漏，输出标签及概率未经本平台临床验证；内部灰色节点没有开放激活数据，不构成对模型内部机制的完整解释。';
  $('saveAudit').disabled=false;$('saveLayers').disabled=false;state('MRI 专用模型真实计算完成','ready');
 }catch(e){state('MRI CNN 运行失败','error');$('warning').textContent='MRI 模型推理失败：'+e.message}
 finally{s.running=false;ready()}
}
function saveBlob(contents,name,mime){const url=URL.createObjectURL(new Blob([contents],{type:mime})),a=document.createElement('a');a.download=name;a.href=url;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000)}
function exportAudit(){if(s.audit)saveBlob(JSON.stringify(s.audit,null,2),'scept-v3-mri-cnn-audit.json','application/json')}
function exportCsv(){if(!s.audit)return;const m=s.audit.observations;saveBlob(['index,baseline,modified,delta',...m.before.map((v,i)=>[i,v,m.after[i],m.delta[i]].join(','))].join('\n'),'scept-v3-mri-cnn-output.csv','text/csv')}
function activate(){
 $('modelInfo').textContent='真实 MRI 训练 EfficientNetB3 · 4 类 T1 MRI 图像分类研究模型；作者披露数据泄漏，不能用于医学诊断。';
 $('warning').textContent='MRI 训练模型已开放工程测试。此模型作者报告训练/测试数据泄漏，且公开 MNI152 模板不代表病人肿瘤；分类变化不能当作临床结论。';
 $('layerTable').innerHTML='<div class="empty">运行 MRI 专用 CNN 后显示真实四个输出变化；内部网络激活尚未开放。</div>';
 $('layerSelect').replaceChildren();const o=document.createElement('option');o.textContent='MRI 分类输出';$('layerSelect').add(o);
 drawNetwork(null);$('metricLayers').textContent='—';$('metricChanged').textContent='—';$('metricMean').textContent='—';
 $('runExperiment').disabled=true;
 if(s.session)ready();else load();
}
window.SCEPT_MRI_CNN={activate,readyIfPossible:ready,run,exportAudit,exportCsv,getAudit:()=>s.audit,isReady:()=>!!s.session};
})();