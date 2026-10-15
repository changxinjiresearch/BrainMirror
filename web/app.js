'use strict';
const $=id=>document.getElementById(id);
let state=null, lastRun=null;
const plotConfig={responsive:true,displaylogo:false,scrollZoom:true};
const base3D={paper_bgcolor:'#0d1726',plot_bgcolor:'#0d1726',font:{color:'#8fa5bf',size:10},margin:{l:0,r:0,t:0,b:0},showlegend:false,scene:{bgcolor:'#0d1726',aspectmode:'cube',xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},camera:{eye:{x:1.55,y:1.4,z:1.1}}}};
function err(ex){$('error').textContent=ex instanceof Error?ex.message:String(ex);$('error').classList.remove('hidden');}
function clearError(){$('error').classList.add('hidden');}
async function request(path,options={}){const res=await fetch(path,options);let out=await res.json();if(!res.ok)throw new Error(out.detail||`${res.status} ${res.statusText}`);return out;}
function body(payload){return {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)};}
function meshTrace(m,color,opacity,name){return {type:'mesh3d',x:m.x,y:m.y,z:m.z,i:m.i,j:m.j,k:m.k,color,opacity,name,hoverinfo:'skip',flatshading:true,lighting:{ambient:.66,diffuse:.57,specular:.14,roughness:.88}};}
function showBrain(result){Plotly.react('brainPlot',[meshTrace(result.brain_mesh,'#66aaca',.18,'MRI threshold'),meshTrace(result.roi_mesh,'#f9a956',.72,'Editable ROI')],base3D,plotConfig);}
function showCloud(cloud){
 const xs=cloud.map(v=>v.x+v.c*5.2),ys=cloud.map(v=>v.y),zs=cloud.map(v=>v.z);
 const mx=Math.max(...cloud.map(v=>Math.abs(v.delta)),1e-9);
 const t={type:'scatter3d',mode:'markers',x:xs,y:ys,z:zs,
   marker:{size:cloud.map(v=>3+7*Math.abs(v.delta)/mx),color:cloud.map(v=>v.delta),
           colorscale:[[0,'#f29c69'],[.5,'#687f94'],[1,'#63d8ce']],cmin:-mx,cmax:mx,showscale:true,
           colorbar:{title:'Δ activation',thickness:11,len:.6},opacity:.95},
   text:cloud.map(v=>`conv3[${v.c},${v.z},${v.y},${v.x}]<br>before=${v.before.toFixed(6)}<br>after=${v.after.toFixed(6)}<br>Δ=${v.delta.toFixed(6)}`),
   customdata:cloud.map(v=>[v.c,v.z,v.y,v.x]),hovertemplate:'%{text}<extra></extra>'};
 Plotly.react('neuralPlot',[t],base3D,plotConfig).then(()=>{
   const plot=$('neuralPlot');
   plot.removeAllListeners('plotly_click');
   plot.on('plotly_click',evt=>{
     const d=evt.points?.[0]?.customdata;if(!d)return;
     $('layer').value='conv3';updateLayerBounds();
     [$('channel').value,$('z').value,$('y').value,$('x').value]=d;
     $('graphView').value='focus';analyze();
   });
 });
}
function showResultGraph(r){
 if($('graphView').value==='cloud')showCloud(r.cloud);
 else showGraph(r.graph);
}
function showGraph(g){
 const edges=g.edges;
 // True input tensor elements, one displayed point per source unit.
 const inputs={x:[],y:[],z:[],text:[],color:[],size:[]};
 const lines={x:[],y:[],z:[]};
 const signs={positive:{x:[],y:[],z:[]},negative:{x:[],y:[],z:[]}};
 let max=Math.max(...edges.map(e=>Math.abs(e.contribution)),1e-9);
 for(const edge of edges){
  const [c,z,y,x]=edge.source;
  // Layer-separated arrangement is a visual layout only, not anatomical geometry.
  const px=(x%16)*.125+(c%4)*2.25-3.5;
  const py=y*.20+(Math.floor(c/4))*2.1-1.5;
  const pz=z*.20-1.35;
  const intensity=Math.abs(edge.contribution)/max;
  inputs.x.push(px);inputs.y.push(py);inputs.z.push(pz);
  inputs.text.push(`${g.source_layer}[${c},${z},${y},${x}]<br>input=${edge.activation.toFixed(6)}<br>weight=${edge.weight.toFixed(6)}<br>weighted contribution=${edge.contribution.toFixed(6)}`);
  inputs.color.push(edge.contribution>=0?'#52cfc5':'#f7a664');inputs.size.push(3+7*intensity);
  const dest={x:2.9,y:1.1,z:1.1};
  const s=edge.contribution>=0?signs.positive:signs.negative;
  s.x.push(px,dest.x,null);s.y.push(py,dest.y,null);s.z.push(pz,dest.z,null);
 }
 const traces=[{type:'scatter3d',mode:'lines',x:signs.positive.x,y:signs.positive.y,z:signs.positive.z,line:{color:'#48adac',width:1.5},hoverinfo:'skip'},
 {type:'scatter3d',mode:'lines',x:signs.negative.x,y:signs.negative.y,z:signs.negative.z,line:{color:'#bc8054',width:1.5},hoverinfo:'skip'},
 {type:'scatter3d',mode:'markers',x:inputs.x,y:inputs.y,z:inputs.z,marker:{size:inputs.size,color:inputs.color,opacity:.84},text:inputs.text,hovertemplate:'%{text}<extra></extra>'},
 {type:'scatter3d',mode:'markers+text',x:[2.9],y:[1.1],z:[1.1],text:[`${g.layer}[${g.unit.join(',')}]`],textposition:'top center',marker:{size:13,color:'#ffe3a4',line:{color:'#fff',width:1}},hovertemplate:`preactivation=${g.preactivation.toFixed(7)}<br>post-ReLU=${g.activation.toFixed(7)}<extra></extra>`}];
 Plotly.react('neuralPlot',traces,base3D,plotConfig).then(()=>$('neuralPlot').removeAllListeners('plotly_click'));
}
function renderMetric(label,num,digits=5){const str=num===null||num===undefined?'—':typeof num==='number'?num.toFixed(digits):String(num);return `<div class="metric"><strong>${str}</strong><span>${label}</span></div>`;}
function metrics(result){const m=result.summary;let html='';
 html+=renderMetric('原始神经元激活',m.baseline.activation);
 html+=renderMetric('修改后神经元激活',m.edited.activation);
 html+=renderMetric('激活变化 Δ',m.activation_delta);
 html+=renderMetric('输入修改后重建 MSE',m.reconstruction_mse_edited);
 html+=renderMetric('重建前后平均绝对差',m.mean_abs_reconstruction_change);
 html+=renderMetric('内部置零引起的输出变化',m.ablation_mean_abs_output_change);
 html+=renderMetric('局部 ROI 体积（原始）',result.qc.target_volume_before,0);
 html+=renderMetric('局部 ROI 体积（修改后）',result.qc.target_volume_after,0);
 html+=renderMetric('非目标远场强度平均变化',result.qc.non_target_mean_abs_change,7);
 $('metrics').innerHTML=html;
 $('runId').textContent=`${result.run_id} · ${result.graph.edges.length} 条核实连接`;
 $('download').href=`/api/runs/${result.run_id}.json`; $('download').classList.remove('hidden');
 $('caution').textContent=`${result.trained?'训练过的':'随机初始化未训练的'}模型 · ${result.source==='synthetic'?'合成 MRI':'上传 MRI'}。内部加权和数值一致性已验证。连接不是经过实验确认的因果边。局部形变并非临床验证。模型 SHA256 前缀: ${result.model_sha256_prefix}`;
}
function updateLayerBounds(){const layer=$('layer').value;let side=layer==='conv1'?16:layer==='conv2'?8:4;let ch=layer==='conv1'?4:8;
 $('channel').max=ch-1;$('channel').value=Math.min(Number($('channel').value),ch-1);
 for(const id of ['z','y','x']){$(id).max=side-1;$(id).value=Math.min(Number($(id).value),side-1);}
}
async function analyze(){clearError();$('analyzeBtn').disabled=true;
 try{
  const r=await request('/api/analyze',body({strength:Number($('strength').value),layer:$('layer').value,channel:Number($('channel').value),z:Number($('z').value),y:Number($('y').value),x:Number($('x').value),ablate:$('ablate').checked}));
  lastRun=r;showBrain(r);showResultGraph(r);metrics(r);
 }catch(e){err(e);}finally{$('analyzeBtn').disabled=false;}
}
async function refresh(){try{
 state=await request('/api/state');
 $('modelStatus').textContent=state.trained?'● 已训练模型 · '+state.model_sha256_prefix:'○ 随机权重 · 未训练';
 $('sourceTag').textContent=state.selected_source==='synthetic'?'SYNTHETIC TEST PHANTOM':'UPLOADED MRI · LOCAL';
 const t=state.training;
 $('trainProgress').textContent=t.state==='running'?`训练中 ${t.epoch}/${t.total} epochs · loss=${t.losses.length?t.losses[t.losses.length-1].toFixed(6):'—'}`:
 t.state==='completed'?`训练完成，最后 loss=${t.losses[t.losses.length-1].toFixed(6)}（仅训练损失）`:
 t.state==='failed'?`训练失败：${t.error}`:`已上传 ${state.real_sample_count} 个真实样本 · 等待训练`;
 if(state.trained && t.state!=='running'){
  $('trainProgress').textContent+=` · 已保存模型来源: ${state.model_provenance.training_source||'unknown'}（${state.model_provenance.training_examples||'?' }个样本）`;
 }
 $('trainBtn').disabled=t.state==='running';
 }catch(e){err(e);}}
$('strength').addEventListener('input',e=>$('strengthLabel').textContent=Number(e.target.value).toFixed(2));
$('layer').addEventListener('change',updateLayerBounds);
$('graphView').addEventListener('change',()=>{if(lastRun)showResultGraph(lastRun);});
$('analyzeBtn').addEventListener('click',analyze);
$('demoBtn').addEventListener('click',async()=>{try{await request('/api/select_demo',{method:'POST'});await refresh();await analyze();}catch(e){err(e);}});
$('uploadForm').addEventListener('submit',async e=>{e.preventDefault();clearError();try{
 const fd=new FormData(e.target);const result=await request('/api/upload',{method:'POST',body:fd});await refresh();await analyze();
 }catch(ex){err(ex);}});
$('trainBtn').addEventListener('click',async()=>{clearError();try{
 await request('/api/train',body({source:$('trainingSource').value,epochs:Number($('epochs').value),phantom_count:12,seed:42}));await refresh();
 }catch(ex){err(ex);}});
updateLayerBounds();
Plotly.newPlot('brainPlot',[],base3D,plotConfig);Plotly.newPlot('neuralPlot',[],base3D,plotConfig);
refresh().then(analyze);
setInterval(refresh,1500);
