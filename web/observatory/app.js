'use strict';
/* SCEPT Observatory v1.0. Browser side is replay/visualization only.
   No fake prediction, activation, patient MRI or GPU API is fabricated. */
const $=id=>document.getElementById(id);
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=6)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—';
const FIXED_SHA='62b649ecf98b05c73d7a36a7e290e85ad90a9b5d0dab442f296342406feb57ba';
const INITIAL_RECORD={
 schema:'brainmirror/screenshot-transcription/v1',
 source_provenance:'User-observed Kaggle console screenshot, 2026-10-16; ORIGINAL JSON NOT IMPORTED',
 scientific_status:'real pretrained model forward and atomic intervention, NOT independent validation',
 checkpoint_source:'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold1_f48_ep300_4gpu_dice0_9059.zip',
 checkpoint_sha256:FIXED_SHA,
 training_data:'BraTS2021 (official pretrained Fold 1)',
 case_id:'BraTS20_Training_002',
 cohort:'BraTS2020 369-case labelled training collection',
 evaluation_scope:'one ground-truth-guided 64x64x64 tumour patch; NOT patient-independent',
 overlap_assessment:'UNKNOWN: BraTS2021 could contain the BraTS2020 case',
 official_modality_order:['flair','t1ce','t1','t2'],
 official_output_order:['TC','WT','ET'],
 patch_dice_NOT_full_volume:{WT:{dice:0.9041616723036165},TC:{dice:0.9479754601226994},ET:{dice:0.8629651506202008}},
 scept_single_atom_intervention:{atom:null,disruption_effect:0.0,restoration_effect:0.0,
 note:'Actual attention atom metadata unavailable until original JSON is imported.'},
 seconds:14.25,
 input_volumes_present:false
};
const S={
 record:INITIAL_RECORD,recordQuality:'screenshot',view:'observatory',mri:null,mask:null,
 renderMode:'slice',graphMode:'architecture',slice:0,yaw:40,opacity:65,
 selectedStage:0,selectedPath:0,history:[],events:[],pointCloud:null
};
function now(){return new Date().toLocaleString('zh-CN',{hour12:false})}
function event(type,description,source){
 S.history.unshift({at:now(),type,subject:S.record?.case_id||'—',source:source||'本地操作'});
 S.events.unshift({type,description,at:now()});
 renderEvents();renderHistory();
}
function notice(message,error=false){
 const el=$('notice');el.textContent=message;el.className='notice show';
 el.style.background=error?'#3b2026':'#14343e';el.style.borderColor=error?'#8a4b54':'#31616a';
 clearTimeout(notice.t);notice.t=setTimeout(()=>el.classList.remove('show'),6500);
}
function getDice(record){
 const d=record.patch_dice_NOT_full_volume||record.patch_dice_not_patient_generalization||{};
 return Object.fromEntries(['WT','TC','ET'].map(k=>[k,typeof d[k]==='number'?d[k]:d[k]?.dice]));
}
function effect(record){
 if(record.scept_roi_guided_search){
  const x=record.scept_roi_guided_search;
  return {disruption:x.selected_effect,restoration:null,status:x.status,reason:x.reason,
          measured:!!x.selected_path,search:true,control:x.negative_control};
 }
 const x=record.scept_single_atom_intervention||record.scept_atomic_intervention||{};
 return {disruption:x.disruption_effect,restoration:x.restoration_effect,status:'single_atom',
         measured:!!x.atom,search:false,control:null};
}
function paths(record){
 const e=record.scept_roi_guided_search;
 if(e&&Array.isArray(e.selected_path)&&e.selected_path.length) return e.selected_path;
 const atom=(record.scept_single_atom_intervention||record.scept_atomic_intervention||{}).atom;
 return atom&&typeof atom.module==='string'?[atom]:[];
}
function updateRecord(){
 const r=S.record,d=getDice(r),e=effect(r);
 $('heroCase').textContent=(r.case_id||'未记录').replace('BraTS20_','');
 const h=r.checkpoint_sha256||r.weights_sha256;
 $('heroHash').textContent=h?h.slice(0,12):'未知';
 $('caseSource').textContent=S.recordQuality==='screenshot'?'屏幕记录转录 · 待原始 JSON':S.recordQuality==='imported'?'原始 JSON 已导入 · 非独立验证':'实验记录导入';
 $('heroEffect').textContent=fmt(e.disruption,6);
 $('effectDescription').textContent=e.search?'SCEPT 自动搜索路径效应':'局部 Attention 干预';
 $('dataQuality').textContent=S.recordQuality==='screenshot'?'截图转录':'原始实验 JSON';
 $('disruptVal').textContent=fmt(e.disruption,6);
 $('restoreVal').textContent=fmt(e.restoration,6);
 $('durationVal').textContent=typeof (r.seconds??r.elapsed_seconds)==='number'?fmt(r.seconds??r.elapsed_seconds,2)+' s':'—';
 const res=['WT','TC','ET'].map(k=>{
   const v=d[k],valid=typeof v==='number'&&Number.isFinite(v);
   return '<div class="dice-line"><span class="dice-label">'+k+'</span><div class="dice-track"><div class="dice-fill" style="width:'+(valid?Math.max(0,Math.min(100,v*100)):0)+'%"></div></div><span class="dice-number">'+(valid?fmt(v,4):'—')+'</span></div>';
 }).join('');
 $('diceBars').innerHTML=res;
 $('effectNote').textContent=e.search?('候选路径搜索状态：'+(e.status||'unknown')+'；'+(e.reason||'无结论')+'。候选效应需要重复病例和负对照，不等于已证明因果路径。'):(e.disruption===0&&e.restoration===0?'本次 Attention 原子破坏 / 恢复效应均为零，不能宣称已发现有效计算机制。':'观测到单原子效应；不等于跨层因果机制已通过独立验证。');
 renderGraph();renderProvenance();
}
function svgNode(x,y,w,h,title,sub,opts={}){
 const stage=opts.stage;
 return '<g class="graph-node '+(opts.active?'active ':'')+(opts.observed?'observed ':'')+'" role="button" tabindex="0" data-stage="'+(stage??'')+'" data-index="'+(opts.index??'')+'"><rect x="'+x+'" y="'+y+'" rx="12" width="'+w+'" height="'+h+'"/><text x="'+(x+w/2)+'" y="'+(y+29)+'" text-anchor="middle" class="graph-text">'+escapeHTML(title)+'</text><text x="'+(x+w/2)+'" y="'+(y+49)+'" text-anchor="middle" class="graph-sub">'+escapeHTML(sub)+'</text></g>';
}
function renderGraph(){
 const svg=$('networkSvg'),p=paths(S.record);
 if(S.graphMode==='path'&&p.length){
   $('graphLabel').textContent=p.length+' recorded atom'+(p.length===1?'':'s');
   const n=p.length,w=Math.min(230,Math.floor(660/n)-26),gap=n>1?(710-n*w)/(n-1):0;
   let code='<text x="42" y="52" class="graph-legend">RECORDED SCEPT ATOM COORDINATES · FROM ORIGINAL JSON</text>';
   p.forEach((a,i)=>{const x=n===1?300:45+i*(w+gap);
      if(i>0){const old=n===1?300:45+(i-1)*(w+gap);code+='<path class="graph-edge observed" d="M '+(old+w)+' 193 L '+x+' 193"/>';}
      code+=svgNode(x,149,w,90,a.module?.split('.').slice(-3).join('.')||'Unknown atom',
         'h'+a.head+' / w'+a.window+' / q'+a.query+'→k'+a.key,
         {index:i,observed:true,active:i===S.selectedPath});
   });
   code+='<text x="400" y="320" class="graph-empty">绿色连接 = 本次记录的候选干预序列，并非任意两单元的真实权重</text>';
   svg.innerHTML=code;
 }else if(S.graphMode==='path'){
   $('graphLabel').textContent='No atom identity in current record';
   svg.innerHTML='<text x="400" y="150" class="graph-empty">尚无可视化的完整路径原子记录</text><text x="400" y="182" class="graph-empty">请导入 Kaggle 导出的 scept_fast_result.json 或 scept_auto_result.json</text><text x="400" y="215" class="graph-empty">不会根据 0.0 效应凭空生成节点和连线</text>';
 }else{
   $('graphLabel').textContent='Swin Transformer · 4 stages';
   const xs=[42,238,434,630],heads=[3,6,12,24],depths=[2,2,2,2];
   let code='<text x="42" y="52" class="graph-legend">OFFICIAL MONAI SWINUNETR / SWIN TRANSFORMER ENCODER</text>';
   for(let i=0;i<3;i++)code+='<path class="graph-edge" d="M '+(xs[i]+130)+' 199 L '+(xs[i+1])+' 199"/>';
   for(let i=0;i<4;i++)code+=svgNode(xs[i],155,130,90,'Stage '+(i+1),depths[i]+' blocks · '+heads[i]+' heads',{stage:i,active:i===S.selectedStage,observed:p.some(a=>String(a.module||'').includes('layers'+(i+1)+'.'))});
   code+='<text x="400" y="319" class="graph-empty">灰色线 = 官方网络架构的阶段顺序，非 SCEPT 测量的原子级因果连线</text>';
   svg.innerHTML=code;
 }
 svg.querySelectorAll('.graph-node').forEach(node=>{
   const act=()=>{if(S.graphMode==='path')S.selectedPath=Number(node.dataset.index);else S.selectedStage=Number(node.dataset.stage);renderGraph()};
   node.addEventListener('click',act);
   node.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();act()}});
 });
 updateInspector();
}
function updateInspector(){
 const p=paths(S.record);
 if(S.graphMode==='path'){
  if(!p.length){$('inspectorTitle').textContent='未导入原子级记录';$('inspectorDescription').textContent='目前仅知道干预数值，不存在能够如实绘制的真实连接图。';$('inspectorType').textContent='缺少原子索引';$('inspectorMeasured').textContent='未记录';return;}
  const a=p[Math.min(S.selectedPath,p.length-1)];
  $('inspectorTitle').textContent=a.module||'Attention atom';
  $('inspectorDescription').textContent='原子类型 '+(a.kind||'unknown')+'；Window '+a.window+' / Head '+a.head+' / Query '+a.query+' / Key '+a.key+'。这是导入的真实实验记录坐标，不是网页生成的激活。';
  $('inspectorType').textContent='Imported SCEPT JSON';
  $('inspectorMeasured').textContent=fmt(effect(S.record).disruption);
  return;
 }
 const names=['layers1.0','layers2.0','layers3.0','layers4.0'],heads=[3,6,12,24];
 $('inspectorTitle').textContent='swinViT.'+names[S.selectedStage];
 $('inspectorDescription').textContent='该阶段含 2 个 Swin Transformer blocks，单块注意力 '+heads[S.selectedStage]+' heads。网络节点来自官方架构；未导入的真实激活数值不会被模拟。';
 $('inspectorType').textContent='MONAI SwinUNETR';
 $('inspectorMeasured').textContent=paths(S.record).some(a=>String(a.module||'').includes('layers'+(S.selectedStage+1)+'.'))?'包含已记录原子':'当前无激活张量';
}
function renderEvents(){
 $('eventCount').textContent=S.events.length+' 条记录';
 $('eventList').innerHTML=S.events.slice(0,8).map(e=>'<div class="event-item"><strong>'+escapeHTML(e.type)+'</strong><span>'+escapeHTML(e.description)+'</span></div>').join('');
}
function renderHistory(){
 $('historyCount').textContent=S.history.length+' 项';
 $('historyBody').innerHTML=S.history.map(e=>'<tr><td>'+escapeHTML(e.at)+'</td><td>'+escapeHTML(e.type)+'</td><td>'+escapeHTML(e.subject)+'</td><td>'+escapeHTML(e.source)+'</td></tr>').join('');
}
function renderProvenance(){
 const r=S.record,d=getDice(r),e=effect(r);
 const items=[
 ['原始来源',S.recordQuality==='screenshot'?'2026-10-16 Kaggle 运行截图转录（未导入完整 JSON）':'用户导入的 SCEPT 实验 JSON'],
 ['患者 ID',r.case_id||'unknown'],
 ['模型',r.training_dataset||r.training_data||r.model||'MONAI SwinUNETR official BraTS21 Fold 1'],
 ['权重 SHA-256',r.checkpoint_sha256||r.weights_sha256||'unknown'],
 ['输入通道',JSON.stringify(r.official_modality_order||r.input_modalities||['flair','t1ce','t1','t2'])],
 ['输出通道',JSON.stringify(r.official_output_order||r.model_outputs||['TC','WT','ET'])],
 ['局部 Dice / WT',fmt(d.WT,6)],['局部 Dice / TC',fmt(d.TC,6)],['局部 Dice / ET',fmt(d.ET,6)],
 ['干预 Delta',fmt(e.disruption)],['恢复 Delta',fmt(e.restoration)],
 ['测试独立性',S.recordQuality==='live'?'用户上传局部 MRI，未执行患者级独立泛化验证':'未证明；BraTS21 / BraTS20 病例来源可能重叠'],
 ['医学范围','局部 64³ Patch，非整例 MRI 泛化测试']
 ];
 $('provenance').innerHTML=items.map(([k,v])=>'<div><dt>'+escapeHTML(k)+'</dt><dd>'+escapeHTML(v)+'</dd></div>').join('');
 $('rawJson').textContent=JSON.stringify(r,null,2);
}
function setView(page){
 S.view=page;
 for(const k of ['observatory','evidence','history'])$('page-'+k).classList.toggle('page-active',page===k);
 document.querySelectorAll('button.side-item[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===page));
 $('crumbView').textContent={observatory:'机制观测台',evidence:'实验证据',history:'实验记录'}[page];
}
async function importJSON(file){
 if(!file)return;
 try{
  if(file.size>10*1024*1024)throw Error('JSON 大于 10 MiB，请先生成轻量科研记录');
  const data=JSON.parse(await file.text()),r=data.record&&data.export_kind==='brainmirror/observatory-audit/v1'?data.record:data;
  if(!r||typeof r!=='object'||typeof r.case_id!=='string')throw Error('非 SCEPT 单病例结果 JSON：缺少 case_id');
  if(!r.patch_dice_NOT_full_volume&&!r.patch_dice_not_patient_generalization&&!r.scept_roi_guided_search)throw Error('没有可识别的真实推理或路径记录');
  S.record=r;S.recordQuality='imported';S.selectedPath=0;S.selectedStage=0;updateRecord();
  event('导入实验 JSON','原始模型数值已加载',file.name);
  notice('SCEPT 原始记录已导入：'+r.case_id+'。MRI 体素文件需单独选择。');
 }catch(ex){notice('JSON 导入失败：'+ex.message,true)}
 finally{$('traceInput').value=''}
}
async function loadNifti(file){
 if(!file)return null;
 if(file.size>256*1024*1024)throw Error('文件超过 256 MiB 浏览器安全上限');
 let buffer=await file.arrayBuffer();
 if(file.name.toLowerCase().endsWith('.gz')){
  if(typeof DecompressionStream==='undefined')throw Error('当前浏览器不支持 gzip 解压，请改用未压缩 .nii 文件');
  buffer=await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 }
 if(buffer.byteLength>300*1024*1024)throw Error('解压后 NIfTI 超过 300 MiB 安全上限');
 const dv=new DataView(buffer),le=dv.getInt32(0,true)===348?true:dv.getInt32(0,false)===348?false:null;
 if(le===null)throw Error('NIfTI-1 header 无效，仅支持 .nii / .nii.gz');
 const dim=[dv.getInt16(42,le),dv.getInt16(44,le),dv.getInt16(46,le)],dtype=dv.getInt16(70,le);
 if(dim.some(v=>v<8||v>512))throw Error('三维 NIfTI 维度无效');
 if(dv.getInt16(40,le)>3)throw Error('请选择单个 3D 影像，而不是 4D 多时点体数据');
 const bytes={2:1,4:2,8:4,16:4,64:8,512:2}[dtype];
 if(!bytes)throw Error('不支持的 NIfTI datatype：'+dtype);
 const offset=Math.floor(dv.getFloat32(108,le)),count=dim[0]*dim[1]*dim[2];
 if(offset<352||offset+count*bytes>buffer.byteLength)throw Error('NIfTI 数据长度与头文件不一致');
 const slope=dv.getFloat32(112,le)||1,intercept=dv.getFloat32(116,le)||0;
 function voxel(i){const o=offset+i*bytes;let v;
  switch(dtype){case 2:v=dv.getUint8(o);break;case 4:v=dv.getInt16(o,le);break;case 8:v=dv.getInt32(o,le);break;case 16:v=dv.getFloat32(o,le);break;case 64:v=dv.getFloat64(o,le);break;case 512:v=dv.getUint16(o,le);break;}
  return v*slope+intercept;
 }
 return {name:file.name,dims:dim,voxel,count,buffer,dtype};
}
function updateViewer(){
 const volume=S.mri,has=!!volume;
 $('mriEmpty').style.display=has?'none':'flex';
 $('mriTag').textContent=has?(S.recordQuality==='live'?'本次实验 FLAIR · 本地':'真实 MRI · 未配对实验'):'未导入影像';
 $('mriFilename').textContent=has?volume.name+' · '+volume.dims.join('×'):'等待导入真实 .nii/.nii.gz';
 $('viewKind').textContent=S.renderMode==='slice'?'AXIAL':'3D VOXEL PROJECTION';
 $('sliceSlider').disabled=!has;
 if(has){
  $('sliceSlider').min=0;$('sliceSlider').max=S.renderMode==='slice'?volume.dims[2]-1:359;
  $('sliceSlider').value=S.renderMode==='slice'?S.slice:S.yaw;
  $('sliceValue').textContent=S.renderMode==='slice'?S.slice+'/'+(volume.dims[2]-1):S.yaw+'°';
 }
 $('roiValue').textContent=S.opacity+'%';
 $('sliceMode').classList.toggle('active',S.renderMode==='slice');
 $('voxelMode').classList.toggle('active',S.renderMode==='voxel');
 if(has)drawMRI();
}
function intensityRange(volume){
 const sampled=[],step=Math.max(1,Math.floor(volume.count/7000));
 for(let i=0;i<volume.count;i+=step){const v=volume.voxel(i);if(v!==0&&Number.isFinite(v))sampled.push(v);}
 if(!sampled.length)return[0,1];
 sampled.sort((a,b)=>a-b);
 const lo=sampled[Math.floor(sampled.length*.02)],hi=sampled[Math.floor(sampled.length*.98)];
 return [lo,hi>lo?hi:lo+1];
}
function paintMRI2D(canvas,volume,z){
 const [nx,ny]=volume.dims,ctx=canvas.getContext('2d');
 const bmp=document.createElement('canvas');bmp.width=nx;bmp.height=ny;
 const bc=bmp.getContext('2d');const img=bc.createImageData(nx,ny),range=volume.range||(volume.range=intensityRange(volume));
 const mask=S.mask&&S.mask.dims.join()===volume.dims.join()?S.mask:null;
 const o=S.opacity/100,offset=nx*ny*z;
 for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
  const index=x+nx*y+offset,intensity=volume.voxel(index);
  const g=Math.round(Math.max(0,Math.min(255,255*(intensity-range[0])/(range[1]-range[0]))));
  const k=(x+nx*(ny-y-1))*4;img.data[k]=g;img.data[k+1]=g;img.data[k+2]=g;img.data[k+3]=255;
  if(mask&&o){const label=mask.voxel(index);
   if(label>0){const c=label===4?[114,228,212]:label===2?[251,193,117]:[238,113,113];for(let j=0;j<3;j++)img.data[k+j]=Math.round((1-o)*g+o*c[j]);}
  }
 }
 bc.putImageData(img,0,0);
 const {width:w,height:h}=canvas;ctx.fillStyle='#06111a';ctx.fillRect(0,0,w,h);
 const size=Math.min(w/nx,h/ny)*.91;const dw=nx*size,dh=ny*size;
 ctx.imageSmoothingEnabled=false;ctx.drawImage(bmp,(w-dw)/2,(h-dh)/2,dw,dh);
 ctx.strokeStyle='#315265';ctx.lineWidth=1;ctx.strokeRect((w-dw)/2,(h-dh)/2,dw,dh);
}
function buildCloud(vol){
 const [nx,ny,nz]=vol.dims,step=Math.ceil(Math.max(nx,ny,nz)/34),list=[];
 const range=vol.range||(vol.range=intensityRange(vol));const mask=S.mask&&S.mask.dims.join()===vol.dims.join()?S.mask:null;
 for(let z=0;z<nz;z+=step)for(let y=0;y<ny;y+=step)for(let x=0;x<nx;x+=step){
  const i=x+nx*(y+ny*z),val=vol.voxel(i);
  if(!Number.isFinite(val)||val<=range[0]+(range[1]-range[0])*.22)continue;
  const seg=mask?mask.voxel(i):0;
  if(list.length<8500)list.push({x:x/nx-.5,y:y/ny-.5,z:z/nz-.5,b:(val-range[0])/(range[1]-range[0]),seg});
 }
 return list;
}
function paintMRI3D(canvas,vol){
 const ctx=canvas.getContext('2d'),{width:w,height:h}=canvas;
 ctx.fillStyle='#06111a';ctx.fillRect(0,0,w,h);
 if(!S.pointCloud)S.pointCloud=buildCloud(vol);
 const angle=S.yaw*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);
 const pts=S.pointCloud.map(p=>{
  const rx=p.x*c-p.y*s,ry=p.x*s+p.y*c;
  const zz=p.z*.85+ry*.3;
  return {x:w/2+rx*430,y:h/2-zz*440,z:ry,seg:p.seg,b:p.b};
 }).sort((a,b)=>a.z-b.z);
 for(const p of pts){
  const a=.12+.45*Math.max(0,Math.min(1,p.b)),gray=Math.round(110+100*p.b);
  let color='rgba('+gray+','+(gray+18)+','+(gray+20)+','+a+')';
  if(p.seg&&S.opacity>0){const rgb=p.seg===4?'105,224,201':p.seg===2?'246,185,102':'228,111,121';color='rgba('+rgb+','+(S.opacity/100*.72)+')';}
  ctx.fillStyle=color;ctx.fillRect(p.x,p.y,2.2,2.2);
 }
 ctx.strokeStyle='#2d5260';ctx.strokeRect(8,8,w-16,h-16);
 ctx.fillStyle='#80aeba';ctx.font='14px ui-monospace,monospace';ctx.fillText('VOXEL PROJECTION / YAW '+S.yaw+'°',20,34);
}
function drawMRI(){
 const c=$('mriCanvas');if(!S.mri)return;
 if(S.renderMode==='slice')paintMRI2D(c,S.mri,S.slice);else paintMRI3D(c,S.mri);
}
async function importMRI(file,isMask=false){
 if(!file)return;
 try{
  notice('正在本地解析 '+file.name+' …');
  const image=await loadNifti(file);
  if(isMask){
   if(S.mri&&image.dims.join()!==S.mri.dims.join())throw Error('ROI 和 MRI 的体素维度不一致，无法覆盖显示');
   S.mask=image;S.pointCloud=null;event('导入 ROI 掩膜',image.dims.join('×'),file.name);
  }else{
   S.mri=image;S.mask=null;S.pointCloud=null;S.slice=Math.floor(image.dims[2]/2);
   event('导入 MRI',image.dims.join('×'),file.name);
  }
  updateViewer();notice('真实 NIfTI 已在浏览器中载入；尚未证明它与当前实验记录属于同一患者。数据未自动上传。');
 }catch(ex){notice('NIfTI 导入失败：'+ex.message,true)}
 finally{$(isMask?'maskInput':'mriInput').value=''}
}
function snapshot(){
 const snap={type:'visual_state_only',at:new Date().toISOString(),case_id:S.record.case_id,
   record_source:S.recordQuality,model_sha256:S.record.checkpoint_sha256||S.record.weights_sha256,
   view:S.renderMode,slice:S.slice,yaw:S.yaw,roi_opacity:S.opacity,selected_stage:S.selectedStage,
   selected_path_index:S.selectedPath,mri_loaded:!!S.mri,roi_loaded:!!S.mask,
   warning:'A visual state is NOT an additional inference, intervention, or scientific measurement.'};
 event('视觉状态快照',S.renderMode==='slice'?'slice='+S.slice:'yaw='+S.yaw+'°','Browser UI');
 S.events[0].snapshot=snap;renderEvents();notice('已记录视觉状态。导出审计包可保存该状态。');
}
function downloadFile(name,blob){
 const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);
}
function exportAudit(){
 const payload={export_kind:'brainmirror/observatory-audit/v1',exported_at:new Date().toISOString(),
 source_quality:S.recordQuality,record:S.record,events:S.events,history:S.history,
 scope:'browser session audit only; no MRI voxel data, no unverified activation graphs'};
 downloadFile('scept_observatory_audit.json',new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
 event('导出审计包','JSON · 无原始 MRI','Browser export');
}
function modal(open){$('modal').classList.toggle('hidden',!open)}
function install(){
 document.querySelectorAll('.side-item[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
 $('traceInput').addEventListener('change',e=>importJSON(e.target.files[0]));
 $('mriInput').addEventListener('change',e=>importMRI(e.target.files[0],false));
 $('maskInput').addEventListener('change',e=>importMRI(e.target.files[0],true));
 $('sliceMode').addEventListener('click',()=>{S.renderMode='slice';updateViewer()});
 $('voxelMode').addEventListener('click',()=>{S.renderMode='voxel';updateViewer()});
 $('architectureMode').addEventListener('click',()=>{S.graphMode='architecture';$('architectureMode').classList.add('active');$('pathMode').classList.remove('active');renderGraph()});
 $('pathMode').addEventListener('click',()=>{S.graphMode='path';$('pathMode').classList.add('active');$('architectureMode').classList.remove('active');renderGraph()});
 $('sliceSlider').addEventListener('input',e=>{if(S.renderMode==='slice')S.slice=Number(e.target.value);else S.yaw=Number(e.target.value);updateViewer()});
 $('roiOpacity').addEventListener('input',e=>{S.opacity=Number(e.target.value);updateViewer()});
 $('snapshotBtn').addEventListener('click',snapshot);
 $('downloadMriBtn').addEventListener('click',()=>{if(!S.mri)return notice('请先导入真实 MRI 图像',true);$('mriCanvas').toBlob(blob=>blob&&downloadFile('scept_mri_'+S.renderMode+'.png',blob),'image/png');event('导出图像','本地 MRI 视图 PNG','Browser export')});
 $('exportBtn').addEventListener('click',exportAudit);
 $('showSourceBtn').addEventListener('click',()=>setView('evidence'));
 $('copyJson').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('rawJson').textContent);notice('实验 JSON 已复制')}catch{notice('浏览器禁止剪贴板访问，请手动复制 JSON',true)}});
 $('newRunBtn').addEventListener('click',()=>modal(true));
 $('closeModal').addEventListener('click',()=>modal(false));
 $('modalOk').addEventListener('click',()=>modal(false));
 $('modal').addEventListener('click',e=>{if(e.target===$('modal'))modal(false)});
 document.addEventListener('keydown',e=>{if(e.key==='Escape')modal(false)});
 let down=false,last=0;
 $('mriCanvas').addEventListener('pointerdown',e=>{if(S.renderMode!=='voxel'||!S.mri)return;down=true;last=e.clientX;$('mriCanvas').setPointerCapture(e.pointerId)});
 $('mriCanvas').addEventListener('pointermove',e=>{if(!down)return;S.yaw=(S.yaw+(e.clientX-last)*.6+360)%360;last=e.clientX;updateViewer()});
 $('mriCanvas').addEventListener('pointerup',()=>down=false);
 $('mriCanvas').addEventListener('pointercancel',()=>down=false);
 event('历史真实实验记录','BraTS20_Training_002 · screenshot transcription','Kaggle screenshot, 2026-10-16');
 updateRecord();updateViewer();setView('observatory');
}
install();
