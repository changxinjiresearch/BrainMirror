'use strict';
/* V5: genuine paired ONNX NDF / multi-counterfactual CRC. ICM remains gated. */
(function(root){if(!root?.document)return;
root.document.addEventListener('DOMContentLoaded',()=>{
const $=id=>document.getElementById(id),S=root.SCEPT_V5_SCIENCE;
if(!$('v5Observe')||!S)return;
let current=null,all=[],mask=null,busy=false,activeRevision=null;
const fmt=x=>Number.isFinite(x)?x.toPrecision(5):'—';
const info=(x,error=false)=>{ $('v5Status').textContent=x;$('v5Status').dataset.error=String(error) };
const hash=async x=>{
 const bytes=x instanceof Uint8Array?x:new Uint8Array(x.buffer,x.byteOffset,x.byteLength);
 const h=await crypto.subtle.digest('SHA-256',bytes);
 return [...new Uint8Array(h)].map(v=>v.toString(16).padStart(2,'0')).join('');
};
const choose=()=>root.SCEPT_V3?.getScientificContext?.();
function geom(a,b){
 if(!b||a.dimensions?.some((v,i)=>v!==b.dimensions[i]))throw Error('NIfTI 体素网格不一致');
 if(a.spacing.some((v,i)=>Math.abs(v-b.spacing[i])>1e-3))throw Error('NIfTI 体素间距不一致');
 if(!a.affine||!b.affine)throw Error('正式配对实验缺少 MRI sform 空间变换，不能假定图像配准');
 for(let i=0;i<3;i++)for(let j=0;j<4;j++)
 if(Math.abs(a.affine[i][j]-b.affine[i][j])>.001)throw Error('A/B/Mask 空间坐标不一致');
}
const parse=async f=>{
 if(!f||f.size>120000000)throw Error('请提供小于120MB的 .nii/.nii.gz 文件');
 return root.SCEPT_MRI.decodeFile(await f.arrayBuffer(),f.name,{allowFlat:true});
};
const predict=async(input,ctx)=>ctx.session.run({
 pixel_values:new root.ort.Tensor('float32',input,[1,3,224,224])
});
function render(record){
 current=record;
 const rows=record.rows,cv=$('v5Fingerprint'),g=cv.getContext('2d'),
 width=Math.max(430,Math.round(cv.getBoundingClientRect().width||820)),ratio=Math.min(2,root.devicePixelRatio||1),
 height=rows.length*43+75;
 cv.width=Math.round(width*ratio);cv.height=Math.round(height*ratio);
 g.setTransform(ratio,0,0,ratio,0,0);g.fillStyle='#071923';g.fillRect(0,0,width,height);
 const left=80,right=width-17,maximum=Math.max(1e-12,...rows.flatMap(r=>r.values.map(v=>v.rms)));
 g.fillStyle='#c2d9db';g.font='12px system-ui';g.fillText('真实内部响应差异 / 同层真实 Token 列位置',left,18);
 rows.forEach((row,i)=>{
  const w=(right-left)/row.units,y=29+43*i;
  g.fillStyle='#abcdd1';g.fillText('Layer '+(row.layer??i),6,y+15);
  row.values.forEach((v,j)=>{
   const z=Math.log1p(25*v.rms/maximum)/Math.log(26);
   g.fillStyle=v.rms<=1e-12?'#253c4b':z>.72?'#edaa65':z>.36?'#62cdbd':'#306881';
   g.fillRect(left+j*w,y,Math.max(1,w*.96),25);
  });
 });
 g.fillStyle='#81a8ad';g.fillText('本次视觉色阶对数增强，不构成统计显著性或跨层统一量纲',left,height-13);
 cv.__grid={left,right,top:29,step:43,rows,ratio};
 $('v5NdfInfo').textContent=rows.length+' 层 · '+rows.reduce((a,r)=>a+r.units,0)+' 个真实计算位置 · '+record.label;
 $('v5NdfConclusion').textContent=record.identicalInputs?
   '本次实际模型输入未发生变化：零差异对照。':'NDF 测量了真实 A/B 神经活动变化，但不能单独证明病理依赖或内部因果关系。';
 $('v5Export').disabled=false;
}
function show3d(records,ctx,a,b){
 const layers=ctx.probes.map(p=>{
  const x=a[p.name],y=b[p.name];if(!x?.data||!y?.data)throw Error('缺少实际内部张量 '+p.name);
  let sum=0,max=0,changed=0,up=0,down=0;
  for(let i=0;i<x.data.length;i++){const d=y.data[i]-x.data[i],v=Math.abs(d);
   sum+=v;max=Math.max(max,v);if(v>1e-7){changed++;d>0?up++:down++}}
  return{layer:p.layer,probe:p.name,shape:[...x.dims],
   before:Array.from(x.data),after:Array.from(y.data),
   summary:{mean:sum/x.data.length,max,changed,up,down,total:x.data.length,groups:[]}};
 });
 root.SCEPT_ACTIVITY3D?.update(layers,root.SCEPT_MRI.state.revision);
 $('viewTokens')?.click();$('tokenOverview')?.click();$('responseDelta')?.click();
}
$('v5Fingerprint').addEventListener('click',e=>{
 if(!current)return;const cv=e.currentTarget,g=cv.__grid,rect=cv.getBoundingClientRect(),
 x=(e.clientX-rect.left)*cv.width/rect.width/g.ratio,
 y=(e.clientY-rect.top)*cv.height/rect.height/g.ratio,
 layer=Math.floor((y-g.top)/g.step);
 if(layer<0||layer>=g.rows.length)return;
 const r=g.rows[layer],unit=Math.floor((x-g.left)/(g.right-g.left)*r.units);
 if(unit<0||unit>=r.units)return;
 const v=r.values[unit];
 $('v5Selection').textContent='真实 '+r.name+' · Token/通道 '+unit+
 ' · RMS '+fmt(v.rms)+' · MAE '+fmt(v.mae)+
 ' · A |激活| '+fmt(v.meanAbsA)+' · B |激活| '+fmt(v.meanAbsB);
 root.SCEPT_ACTIVITY3D?.select(r.layer,unit);
});
async function runNumeric(){
 if($('modelMode').value!=='vit'||!root.SCEPT_V3?.getSnapshot().ready)throw Error('请等待真实 ViT 与 MRI 就绪');
 const before=root.SCEPT_V3.getSnapshot().runId;
 await root.SCEPT_V3.run();
 const a=root.SCEPT_V3.getAudit();
 if(!a||a.at===before||a.mri.revision!==root.SCEPT_MRI.state.revision)throw Error('没有完成本次真实 A/B 推理');
 all=[];mask=null;activeRevision=a.mri.revision;
 return S.ndf({modelSha256:a.model_sha256,baselineHash:a.baseline_input_sha256,
 counterfactualHash:a.intervened_input_sha256,profile:'vit-token-channel',
 label:'exploratory-intensity-edit-not-lesion-removal',
 layers:a.layers.map(l=>({name:l.probe,layer:l.layer,
  a:{dims:l.shape,data:l.before},b:{dims:l.shape,data:l.after}}))});
}
async function runPair(){
 if($('modelMode').value!=='vit'||!root.SCEPT_V3?.getSnapshot().ready)throw Error('当前真实 A/B 指纹仅支持 ViT 观测模型');
 const bFiles=[...$('v5BFiles').files],maskFile=$('v5Mask').files[0],mri=root.SCEPT_MRI.state;
 if(!bFiles.length||!maskFile)throw Error('正式反事实模式必须上传 B MRI 与病灶掩膜');
 if(bFiles.length>12)throw Error('一次最多12份反事实 MRI');
 const M=await parse(maskFile);geom(mri,M);
 const count=M.values.reduce((n,v)=>n+(v>.5?1:0),0);
 if(count<5)throw Error('有效病灶掩膜体素不足');
 const nx=mri.dimensions[0],ny=mri.dimensions[1],k=mri.slice[2],offset=k*nx*ny;
 if(!M.values.slice(offset,offset+nx*ny).some(v=>v>.5))
  throw Error('当前 MRI 轴位未穿过掩膜，请先切换到有病灶的层面');
 const vols=[];for(const f of bFiles){const v=await parse(f);geom(mri,v);vols.push({name:f.name,v})}
 const ctx=choose();if(!ctx?.session)throw Error('真实 ONNX 模型尚未加载');
 const X=root.SCEPT_MRI.makeSlice(false,2),ax=await hash(X);
 info('真实 ONNX：正在执行原始 MRI A');
 const A=await predict(X,ctx),records=[],fingerprints=new Set();
 for(let i=0;i<vols.length;i++){
  const b=root.SCEPT_MRI.makeSlice(false,2,vols[i].v.values),bh=await hash(b);
  if(fingerprints.has(bh))throw Error('重复的反事实模型输入，拒绝伪重复');
  fingerprints.add(bh);info('真实 ONNX：反事实 B '+(i+1)+'/'+vols.length);
  const B=await predict(b,ctx);
  const rec=S.ndf({modelSha256:ctx.sha256,baselineHash:ax,counterfactualHash:bh,
   profile:'vit-token-channel',label:'uploaded-mri-counterfactual-not-validated-pathology',
   layers:ctx.probes.map(p=>({name:p.name,layer:p.layer,a:A[p.name],b:B[p.name]}))});
  if(!i)show3d(records,ctx,A,B);
  records.push(rec);
 }
 if(root.SCEPT_MRI.state.revision!==mri.revision)throw Error('运行期间 MRI 或层面已变更；拒绝混淆不同实验');
 all=records;mask={filename:maskFile.name,maskSha256:await hash(M.values),count,
  warnings:['NIfTI几何通过不代表伪健康组织合理','本 ViT 不是临床医学模型',
  '肿瘤和修复伪影尚未通过病理特异性控制实验']};
 activeRevision=mri.revision;
 $('v5MaskInfo').textContent='掩膜体素：'+count+' · B 版本：'+records.length+' · 真实同网格 MRI；组织修复质量尚未被验证';
 $('v5Crc').textContent=all.length>=3?'可运行 CRC（≥3 个不同 B 输入）。':'只有 '+all.length+' 个反事实版本，CRC 尚不可评价（至少3个）。';
 return records[0];
}
async function doRun(task,id){
 if(busy)return;busy=true;$(id).disabled=true;
 try{
  info(task==='pair'?'准备真实 MRI A/B + 掩膜…':'执行同一区域的实际强度 A/B…');
  const rec=task==='pair'?await runPair():await runNumeric();
  render(rec);$('v5CrcRun').disabled=all.length<3;
  info(task==='pair'?'真实配对反事实 NDF 完成；科学结论仍受反事实质量限制。':'真实强度编辑 NDF 完成；不属于肿瘤移除。');
 }catch(e){info('本次不可评价：'+e.message,true)}
 finally{busy=false;$(id).disabled=false}
}
$('v5Observe').onclick=()=>doRun('numeric','v5Observe');
$('v5PairRun').onclick=()=>doRun('pair','v5PairRun');
$('v5CrcRun').onclick=()=>{
 try{
  if(!all.length||root.SCEPT_MRI.state.revision!==activeRevision)throw Error('当前 MRI 与反事实记录不一致');
  const r=S.crc(all);
  $('v5Crc').textContent=r.rows.map(x=>'L'+x.layer+' Top-K重复度='+fmt(x.topKOverlap)+'；排序相关='+fmt(x.rankCorrelation)).join(' ｜ ')+
   '。仅比较同一病例多个反事实结果，不是独立患者统计。';
  root.SCEPT_V5_CRC=r;info('CRC 完成：全部指标来自真实多份 B 的 ONNX 激活。');
 }catch(e){$('v5Crc').textContent='不可评价：'+e.message}
};
$('v5Export').onclick=()=>{
 if(!current)return;
 const data={schema:'scept.v5.evidence.v1',ndf:current,crc:root.SCEPT_V5_CRC||null,
  provenance:mask,scientificClaims:{clinical:'NOT_ESTABLISHED',causal:'NOT_ESTABLISHED',
   pathology_specificity:'NOT_ESTABLISHED',m4:'NOT_ACHIEVED'}};
 const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),
 link=document.createElement('a');link.href=url;link.download='scept-v5-genuine-neural-fingerprint.json';link.click();
 setTimeout(()=>URL.revokeObjectURL(url),1500);
};
$('v5Icm').textContent='内部激活替换须使用真实改写后的可执行 ONNX，当前页面不提供假设性贡献热图。';
$('v5Selection').textContent='点击真实差异指纹某一格，三维点云将定位到该层和 Token。';
info('就绪后点击「NDF 强度实验」，或上传真实 MRI B 与病灶掩膜运行配对实验。');
root.SCEPT_V5={getSnapshot:()=>({hasNdf:!!current,hasCrc:!!root.SCEPT_V5_CRC,
  measuredRealizations:all.length,modelSha256:current?.modelSha256,mode:current?.label}),
 getNdf:()=>current};
});
})(typeof window==='undefined'?null:window);
