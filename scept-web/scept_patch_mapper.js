'use strict';
/* Render EXACT ViT 224x224 slice/patches. No anatomical association is inferred. */
(function(){
 const $=id=>document.getElementById(id),canvas=$('patchInputCanvas');
 if(!canvas)return;
 const s={mapping:null,selected:null,revision:0,changedPixels:0};
 function refresh(){
  if(!window.SCEPT_MRI?.state?.loaded||!window.SCEPT_TENSOR_ADAPTER)return null;
  try{
   const base=SCEPT_MRI.makeSlice(false,2),edited=SCEPT_MRI.makeSlice(true,2);
   const map=SCEPT_TENSOR_ADAPTER.mapInput(base,edited);
   s.mapping=map;s.revision=SCEPT_MRI.state.revision;
   const ctx=canvas.getContext('2d'),img=ctx.createImageData(224,224);
   for(let i=0;i<224*224;i++){
    const value=Math.round(Math.max(0,Math.min(1,edited[i]))*255);
    img.data[4*i]=img.data[4*i+1]=img.data[4*i+2]=value;img.data[4*i+3]=255;
   }
   ctx.putImageData(img,0,0);
   for(const p of map.patches)if(p.changed){
    ctx.fillStyle='rgba(55,205,159,.09)';ctx.fillRect(p.col*16,p.row*16,16,16);
    ctx.lineWidth=1.5;ctx.strokeStyle='#52e6b0';ctx.strokeRect(p.col*16+.75,p.row*16+.75,14.5,14.5);
   }
   ctx.lineWidth=.35;ctx.strokeStyle='rgba(170,220,220,.25)';
   for(let i=1;i<14;i++){ctx.beginPath();ctx.moveTo(i*16,0);ctx.lineTo(i*16,224);ctx.moveTo(0,i*16);ctx.lineTo(224,i*16);ctx.stroke()}
   if(s.selected!==null){
    const x=s.selected%14,y=Math.floor(s.selected/14);
    ctx.strokeStyle='#f8ba6a';ctx.lineWidth=2.6;ctx.strokeRect(x*16+1.4,y*16+1.4,13.2,13.2);
   }
   $('patchCounts').textContent=map.changedPixels.toLocaleString('zh-CN')+' 个输入像素改变 · '+map.changedPatches+'/196 个 Patch 改变';
   $('patchStatus').textContent=map.empty?'目前没有实际输入差异':'已核对实际模型输入';
   $('patchSelection').textContent=s.selected===null?'点击 14×14 网格查看对应 Token':('选中 Patch '+s.selected+' → Token '+(s.selected+1)+(map.affectedPatchIds.includes(s.selected)?' · 输入确有改变':' · 输入未改变'));
   $('patchClear').disabled=s.selected===null;
   window.SCEPT_ACTIVITY3D?.syncPatchMap(map,s.selected);
   return map;
  }catch(err){$('patchStatus').textContent='Patch 映射失败：'+err.message;return null}
 }
 function choosePatch(patch){
  if(!Number.isInteger(patch)||patch<0||patch>=196)return;
  s.selected=patch;
  refresh();
  window.SCEPT_ACTIVITY3D?.focusPatch(patch);
 }
 function updateFromEvent(e){
  const rect=canvas.getBoundingClientRect(),col=Math.floor((e.clientX-rect.left)/rect.width*14),row=Math.floor((e.clientY-rect.top)/rect.height*14);
  choosePatch(Math.min(13,Math.max(0,row))*14+Math.min(13,Math.max(0,col)));
 }
 canvas.addEventListener('click',updateFromEvent);
 canvas.addEventListener('keydown',e=>{if(e.key!=='Enter'&&e.key!==' ')return;e.preventDefault();choosePatch(s.selected===null?0:s.selected)});
 $('patchClear').onclick=()=>{s.selected=null;refresh();window.SCEPT_ACTIVITY3D?.focusPatch(null)};
 // MRI is a separate, already-running module; connect at the viewer render point.
 window.SCEPT_PATCH={refresh,choosePatch,getSnapshot:()=>({map:s.mapping,selected:s.selected,revision:s.revision})};
})();
