'use strict';
/* Browser-local NIfTI-1 MRI workbench.
   3D native voxel axes, no implied anatomical world orientation.
   Exploratory perturbations only, NOT anatomically or clinically validated. */
(function(){
const byId=id=>document.getElementById(id);
const mount=byId('mriMount'); if(!mount)return;
mount.innerHTML=`
<div class="mri-note">NIfTI-1 (.nii / .nii.gz) · local 3-axis inspection · reversible ROI edits. These are native voxel planes, not verified anatomical orientations.</div>
<div class="mri-toolbar">
<label class="upload subtle">＋ Import local MRI (.nii / .nii.gz)<input type="file" id="mriUpload" accept=".nii,.nii.gz"></label>
<select id="mriMode"><option value="intensity">Intensity-only ROI edit</option><option value="geometry">Smooth geometric displacement (exploratory)</option></select>
<button class="ghost" id="mriApply" disabled>Apply ROI edit</button><button class="ghost" id="mriReset" disabled>Undo all edits</button>
</div>
<div class="mri-panels">
<div class="mri-view"><span>Native i-j plane</span><canvas id="mriAxial" width="240" height="240"></canvas><input id="sliceZ" type="range" min="0" max="1" value="0"></div>
<div class="mri-view"><span>Native i-k plane</span><canvas id="mriCoronal" width="240" height="240"></canvas><input id="sliceY" type="range" min="0" max="1" value="0"></div>
<div class="mri-view"><span>Native j-k plane</span><canvas id="mriSagittal" width="240" height="240"></canvas><input id="sliceX" type="range" min="0" max="1" value="0"></div>
<div class="mri-view"><span>Volume projection · drag to rotate</span><canvas id="mriVolume" width="240" height="240"></canvas><div class="micro">Sampled voxels · projected, not surface segmentation</div></div>
</div>
<div class="mri-controls"><label>ROI sphere radius (voxels)<input type="number" min="1" max="64" value="8" id="mriRadius"></label><label>Change magnitude<input type="number" step=".05" value=".1" id="mriMagnitude"></label><label>Model input normalization<select id="mriNorm"><option value="raw">Keep input MRI intensities</option><option value="unit">Normalize to [0,1] based on initial MRI percentiles</option></select></label><button id="mriToModel" class="ghost" disabled>Use edited MRI as model input →</button><button id="mriExport" class="ghost" disabled>Export edited NIfTI ↓</button></div>
<div id="mriInfo" class="mri-report" aria-live="polite">No volume imported. Nothing is uploaded to the server.</div>`;
let mri=null,volume=null,raw=null,roi=null,cloud=null,rotation=.6,angle=.3,percentiles=[0,1],lastQC=null;
const index=(x,y,z,d)=>x+d[0]*(y+d[1]*z);
const finite=x=>typeof x==='number'&&Number.isFinite(x);
function niftiHeader(buf){
 if(buf.byteLength<352)throw Error('NIfTI file too small');
 const h=new DataView(buf);let le=true;
 if(h.getInt32(0,true)!==348){if(h.getInt32(0,false)!==348)throw Error('Not a NIfTI-1 single-file header');le=false}
 const magic=String.fromCharCode(...new Uint8Array(buf,344,3));if(magic!=='n+1')throw Error('Only single-file NIfTI-1 (.nii) supported');
 const num=h.getInt16(40,le);if(num!==3&&num!==4)throw Error('Only 3D or single-frame 4D NIfTI is supported');
 const d=[h.getInt16(42,le),h.getInt16(44,le),h.getInt16(46,le)];
 if(d.some(x=>x<2)||d.reduce((a,b)=>a*b,1)>12000000)throw Error('Invalid volume dimensions or volume exceeds 12M voxels');
 if(num===4&&h.getInt16(48,le)!==1)throw Error('Multi-frame MRI must be selected/converted before loading');
 const dtype=h.getInt16(70,le),bitpix=h.getInt16(72,le);
 const types={2:[1,'getUint8'],4:[2,'getInt16'],8:[4,'getInt32'],16:[4,'getFloat32'],64:[8,'getFloat64'],512:[2,'getUint16'],768:[4,'getUint32']};
 if(!types[dtype]||bitpix!==8*types[dtype][0])throw Error('Unsupported NIfTI datatype / bitpix');
 const offset=h.getFloat32(108,le),dataOffset=Math.ceil(offset);
 const n=d.reduce((a,b)=>a*b,1),bytes=types[dtype][0],method=types[dtype][1];
 if(!finite(offset)||offset<352||dataOffset+n*bytes>buf.byteLength)throw Error('Invalid NIfTI voxel data offset or truncated payload');
 const slope=h.getFloat32(112,le),intercept=h.getFloat32(116,le),m=slope===0?1:slope,t=slope===0?0:intercept;
 if(!finite(m)||!finite(t))throw Error('Invalid NIfTI scaling');
 const values=new Float32Array(n);
 for(let i=0;i<n;i++){
   const val=method==='getUint8'?h.getUint8(dataOffset+i):h[method](dataOffset+i*bytes,le),scaled=val*m+t;
   if(!finite(scaled))throw Error('Non-finite MRI voxel data');values[i]=scaled;
 }
 const px=[h.getFloat32(80,le),h.getFloat32(84,le),h.getFloat32(88,le)];
 if(px.some(v=>!finite(v)||v<=0))throw Error('Invalid voxel spacing');
 const sf=h.getInt16(254,le),qf=h.getInt16(252,le);
 let affine=[],affineSource='';
 if(sf>0){
  affine=Array.from({length:3},(_,r)=>Array.from({length:4},(_,j)=>h.getFloat32(280+r*16+j*4,le)));
  affineSource='sform';
 }else if(qf>0){
  const b=h.getFloat32(256,le),c=h.getFloat32(260,le),d0=h.getFloat32(264,le),
   a=Math.sqrt(Math.max(0,1-b*b-c*c-d0*d0)),qfac=h.getFloat32(76,le)<0?-1:1;
  const matrix=[
  [a*a+b*b-c*c-d0*d0,2*(b*c-a*d0),2*(b*d0+a*c)],
  [2*(b*c+a*d0),a*a+c*c-b*b-d0*d0,2*(c*d0-a*b)],
  [2*(b*d0-a*c),2*(c*d0+a*b),a*a+d0*d0-c*c-b*b]
  ];
  const scales=[px[0],px[1],px[2]*qfac];
  affine=matrix.map((row,r)=>[...row.map((v,j)=>v*scales[j]),h.getFloat32(268+r*4,le)]);
  affineSource='qform';
 }else{
  affine=[[px[0],0,0,0],[0,px[1],0,0],[0,0,px[2],0]];
  affineSource='unknown_orientation_spacing_only';
 }
 if(affine.some(row=>row.some(v=>!finite(v))))throw Error('Invalid NIfTI affine');
 return {dims:d,datatype:dtype,bytes,le,values,spacing:px,affine,affineSource,
   rawHeader:new Uint8Array(buf.slice(0,352)),dataOffset,originalBytes:buf.byteLength};
}
async function readNifti(file){
 let buffer=await file.arrayBuffer();
 if(file.name.toLowerCase().endsWith('.gz')){
  if(typeof DecompressionStream==='undefined')throw Error('Browser gzip decompression unavailable');
  const readable=new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  buffer=await new Response(readable).arrayBuffer();
 }
 return niftiHeader(buffer);
}
function percentile(values){
 const sample=[];const step=Math.max(1,Math.floor(values.length/8000));
 for(let i=0;i<values.length;i+=step)sample.push(values[i]);
 sample.sort((a,b)=>a-b);
 return [sample[Math.floor(sample.length*.02)],sample[Math.floor(sample.length*.98)]];}
function normByte(v){const a=percentiles[0],b=percentiles[1];return Math.round(255*Math.min(1,Math.max(0,(v-a)/(b-a||1))))}
function center(){return [Number(byId('sliceX').value),Number(byId('sliceY').value),Number(byId('sliceZ').value)]}
function generateROI(){
 const d=mri.dims,[cx,cy,cz]=center(),rad=Number(byId('mriRadius').value);
 if(!finite(rad)||rad<1||rad>64)throw Error('ROI radius must be 1–64 voxels');
 const out=new Uint8Array(raw.length),r2=rad*rad;
 for(let z=Math.max(0,Math.floor(cz-rad));z<=Math.min(d[2]-1,Math.ceil(cz+rad));z++)
 for(let y=Math.max(0,Math.floor(cy-rad));y<=Math.min(d[1]-1,Math.ceil(cy+rad));y++)
 for(let x=Math.max(0,Math.floor(cx-rad));x<=Math.min(d[0]-1,Math.ceil(cx+rad));x++)
 if((x-cx)**2+(y-cy)**2+(z-cz)**2<=r2)out[index(x,y,z,d)]=1;
 return out;
}
function interpolate(data,d,x,y,z){
 x=Math.min(d[0]-1,Math.max(0,x));y=Math.min(d[1]-1,Math.max(0,y));z=Math.min(d[2]-1,Math.max(0,z));
 const x0=Math.floor(x),y0=Math.floor(y),z0=Math.floor(z),x1=Math.min(d[0]-1,x0+1),y1=Math.min(d[1]-1,y0+1),z1=Math.min(d[2]-1,z0+1);
 const a=x-x0,b=y-y0,c=z-z0,lerp=(v0,v1,t)=>v0*(1-t)+v1*t;
 const xy=(zz)=>lerp(lerp(data[index(x0,y0,zz,d)],data[index(x1,y0,zz,d)],a),lerp(data[index(x0,y1,zz,d)],data[index(x1,y1,zz,d)],a),b);
 return lerp(xy(z0),xy(z1),c);
}
function editROI(kind,magnitude,mask){
 if(!finite(magnitude)||Math.abs(magnitude)>3)throw Error('Magnitude outside safe exploratory range [-3,3]');
 const edited=new Float32Array(raw),d=mri.dims,rad=Number(byId('mriRadius').value),[cx,cy,cz]=center();
 let changed=0,outside=0,jacMin=Infinity,roiCount=0;
 for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=0;x<d[0];x++){
  const ii=index(x,y,z,d);if(!mask[ii])continue;roiCount++;
  if(kind==='intensity'){edited[ii]=raw[ii]+magnitude;}
  else{
   const xx=(x-cx),yy=(y-cy),zz=(z-cz),r2=xx*xx+yy*yy+zz*zz,sigma=Math.max(1,rad/2);
   const displacement=magnitude*Math.exp(-r2/(2*sigma*sigma));
   const jac=1-displacement*xx/(sigma*sigma);jacMin=Math.min(jacMin,jac);
   edited[ii]=interpolate(raw,d,x-displacement,y,z);
  }
  if(Math.abs(edited[ii]-raw[ii])>1e-6)changed++;
 }
 if(kind==='geometry'&&jacMin<=.2)throw Error('Local Jacobian QC failed: excessive displacement');
 return {edited,report:{mode:kind,roi_voxels:roiCount,changed_voxels:changed,changed_outside_roi:outside,
    minimum_approx_jacobian:kind==='geometry'?jacMin:null,
    anatomical_validity:'NOT_VALIDATED',note:'Exploratory voxel-space transformation, not clinically valid anatomy.'}};
}
function sliceCanvas(id,plane){
 if(!mri)return;const c=byId(id),ctx=c.getContext('2d'),d=mri.dims,[cx,cy,cz]=center();
 const dims=plane===0?[d[0],d[1]]:plane===1?[d[0],d[2]]:[d[1],d[2]];
 const tmp=document.createElement('canvas');tmp.width=dims[0];tmp.height=dims[1];const t=tmp.getContext('2d'),im=t.createImageData(...dims);
 for(let y=0;y<dims[1];y++)for(let x=0;x<dims[0];x++){
   const [a,b,z]=plane===0?[x,y,cz]:plane===1?[x,cy,y]:[cx,x,y],idx=index(a,b,z,d),g=normByte(volume[idx]),j=(x+(dims[1]-1-y)*dims[0])*4;
   im.data[j]=g;im.data[j+1]=g;im.data[j+2]=g;if(roi&&roi[idx]){im.data[j]=Math.min(255,g+90);im.data[j+1]=Math.floor(g*.5);}im.data[j+3]=255;
 }
 t.putImageData(im,0,0);ctx.clearRect(0,0,c.width,c.height);ctx.imageSmoothingEnabled=false;ctx.drawImage(tmp,0,0,c.width,c.height);
}
function buildCloud(){
 if(!mri)return;const d=mri.dims,pts=[],skip=Math.max(2,Math.ceil(Math.cbrt(raw.length/10000))),threshold=percentiles[0]+(percentiles[1]-percentiles[0])*.75;
 for(let z=0;z<d[2];z+=skip)for(let y=0;y<d[1];y+=skip)for(let x=0;x<d[0];x+=skip){
  const val=volume[index(x,y,z,d)];if(val>=threshold)pts.push([(x/d[0]-.5)*1.8,(y/d[1]-.5)*1.8,(z/d[2]-.5)*1.8,val]);
 }
 cloud=pts.slice(0,16000);
}
function renderCloud(){
 const c=byId('mriVolume'),ctx=c.getContext('2d');ctx.fillStyle='#071820';ctx.fillRect(0,0,c.width,c.height);if(!cloud)return;
 const ca=Math.cos(angle),sa=Math.sin(angle),cr=Math.cos(rotation),sr=Math.sin(rotation);
 const projected=cloud.map(p=>{const [x,y,z,v]=p;const x1=cr*x-sr*y,y1=sr*x+cr*y,z1=ca*z-sa*y1;return [c.width/2+x1*105,c.height/2+(sa*z+ca*y1)*105,z1,v]});
 projected.sort((a,b)=>a[2]-b[2]);
 for(const [x,y,z,v] of projected){const g=normByte(v);ctx.fillStyle='rgba('+Math.floor(g*.55)+','+Math.min(255,g+45)+','+Math.min(255,g+65)+',.55)';ctx.fillRect(x,y,2,2)}
}
function render(){if(!mri)return;roi=generateROI();sliceCanvas('mriAxial',0);sliceCanvas('mriCoronal',1);sliceCanvas('mriSagittal',2);buildCloud();renderCloud();}
function report(extra=''){if(!mri)return;const d=mri.dims;byId('mriInfo').textContent='Size '+d.join('×')+' voxels · spacing '+mri.spacing.map(x=>x.toFixed(2)).join('×')+' mm · affine '+mri.affineSource+' · voxel axes not auto-reoriented. '+extra;}
byId('mriUpload').onchange=async e=>{
 const f=e.target.files[0];if(!f)return;byId('mriInfo').textContent='Reading MRI locally…';
 try{mri=await readNifti(f);raw=new Float32Array(mri.values);volume=new Float32Array(raw);lastQC=null;roi=null;percentiles=percentile(raw);
  ['sliceX','sliceY','sliceZ'].forEach((id,i)=>{const el=byId(id);el.max=mri.dims[i]-1;el.value=Math.floor(mri.dims[i]/2)});
  ['mriApply','mriReset','mriToModel','mriExport'].forEach(id=>byId(id).disabled=false);render();report('Imported '+f.name);
 }catch(err){mri=null;raw=null;volume=null;byId('mriInfo').textContent='Import failed: '+err.message}
};
['sliceX','sliceY','sliceZ','mriRadius'].forEach(id=>byId(id).oninput=()=>{try{render()}catch(e){byId('mriInfo').textContent=e.message}});
byId('mriApply').onclick=()=>{try{roi=generateROI();const res=editROI(byId('mriMode').value,Number(byId('mriMagnitude').value),roi);volume=res.edited;lastQC=res.report;render();report('QC: '+JSON.stringify(lastQC))}catch(e){byId('mriInfo').textContent='Edit rejected: '+e.message}};
byId('mriReset').onclick=()=>{volume=new Float32Array(raw);lastQC=null;render();report('Restored exactly to imported intensity array.')};
byId('mriToModel').onclick=()=>{
 if(!mri)return;try{
  const data=new Float32Array(volume);
  if(byId('mriNorm').value==='unit'){const lo=percentiles[0],span=percentiles[1]-lo||1;for(let i=0;i<data.length;i++)data[i]=Math.min(1,Math.max(0,(data[i]-lo)/span))}
  if(!window.SCEPT_IMPORT_VOLUME_TENSOR)throw Error('Inference workbench not initialized');
  const d=mri.dims;
  window.SCEPT_IMPORT_VOLUME_TENSOR({dims:[1,1,d[2],d[1],d[0]],data,source:'real_nifti',affine:mri.affine,mode:lastQC?.mode||'none',qc:lastQC});
  report('Prepared model tensor in memory [1,1,z,y,x]. Check model-specific preprocessing before running.');
 }catch(e){byId('mriInfo').textContent='Tensor bridge failed: '+e.message}
};
byId('mriExport').onclick=()=>{
 if(!mri)return;const out=new ArrayBuffer(352+volume.length*4),header=new Uint8Array(out,0,352);header.set(mri.rawHeader);const h=new DataView(out);h.setInt16(70,16,mri.le);h.setInt16(72,32,mri.le);h.setFloat32(108,352,mri.le);h.setFloat32(112,1,mri.le);h.setFloat32(116,0,mri.le);
 for(let i=0;i<volume.length;i++)h.setFloat32(352+i*4,volume[i],mri.le);
 const url=URL.createObjectURL(new Blob([out],{type:'application/octet-stream'})),a=document.createElement('a');a.href=url;a.download='scept-edited-mri.nii';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
let lastMouse=null;
byId('mriVolume').onpointerdown=e=>{lastMouse=[e.clientX,e.clientY];e.currentTarget.setPointerCapture(e.pointerId)};
byId('mriVolume').onpointermove=e=>{if(!lastMouse)return;const [x,y]=lastMouse;rotation+=(e.clientX-x)*.015;angle+=(e.clientY-y)*.015;lastMouse=[e.clientX,e.clientY];renderCloud()};
byId('mriVolume').onpointerup=()=>lastMouse=null;
window.SCEPT_NIFTI={parse:niftiHeader};
})();