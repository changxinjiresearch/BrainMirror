/** BrainMirror local-only, native FreeSurfer/FastSurfer cortex + aseg viewer.
 * Individual MRI derivatives MUST NOT be auto-uploaded to the public site.
 * This viewer is NOT registered to the public IXI 467-person group mean.
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js';

let renderer,scene,camera,brain,container,canvas;
let data=null, meshItems=[], callback=null,mode='none';
let selecting='',showLH=true,showRH=true,showDeep=false;
let pointerDown=null,dist=310;
const raycaster=new THREE.Raycaster();
const pointer=new THREE.Vector2();

function parseBase64(src,kind,expected){
  if(typeof src!=='string'||src.length>85000000)throw Error('Invalid oversized anatomical mesh');
  const bin=atob(src);
  if(bin.length!==expected*(kind==='f32'||kind==='u32'?4:2))throw Error('Surface array size mismatch');
  const buffer=new ArrayBuffer(bin.length);
  const arr=new Uint8Array(buffer);
  for(let i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);
  if(kind==='f32')return new Float32Array(buffer);
  if(kind==='u32')return new Uint32Array(buffer);
  return new Uint16Array(buffer);
}
function validateAndMake(obj,role){
  const nv=Number(obj.vertex_count),nf=Number(obj.face_count);
  if(!Number.isInteger(nv)||!Number.isInteger(nf)||nv<30||nv>500000||nf<100||nf>1500000)throw Error('Unreasonable anatomy mesh dimensions');
  const pos=parseBase64(obj.positions_f32_b64,'f32',nv*3);
  const faces=parseBase64(obj.faces_u32_b64,'u32',nf*3);
  for(let i=0;i<pos.length;i++)if(!Number.isFinite(pos[i])||Math.abs(pos[i])>900)throw Error('Anatomy coordinates invalid');
  for(let i=0;i<faces.length;i++)if(faces[i]>=nv)throw Error('Mesh face vertex index out of bounds');
  const labels=role==='surface'?parseBase64(obj.region_indices_u16_b64,'u16',nv):null;
  const names=role==='surface'?obj.region_names:null;
  if(role==='surface'&&(!Array.isArray(names)||names.length<10||names.length>512))throw Error('Missing cortical parcellation');
  if(labels){for(let i=0;i<labels.length;i++)if(labels[i]!==65535&&labels[i]>=names.length)throw Error('Invalid annotation index')}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setIndex(new THREE.BufferAttribute(faces,1));
  geo.computeVertexNormals();
  let material;
  if(role==='surface'){
    geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(nv*3),3));
    material=new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:.92,metalness:0});
  }else{
    material=new THREE.MeshStandardMaterial({color:0xb5d5d5,side:THREE.DoubleSide,roughness:.86,metalness:0,transparent:true,opacity:.94});
  }
  const mesh=new THREE.Mesh(geo,material);
  mesh.userData={role,id:role==='surface'?obj.hemisphere:'sub:'+obj.id,labels,names,regionName:obj.name};
  return mesh;
}
function recolor(){
  for(const mesh of meshItems){
    if(mesh.userData.role!=='surface')continue;
    const {labels,id}=mesh.userData;
    const att=mesh.geometry.getAttribute('color');
    const chosen=selecting.startsWith(id+':')?Number(selecting.split(':')[1]):-1;
    for(let i=0;i<labels.length;i++){
      const isTarget=chosen>=0&&labels[i]===chosen;
      const background=chosen>=0 ? .27 : 1;
      att.setXYZ(i,isTarget?1.0:.82*background,isTarget?.58:.71*background,isTarget?.16:.65*background);
    }
    att.needsUpdate=true;
  }
  for(const mesh of meshItems){
    if(mesh.userData.role!=='subcortical')continue;
    const picked=selecting==='sub:'+mesh.userData.id.split(':')[1];
    mesh.material.color.set(picked?0xffb84a:0x80cbd5);
  }
}
function visibility(){
  for(const mesh of meshItems){
    if(mesh.userData.id==='lh')mesh.visible=showLH;
    if(mesh.userData.id==='rh')mesh.visible=showRH;
    if(mesh.userData.role==='subcortical')mesh.visible=showDeep;
  }
  recolor();render();
}
function render(){if(renderer&&scene&&camera)renderer.render(scene,camera)}
function size(){
 if(!renderer)return;
 const w=Math.max(300,container.clientWidth),h=Math.max(260,container.clientHeight);
 renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();render();
}
function setCamera(){
  brain.position.set(0,0,0);
  const box=new THREE.Box3().setFromObject(brain);const mid=new THREE.Vector3();const span=new THREE.Vector3();
  box.getCenter(mid);box.getSize(span);brain.position.sub(mid);
  dist=Math.max(120,span.length()*1.15);
  camera.position.set(dist*.70,-dist*.70,dist*.25);camera.position.setLength(dist);
  camera.lookAt(0,0,0);brain.rotation.set(0,0,0);render();
}
export function mount(root){
  if(renderer)return;
  container=root;
  if(!container)throw Error('Brain 3D container unavailable');
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.5));
  renderer.setClearColor(0,0);renderer.outputColorSpace=THREE.SRGBColorSpace;
  canvas=renderer.domElement;
  canvas.className='native-brain-canvas';
  canvas.style.cssText='position:absolute;inset:0;width:100%;height:100%;z-index:6;display:none;touch-action:none;cursor:grab';
  container.appendChild(canvas);
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,1,.1,1500);camera.up.set(0,0,1);
  scene.add(new THREE.HemisphereLight(0xedf6ff,0x5c7891,2.3));
  const l1=new THREE.DirectionalLight(0xfff4ea,3.0);l1.position.set(140,-210,260);scene.add(l1);
  const l2=new THREE.DirectionalLight(0xadd9ff,1.8);l2.position.set(-200,140,70);scene.add(l2);
  brain=new THREE.Group();scene.add(brain);
  canvas.addEventListener('pointerdown',e=>{
    pointerDown={x:e.clientX,y:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing';e.stopPropagation();
  });
  canvas.addEventListener('pointermove',e=>{
    if(!pointerDown)return;
    const dx=e.clientX-pointerDown.x,dy=e.clientY-pointerDown.y;
    if(Math.abs(dx)+Math.abs(dy)>2)pointerDown.moved=true;
    brain.rotation.z+=dx*.006;brain.rotation.x=Math.max(-1.1,Math.min(1.1,brain.rotation.x+dy*.006));
    pointerDown.x=e.clientX;pointerDown.y=e.clientY;render();e.stopPropagation();
  });
  canvas.addEventListener('pointerup',e=>{
    const shouldPick=pointerDown&&!pointerDown.moved;
    pointerDown=null;canvas.style.cursor='grab';
    if(shouldPick&&data){
      const rect=canvas.getBoundingClientRect();
      pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);
      raycaster.setFromCamera(pointer,camera);
      const hit=raycaster.intersectObjects(meshItems.filter(x=>x.visible),false)[0];
      if(hit){
        const mesh=hit.object;
        const role=mesh.userData.role;
        if(role==='surface'){
          const labels=mesh.userData.labels,idx=labels[hit.face.a];
          if(idx!==65535){
            selecting=mesh.userData.id+':'+idx;
            const name=mesh.userData.names[idx];
            callback?.({key:selecting,name,source:'native FreeSurfer annotation'});
          }
        }else{
          selecting=mesh.userData.id;
          callback?.({key:selecting,name:mesh.userData.regionName,source:'native FreeSurfer aseg'});
        }
        recolor();render();
      }
    }
    e.stopPropagation();
  });
  canvas.addEventListener('wheel',e=>{
    const mult=Math.exp(e.deltaY*.001);
    camera.position.multiplyScalar(Math.max(.8,Math.min(1.25,mult)));
    const length=camera.position.length();
    camera.position.setLength(Math.max(75,Math.min(950,length)));
    camera.lookAt(0,0,0);render();e.preventDefault();e.stopPropagation();
  },{passive:false});
  window.addEventListener('resize',size);
  new ResizeObserver(size).observe(container);
  size();
}
export function loadNativePack(pack,onSelect){
  if(pack?.schema!=='brainmirror.native-anatomy.v1'||pack.provenance?.source_kind!=='individual_native_surface'||pack.provenance?.individual_data_private_only!==true||pack.provenance?.registered_to_ixi_population_mean!==false)throw Error('Not a verified private native anatomy pack');
  if(!Array.isArray(pack.hemispheres)||pack.hemispheres.length!==2||!pack.hemispheres.some(h=>h.hemisphere==='lh')||!pack.hemispheres.some(h=>h.hemisphere==='rh'))throw Error('Both native cortical hemispheres are required');
  if(!Array.isArray(pack.subcortical)||pack.subcortical.length>36)throw Error('Invalid subcortical list');
  if(!renderer)throw Error('3D viewer not mounted');
  let incoming=[];
  try{
    for(const h of pack.hemispheres)incoming.push(validateAndMake(h,'surface'));
    for(const r of pack.subcortical)incoming.push(validateAndMake(r,'subcortical'));
  }catch(e){for(const mesh of incoming){mesh.geometry.dispose();mesh.material.dispose()}throw e}
  for(const mesh of meshItems){brain.remove(mesh);mesh.geometry.dispose();mesh.material.dispose()}
  meshItems=incoming;
  for(const mesh of meshItems)brain.add(mesh);
  data=pack;callback=onSelect;selecting='';showLH=true;showRH=true;showDeep=false;
  setCamera();visibility();show(true);
  return {hemispheres:pack.hemispheres.map(h=>({hemi:h.hemisphere,vertices:h.vertex_count,regions:h.region_names})),subcortical:pack.subcortical.map(s=>({key:'sub:'+s.id,name:s.name})),source_kind:pack.provenance.source_kind};
}
export function setSelection(value){selecting=String(value||'');recolor();render()}
export function setVisibility(opts){showLH=opts.left!==false;showRH=opts.right!==false;showDeep=!!opts.deep;visibility()}
export function show(value){if(!canvas)return;mode=value?'native':'none';canvas.style.display=value?'block':'none';render()}
export function reset(){if(!renderer||!brain)return;setCamera()}
export function isLoaded(){return !!data}