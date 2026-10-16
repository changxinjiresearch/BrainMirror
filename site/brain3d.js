/**
 * BrainMirror IXI T1 population reference — continuous brain-only surface.
 * This is an actual isosurface extracted from the sanitized group MRI preview.
 * Not a pial reconstruction, cortical parcellation, neuron map, or inference.
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js';

let renderer, scene, camera, group, root, frame, canvas, current;
let dragging = false, px = 0, py = 0;
let zoom = 1;
let naturalDistance = 350;

const TETRAS = [
  [0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
  [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]
];
const CORNERS = [
  [0,0,0],[1,0,0],[1,1,0],[0,1,0],
  [0,0,1],[1,0,1],[1,1,1],[0,1,1]
];

function densityField(volume) {
  const dims = volume.shape;
  const size = dims.map(n => Math.max(14, Math.ceil((n - 1) / Math.ceil(Math.max(...dims) / 72)) + 1));
  const [nx,ny,nz] = size;
  const ix = (x,y,z) => x + nx * (y + ny*z);
  const base = new Float32Array(nx*ny*nz);
  const srcIx = (x,y,z) => x + dims[0] * (y + dims[1]*z);

  for (let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const xx=Math.min(dims[0]-1,Math.round(x*(dims[0]-1)/(nx-1)));
    const yy=Math.min(dims[1]-1,Math.round(y*(dims[1]-1)/(ny-1)));
    const zz=Math.min(dims[2]-1,Math.round(z*(dims[2]-1)/(nz-1)));
    // Brain-masked group MRI: nonzero intensity is the anatomical support.
    // A low nonzero cutoff avoids constructing interior grey-white interfaces.
    base[ix(x,y,z)] = volume.data[srcIx(xx,yy,zz)] > 0.012 ? 1 : 0;
  }
  const soften = data => {
    const output = new Float32Array(data.length);
    for(let z=1;z<nz-1;z++)for(let y=1;y<ny-1;y++)for(let x=1;x<nx-1;x++){
      const i=ix(x,y,z);
      const neighbor=data[ix(x-1,y,z)]+data[ix(x+1,y,z)]
         +data[ix(x,y-1,z)]+data[ix(x,y+1,z)]
         +data[ix(x,y,z-1)]+data[ix(x,y,z+1)];
      output[i]=(4*data[i]+neighbor)/10;
    }
    return output;
  };
  // Smooth low-res quantization; do not describe as anatomical segmentation.
  return {dims:size,data:soften(soften(base))};
}

function makeSurface(volume) {
  const {dims, data} = densityField(volume);
  const [nx,ny,nz]=dims;
  const at=(x,y,z)=>data[x+nx*(y+ny*z)];
  const origin = volume.affine;
  if(!Array.isArray(origin)||origin.length<3) throw Error('Missing physical MRI affine');
  const scaling=volume.shape.map((n,i)=>(n-1)/(dims[i]-1));
  const m=new THREE.Matrix3().set(
    origin[0][0]*scaling[0],origin[0][1]*scaling[1],origin[0][2]*scaling[2],
    origin[1][0]*scaling[0],origin[1][1]*scaling[1],origin[1][2]*scaling[2],
    origin[2][0]*scaling[0],origin[2][1]*scaling[1],origin[2][2]*scaling[2]
  );
  const det=m.determinant();
  if(!Number.isFinite(det)||Math.abs(det)<1e-6)throw Error('MRI affine is singular');
  const nm=m.clone().invert().transpose();
  const g=(x,y,z)=>new THREE.Vector3(
    at(Math.min(nx-1,x+1),y,z)-at(Math.max(0,x-1),y,z),
    at(x,Math.min(ny-1,y+1),z)-at(x,Math.max(0,y-1),z),
    at(x,y,Math.min(nz-1,z+1))-at(x,y,Math.max(0,z-1))
  );
  const point=(x,y,z)=>{
    const v=new THREE.Vector3(x,y,z).applyMatrix3(m);
    return v.add(new THREE.Vector3(origin[0][3],origin[1][3],origin[2][3]));
  };
  const positions=[], normals=[];
  let polygons=0;
  const ISO=0.38;
  function interpolate(a,b){
    const d=b.v-a.v;
    const t=Math.max(0,Math.min(1,Math.abs(d)<1e-7?.5:(ISO-a.v)/d));
    const world=point(
      a.x+t*(b.x-a.x),
      a.y+t*(b.y-a.y),
      a.z+t*(b.z-a.z)
    );
    const normal=g(a.x,a.y,a.z).multiplyScalar(1-t)
       .add(g(b.x,b.y,b.z).multiplyScalar(t))
       .multiplyScalar(-1).applyMatrix3(nm).normalize();
    return {p:world,n:normal};
  }
  function tri(a,b,c){
    // Face winding is checked against the outward density-field gradient.
    const face=new THREE.Vector3().subVectors(b.p,a.p)
        .cross(new THREE.Vector3().subVectors(c.p,a.p));
    const expected=a.n.clone().add(b.n).add(c.n);
    if(face.dot(expected)<0){const t=b;b=c;c=t;}
    for(const x of [a,b,c]){
      positions.push(x.p.x,x.p.y,x.p.z);
      normals.push(x.n.x,x.n.y,x.n.z);
    }
    polygons++;
  }
  for(let z=1;z<nz-2;z++)for(let y=1;y<ny-2;y++)for(let x=1;x<nx-2;x++){
    const cube=CORNERS.map(([dx,dy,dz])=>({
      x:x+dx,y:y+dy,z:z+dz,v:at(x+dx,y+dy,z+dz)
    }));
    let allInside=true,allOutside=true;
    for(const c of cube){if(c.v>=ISO)allOutside=false;else allInside=false;}
    if(allInside||allOutside)continue;
    for(const ids of TETRAS){
      const inside=[],outside=[];
      for(const id of ids){
        if(cube[id].v>=ISO)inside.push(cube[id]);
        else outside.push(cube[id]);
      }
      if(inside.length===0||inside.length===4)continue;
      if(inside.length===1){
        const a=inside[0];
        tri(interpolate(a,outside[0]),interpolate(a,outside[1]),interpolate(a,outside[2]));
      }else if(inside.length===3){
        const a=outside[0];
        tri(interpolate(a,inside[0]),interpolate(a,inside[2]),interpolate(a,inside[1]));
      }else{
        const [a,b]=inside,[c,d]=outside;
        const ac=interpolate(a,c),ad=interpolate(a,d);
        const bc=interpolate(b,c),bd=interpolate(b,d);
        tri(ac,ad,bc);tri(ad,bd,bc);
      }
    }
  }
  if(polygons<300||polygons>850000) throw Error('Invalid MRI surface: '+polygons+' polygons');
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('normal',new THREE.Float32BufferAttribute(normals,3));
  geometry.computeBoundingSphere();
  const sphere=geometry.boundingSphere.clone();
  geometry.translate(-sphere.center.x,-sphere.center.y,-sphere.center.z);
  return {geometry,polygons,radius:sphere.radius};
}

function redraw(){
  if(renderer&&scene&&camera) renderer.render(scene,camera);
}
function fit(){
  if(!renderer||!root)return;
  const w=Math.max(300,root.clientWidth),h=Math.max(260,root.clientHeight);
  renderer.setSize(w,h,false);
  camera.aspect=w/h;
  camera.updateProjectionMatrix();
  redraw();
}
export function mount({container}) {
  root=container;
  if(!root)throw Error('Brain viewport missing');
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.5));
  renderer.setClearColor(0x000000,0);
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.domElement.style.cssText='position:absolute;inset:0;width:100%;height:100%;z-index:3;touch-action:none;cursor:grab;';
  renderer.domElement.className='brainmesh-canvas';
  root.appendChild(renderer.domElement);
  canvas=renderer.domElement;
  const old=document.getElementById('brainCanvas');
  if(old)old.style.visibility='hidden';
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(38,1,0.1,2000);
  camera.up.set(0,0,1);
  scene.add(new THREE.HemisphereLight(0xe6f5ff,0x637c97,2.0));
  const key=new THREE.DirectionalLight(0xffffff,3.2);
  key.position.set(200,-160,280);scene.add(key);
  const fill=new THREE.DirectionalLight(0xb4d7ff,1.4);
  fill.position.set(-160,160,60);scene.add(fill);
  group=new THREE.Group();scene.add(group);
  const move=e=>{
    if(!dragging)return;
    group.rotation.z+=(e.clientX-px)*0.007;
    group.rotation.x=Math.max(-1.2,Math.min(1.2,group.rotation.x+(e.clientY-py)*0.006));
    px=e.clientX;py=e.clientY;redraw();
  };
  canvas.addEventListener('pointerdown',e=>{
    dragging=true;px=e.clientX;py=e.clientY;
    canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing';
    e.stopPropagation();
  });
  canvas.addEventListener('pointermove',e=>{move(e);if(dragging)e.stopPropagation()});
  canvas.addEventListener('pointerup',e=>{dragging=false;canvas.style.cursor='grab';e.stopPropagation()});
  canvas.addEventListener('pointercancel',()=>{dragging=false;canvas.style.cursor='grab'});
  canvas.addEventListener('wheel',e=>{
    zoom=Math.max(0.55,Math.min(2.7,zoom*Math.exp(e.deltaY*0.001)));
    camera.position.setLength(naturalDistance*zoom);
    camera.lookAt(0,0,0);redraw();e.preventDefault();e.stopPropagation();
  },{passive:false});
  window.addEventListener('resize',fit);
  new ResizeObserver(fit).observe(root);
  fit();
  return true;
}
export function loadVolume(volume) {
  if(!renderer||!group)throw Error('3D renderer not initialized');
  const {geometry,polygons,radius}=makeSurface(volume);
  if(current){group.remove(current);current.geometry.dispose();current.material.dispose();}
  const material=new THREE.MeshStandardMaterial({
    color:0xd9c4b4,roughness:0.94,metalness:0,
    side:THREE.DoubleSide
  });
  current=new THREE.Mesh(geometry,material);
  group.add(current);
  group.rotation.set(0,0,0);
  naturalDistance=Math.max(100,radius*3.3);
  zoom=1;
  camera.position.set(naturalDistance*.68,-naturalDistance*.74,naturalDistance*.40);
  camera.position.setLength(naturalDistance);
  camera.lookAt(0,0,0);
  const caption=document.getElementById('brainCanvasTop');
  if(caption)caption.textContent='IXI 467 · REAL MRI-DERIVED 3D SURFACE · '+polygons.toLocaleString()+' TRIANGLES';
  const hint=document.querySelector('#brainSpace .hint');
  if(hint)hint.textContent='真实 MRI 派生表面 · 拖动旋转 · 滚轮缩放 · 非皮层精细重建';
  redraw();
  return {polygons,origin:'real-ixi-group-volume',reviewed:false};
}
