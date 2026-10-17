'use strict';
/* Autoload a pinned, public, pretrained Tiny ViT into the browser.
   No training. All forward passes use the user's own device.
   Before use, verify the original weights against the pinned SHA-256.
   The mid-block attention value tensor is exposed as an additional graph output;
   the original learned initializer tensors are not altered. */
(function(){
const url='https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/resolve/ebffd7b9c0c53b51bada44116abd8f01aed41be6/onnx/model.onnx';
const expected='32f1cf3b4868cc44757814cbf6237ae5a3658a45980bf0fe93914dca77b815e3';
const tensor='/vit/encoder/layer.6/attention/attention/value/MatMul_output_0';
const $=id=>document.getElementById(id);
function encodeVarint(n){const a=[];do{let v=n%128;n=Math.floor(n/128);if(n)v+=128;a.push(v)}while(n);return Uint8Array.from(a)}
function readVarint(a,p,end){let n=0,mul=1;for(let i=0;i<9&&p<end;i++){const v=a[p++];n+=(v&127)*mul;if(!(v&128))return{value:n,end:p};mul*=128}throw Error('Invalid ONNX protobuf varint')}
function fields(a,start,end,fn){
 let p=start;
 while(p<end){
  const keyStart=p,tag=readVarint(a,p,end);p=tag.end;const field=Math.floor(tag.value/8),wire=tag.value%8;
  if(wire===2){const l=readVarint(a,p,end);const payloadStart=l.end,payloadEnd=payloadStart+l.value;
    if(!Number.isSafeInteger(payloadEnd)||payloadEnd>end)throw Error('Invalid ONNX protobuf length');
    fn({field,wire,keyStart,payloadStart,payloadEnd,fieldEnd:payloadEnd});
    p=payloadEnd;
  }else if(wire===0){p=readVarint(a,p,end).end}
  else if(wire===1){p+=8}
  else if(wire===5){p+=4}
  else throw Error('Unsupported ONNX protobuf field');
  if(p>end)throw Error('Malformed ONNX protobuf boundaries');
 }
}
function concat(...blocks){let len=0;for(const b of blocks)len+=b.length;const result=new Uint8Array(len);let p=0;for(const b of blocks){result.set(b,p);p+=b.length}return result}
function exposeIntermediate(modelBuffer,name){
 const a=new Uint8Array(modelBuffer);if(a.length>150000000)throw Error('ONNX model too large for browser instrumentation');
 let graphField=null;
 fields(a,0,a.length,f=>{if(f.field===7&&f.wire===2)graphField=f});
 if(!graphField)throw Error('Model does not contain ONNX graph');
 let found=null,alreadyOutput=false;
 const decoder=new TextDecoder('utf-8');
 fields(a,graphField.payloadStart,graphField.payloadEnd,f=>{
  if(f.field!==13&&f.field!==12)return;
  let label=null;
  fields(a,f.payloadStart,f.payloadEnd,v=>{if(v.field===1&&v.wire===2)label=decoder.decode(a.subarray(v.payloadStart,v.payloadEnd))});
  if(label===name){if(f.field===12)alreadyOutput=true;else found=a.slice(f.payloadStart,f.payloadEnd)}
 });
 if(alreadyOutput)return modelBuffer;
 if(!found)throw Error('Requested ViT intermediate tensor metadata not found: '+name);
 const observedOutput=concat(encodeVarint((12<<3)|2),encodeVarint(found.length),found);
 const newGraph=concat(a.subarray(graphField.payloadStart,graphField.payloadEnd),observedOutput);
 const newModel=concat(a.subarray(0,graphField.keyStart),encodeVarint((7<<3)|2),encodeVarint(newGraph.length),newGraph,a.subarray(graphField.fieldEnd));
 return newModel.buffer;
}
const banner=$('transformerStatus'),button=$('transformerDemo');
if(!button)return;
button.onclick=async()=>{
 button.disabled=true;const old=button.textContent;
 try{
  banner.textContent='Downloading official pretrained ViT weights (23.1 MB)…';
  const resp=await fetch(url,{mode:'cors',redirect:'follow',cache:'force-cache'});
  if(!resp.ok)throw Error('Model download failed HTTP '+resp.status);
  const original=await resp.arrayBuffer();
  banner.textContent='Verifying original pretrained weight SHA-256…';
  const originalHash=await sha256(original);
  if(originalHash!==expected)throw Error('Published pretrained ViT SHA-256 mismatch, refusing to run');
  banner.textContent='Exposing genuine layer-6 Attention value tensor (no retraining)…';
  const instrumented=exposeIntermediate(original,tensor);
  await selectModel(instrumented,'Pretrained Tiny ViT · layer-6 observed (FP32)','VERIFIED_PRETRAINED_TRANSFORMER');
  window.SCEPT_SOURCE_MODEL_SHA256=originalHash;
  $('experimentKind').value='input';
  $('runtimeSelect').value='wasm';
  banner.textContent='Initializing real browser CPU inference…';
  await $('loadBtn').onclick();
  if(!window.SCEPT_DEMO_CHECK_READY())throw Error($('notify').textContent||'ViT ONNX session did not initialize');
  const data=new Float32Array(1*3*224*224);
  for(let i=0;i<data.length;i++)data[i]=((i*73)%256)/255;
  window.SCEPT_IMPORT_VOLUME_TENSOR({dims:[1,3,224,224],data});
  $('offset').value='0';$('count').value='1';$('strength').value='0.15';$('perturbMethod').value='add';
  banner.textContent='Running frozen ViT twice and comparing real internal activations…';
  await $('runBtn').onclick();
  if(!window.SCEPT_DEMO_HAS_RESULT())throw Error($('notify').textContent||'Transformer forward pass did not return numeric output');
  $('probeSelect').value=tensor;$('probeSelect').dispatchEvent(new Event('change'));
  banner.textContent='PASS · Pretrained ViT, verified source SHA '+originalHash.slice(0,12)+'…, layer-6 hidden tensor '+$('changed').textContent+' values changed. Test input is synthetic and NOT an MRI.';
  $('statusPill').textContent='PRETRAINED TRANSFORMER OBSERVED';
 }catch(e){
  banner.textContent='Cannot complete one-click Transformer test: '+e.message+'. You can still import a pretrained ONNX manually.';
  $('notify').textContent=e.message;
 }finally{button.disabled=false;button.textContent=old;}
};
window.SCEPT_DEMO_EXPOSE=exposeIntermediate;
})();