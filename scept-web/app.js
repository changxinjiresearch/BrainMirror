'use strict';
const $=id=>document.getElementById(id);
let session=null,bytes=null,sha=null,selectedData=null,last=null,history=[],modelClass="UPLOADED_MODEL_UNVERIFIED",modelFileName="",externalMRI=null,graph=null;
try{history=JSON.parse(localStorage.getItem('scept-audit-v2')||'[]')}catch{}
const fmt=n=>Number.isFinite(n)?(Math.abs(n)>1e4?n.toExponential(3):n.toFixed(6)):'—';
function metaFor(name,which='input'){const names=which==='input'?session?.inputNames:session?.outputNames;const arr=which==='input'?session?.inputMetadata:session?.outputMetadata;return arr?.[names?.indexOf(name)]||graph?.[which+'Metadata']?.[name]||{};}

const err=e=>{$('notify').textContent=typeof e==='string'?e:(e?.message||String(e));};
const clear=()=>{$('notify').textContent='';};
const sumshape=a=>Array.isArray(a)?a.join('×'):String(a||'?');
const sha256=async buffer=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))).map(b=>b.toString(16).padStart(2,'0')).join('');
function available(){if(!window.ort)throw Error('ONNX Runtime Web could not load. Check network/CDN connectivity.');}
document.querySelectorAll('.navitem').forEach(b=>b.onclick=()=>{document.querySelectorAll('.navitem').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.view').forEach(x=>x.classList.add('hidden'));$(b.dataset.target).classList.remove('hidden');if(b.dataset.target==='experiments')renderHistory()});
async function selectModel(source,name,cls='UPLOADED_MODEL_UNVERIFIED'){clear();if(session){await session.release();session=null;}bytes=source;sha=await sha256(bytes);modelClass=cls;modelFileName=name;externalMRI=null;try{graph=window.SCEPT_INSPECT_ONNX(source);renderGraph()}catch(e){graph=null;$('graphMeta').textContent='Graph inspector unavailable: '+e.message;$('graphNodes').replaceChildren()};$('modelName').textContent=name+' · '+(bytes.byteLength/1048576).toFixed(3)+' MB';$('loadBtn').disabled=false;$('modelHash').textContent=sha.slice(0,16)+'…';last=null;selectedData=null;$('exportBtn').disabled=true;$('exportCsv').disabled=true;$('probeSelect').disabled=true;$('probeSelect').innerHTML='<option>No current results</option>';$('mae').textContent='—';$('maxd').textContent='—';$('changed').textContent='—';$('statusPill').textContent=cls==='UPLOADED_MODEL_UNVERIFIED'?'MODEL SELECTED':'ENGINEERING FIXTURE';$('runBtn').disabled=true;}
$('modelFile').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{await selectModel(await f.arrayBuffer(),f.name);}catch(ex){err(ex)}};
async function selectFixture(key){try{const fixture=window.SCEPT_FIXTURES[key],bin=atob(fixture.b64),data=Uint8Array.from(bin,c=>c.charCodeAt(0)).buffer;await selectModel(data,fixture.name,fixture.classification);if(sha!==fixture.sha)throw Error('Engineering fixture integrity mismatch');$('tensorText').value=JSON.stringify(fixture.input);$('experimentKind').value=key==='ablation'?'ablation':'input';$('offset').value='0';$('count').value='1';await $('loadBtn').onclick();}catch(ex){err(ex)}}
$('demoLinear').onclick=()=>selectFixture('linear');$('demoAblation').onclick=()=>selectFixture('ablation');
$('loadBtn').onclick=async()=>{clear();try{available();if(!bytes)throw Error('Select a model file first.');$('loadBtn').disabled=true;$('execMode').textContent='Loading…';ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';ort.env.wasm.numThreads=1;const provider=$('runtimeSelect').value;if(provider==='webgpu'&&!navigator.gpu)throw Error('WebGPU unsupported on this device; select CPU/WASM.');session=await ort.InferenceSession.create(new Uint8Array(bytes),{executionProviders:[provider]});$('execMode').textContent=provider.toUpperCase()+' · local inference';$('engineState').textContent='● '+provider.toUpperCase()+' · active';$('statusPill').textContent='REAL MODEL LOADED';$('inputDefs').textContent=session.inputNames.map((n,i)=>n+': '+sumshape(metaFor(n).shape||'?')).join(' | ');$('outputDefs').textContent=session.outputNames.map((n,i)=>n+': '+sumshape(metaFor(n,'output').shape||'?')).join(' | ');$('inputTensor').innerHTML=session.inputNames.filter(n=>n!=='scept_mask').map(n=>'<option value="'+escapeHtml(n)+'">'+escapeHtml(n)+'</option>').join('');$('runBtn').disabled=false;$('generateProbe').disabled=false;$('modelName').textContent+=' · Ready';}catch(e){err(e);$('execMode').textContent='Load failed';}finally{$('loadBtn').disabled=false}};
const escapeHtml=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
$('tensorFile').onchange=async e=>{const f=e.target.files[0];if(f)$('tensorText').value=await f.text();};
function parseData(){if(externalMRI)return externalMRI;let d;try{d=JSON.parse($('tensorText').value)}catch{throw Error('Input must be valid JSON.')}if(d&&d.feeds)d=d.feeds[$('inputTensor').value];if(!d||!Array.isArray(d.shape)||!Array.isArray(d.data))throw Error('Required JSON: {shape,data} or {feeds:{inputName:{shape,data}}}.');const size=d.shape.reduce((a,b)=>a*b,1);if(!d.shape.length||!d.shape.every(v=>Number.isInteger(v)&&v>0)||size!==d.data.length||size>10000000)throw Error('Invalid shape or tensor length (10M element safety cap).');if(!d.data.every(v=>typeof v==='number'&&Number.isFinite(v)))throw Error('Input tensor must contain finite numeric values.');return{dims:d.shape,data:Float32Array.from(d.data)}}
function perturb(input){const out=new Float32Array(input.data);const start=Number($('offset').value),cnt=Number($('count').value),v=Number($('strength').value),op=$('perturbMethod').value;if(!Number.isInteger(start)||start<0||!Number.isInteger(cnt)||cnt<1||start+cnt>out.length||!Number.isFinite(v))throw Error('Perturbation range or magnitude invalid.');for(let i=start;i<start+cnt;i++){out[i]=op==='add'?out[i]+v:op==='multiply'?out[i]*v:0;}return{dims:input.dims,data:out}}
function toReadable(t){if(!t||!t.data||!t.dims)throw Error('Output is not a dense numeric tensor.');if(t.data.length>5000000)throw Error('Output too large for in-browser audit (5M limit).');return{shape:Array.from(t.dims),values:Array.from(t.data,Number)}}
async function run(){clear();if(!session)throw Error('Load an ONNX model first.');const kind=$('experimentKind').value,original=parseData(),edited=kind==='ablation'?original:perturb(original),name=$('inputTensor').value;const otherNames=session.inputNames.filter(n=>n!==name);
let suppliedFeeds={};try{suppliedFeeds=JSON.parse($('tensorText').value).feeds||{}}catch{}
if(otherNames.some(n=>n!=='scept_mask'&&!suppliedFeeds[n]))throw Error('Missing additional inputs. Use JSON {feeds:{inputName:{shape,data},...}}.');
if(kind==='ablation'&&!session.inputNames.includes('scept_mask'))throw Error('Internal ablation requires an instrumented ONNX graph with a scept_mask input.');
const inputMeta=metaFor(name);const expected=inputMeta?.shape||[];if(inputMeta?.type!=='float32')throw Error('This adapter supports float32 model inputs only; convert/prepare compatible ONNX model.');
if(expected.length && (expected.length!==original.dims.length || expected.some((v,i)=>typeof v==='number'&&v>0&&v!==original.dims[i])))throw Error('Input shape does not match model metadata: '+sumshape(expected));
const baselineFeeds={[name]:new ort.Tensor('float32',original.data,original.dims)};
const editedFeeds={[name]:new ort.Tensor('float32',kind==='ablation'?original.data:edited.data,original.dims)};
for(const n of otherNames.filter(n=>n!=='scept_mask')){
 const v=suppliedFeeds[n];if(!v||!Array.isArray(v.shape)||!Array.isArray(v.data))throw Error('Additional feed '+n+' requires shape/data arrays');
 const k=v.shape.reduce((a,b)=>a*b,1);if(k!==v.data.length||k>10000000||!v.shape.every(x=>Number.isInteger(x)&&x>0)||!v.data.every(x=>Number.isFinite(x)))throw Error('Invalid extra input '+n);
 const meta=metaFor(n);if(meta?.type!=='float32')throw Error('Only float32 ONNX inputs supported: '+n);
 if(meta?.shape?.length && (v.shape.length!==meta.shape.length || meta.shape.some((x,i)=>typeof x==='number'&&x>0&&x!==v.shape[i])))throw Error('Shape mismatch for '+n);
 const tensor=new ort.Tensor('float32',Float32Array.from(v.data),v.shape);baselineFeeds[n]=tensor;editedFeeds[n]=tensor;
}
if(session.inputNames.includes('scept_mask')){
const mi=session.inputNames.indexOf('scept_mask'),dims=metaFor('scept_mask').shape||[];
if(!dims.length||!dims.every(v=>Number.isInteger(v)&&v>0))throw Error('scept_mask requires fixed positive dimensions.');
const len=dims.reduce((a,b)=>a*b,1);if(len>10000000)throw Error('Mask is too large');
const ones=new Float32Array(len).fill(1),mask=new Float32Array(ones);
if(kind==='ablation'){
const start=Number($('offset').value),count=Number($('count').value);
if(!Number.isInteger(start)||!Number.isInteger(count)||start<0||count<1||start+count>len)throw Error('Ablation indices exceed the internal activation tensor');
mask.fill(0,start,start+count);
}
baselineFeeds.scept_mask=new ort.Tensor('float32',ones,dims);
editedFeeds.scept_mask=new ort.Tensor('float32',mask,dims);
}
const baseline=await session.run(baselineFeeds);const intervention=await session.run(editedFeeds);const observations={};for(const n of session.outputNames){const a=toReadable(baseline[n]),b=toReadable(intervention[n]);if(a.values.length!==b.values.length)throw Error('Output shape changed for '+n);observations[n]={shape:a.shape,before:a.values,after:b.values};}const baseHash=await sha256(original.data.buffer),editHash=await sha256(edited.data.buffer);
if(modelClass==='ENGINEERING_FIXTURE_NOT_PRETRAINED' && $('experimentKind').value==='ablation'){
const actual=baseline.y?.data;if(!actual||actual.length!==2||Math.abs(actual[0]+.02)>1e-4||Math.abs(actual[1]-1.05)>1e-4)throw Error('Internal ablation reference baseline numerical verification FAILED');
}
const rec={schema:'scept.audit.v2',id:crypto.randomUUID(),date:new Date().toISOString(),experiment_kind:kind,model_classification:modelClass,model_name:modelFileName,evidence_level:kind==='ablation'?'L3_LOCAL_INTERNAL_INTERVENTION_UNREPLICATED':'L1_REAL_NUMERIC',model_sha256:sha,base_sha256:baseHash,edited_sha256:kind==='ablation'?baseHash:editHash,model_input:name,backend:$('runtimeSelect').value,perturbation:{method:$('perturbMethod').value,offset:Number($('offset').value),count:Number($('count').value),magnitude:Number($('strength').value)},inputs:{shape:original.dims,baseline:Array.from(original.data),edited:Array.from(kind==='ablation'?original.data:edited.data)},observations};last=rec;history.unshift({id:rec.id,date:rec.date,model:sha.slice(0,16),outputs:Object.keys(observations),evidence:rec.evidence_level});history=history.slice(0,40);try{localStorage.setItem('scept-audit-v2',JSON.stringify(history))}catch{}selectedData=observations;$('probeSelect').innerHTML=Object.keys(observations).map(n=>'<option value="'+escapeHtml(n)+'">'+escapeHtml(n)+'</option>').join('');$('probeSelect').disabled=false;$('exportBtn').disabled=false;$('exportCsv').disabled=false;$('resultType').textContent=(kind==='ablation'?'INTERNAL ABLATION · LOCAL L3':'REAL COMPUTATION · L1')+(modelClass==='UPLOADED_MODEL_UNVERIFIED'?'':' · ENGINEERING FIXTURE');$('inputReport').textContent='Baseline '+baseHash.slice(0,12)+'… · Edited '+editHash.slice(0,12)+'…';renderProbe();}
$('runBtn').onclick=async()=>{const b=$('runBtn');b.disabled=true;b.textContent='Computing locally…';try{await run()}catch(e){err(e)}finally{b.disabled=false;b.textContent='▶ Run baseline + intervention'}};
$('probeSelect').onchange=renderProbe;
function renderProbe(){if(!selectedData)return;window.SCEPT_GRAPH3D?.render(graph,selectedData);const t=selectedData[$('probeSelect').value];if(!t)return;const deltas=t.before.map((v,i)=>t.after[i]-v),abs=deltas.map(Math.abs),n=abs.length,mae=n?abs.reduce((a,b)=>a+b,0)/n:0,mx=n?abs.reduce((a,b)=>Math.max(a,b),0):0,changed=abs.filter(x=>x>1e-7).length;$('mae').textContent=fmt(mae);$('maxd').textContent=fmt(mx);$('changed').textContent=changed+'/'+n;const limit=Math.min(n,64),scale=Math.max(...abs.slice(0,limit),1e-12);$('diagram').innerHTML='<div class="barlist">'+Array.from({length:limit},(_,i)=>'<div class="barrow"><span>#'+i+'</span><div class="barbg"><i style="width:'+Math.max(1,100*abs[i]/scale)+'%"></i></div><strong>'+fmt(deltas[i])+'</strong></div>').join('')+'</div>';$('tensorRows').innerHTML=Array.from({length:Math.min(n,200)},(_,i)=>'<tr><td>['+i+']</td><td>'+fmt(t.before[i])+'</td><td>'+fmt(t.after[i])+'</td><td>'+fmt(deltas[i])+'</td></tr>').join('');}
$('exportCsv').onclick=()=>{if(!last)return;const output=last.observations[$('probeSelect').value];const lines=['index,baseline,intervention,delta'];for(let i=0;i<output.before.length;i++)lines.push([i,output.before[i],output.after[i],output.after[i]-output.before[i]].join(','));const a=document.createElement('a'),url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));a.href=url;a.download='scept-observed-tensor.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('exportBtn').onclick=()=>{if(!last)return;const blob=new Blob([JSON.stringify(last,null,2)],{type:'application/json'}),a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download='scept-audit-'+last.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
function renderHistory(){$('records').innerHTML=history.length?history.map(r=>'<div class="record"><div><b>'+escapeHtml(r.id)+'</b><div class="micro">'+escapeHtml(r.date)+' · '+escapeHtml(r.evidence)+'</div></div><small>model '+escapeHtml(r.model)+'…<br>'+r.outputs.length+' observed outputs</small></div>').join(''):'<p class="hint">No runs recorded in this browser.</p>';}
$('clearRecords').onclick=()=>{history=[];localStorage.removeItem('scept-audit-v2');renderHistory();};
window.addEventListener('unhandledrejection',e=>err(e.reason));

$('generateProbe').onclick=()=>{clear();try{if(!session)throw Error('Load a model first');const primary=$('inputTensor').value,meta=metaFor(primary),shape=meta.shape||[];if(!shape.length||!shape.every(v=>typeof v==='number'&&Number.isInteger(v)&&v>0))throw Error('Model input must have fixed positive dimensions for automatic generation');const len=shape.reduce((a,b)=>a*b,1);if(len>2500000)throw Error('Model input too large for auto-generated probe');let seed=314159265;const data=new Float32Array(len);for(let i=0;i<len;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;data[i]=(seed/4294967295-.5)*.2}externalMRI={dims:shape,data};$('tensorText').value='';$('tensorText').placeholder='Deterministic numeric test tensor loaded in memory';$('inputReport').textContent='Synthetic numerical probe · seed 314159265 · shape '+shape.join('×')+'. NOT a valid medical image or a preprocessed real photograph.';}catch(ex){err(ex)}};
window.SCEPT_IMPORT_VOLUME_TENSOR=(tensor)=>{if(!tensor||!(tensor.data instanceof Float32Array)||!Array.isArray(tensor.dims))throw Error('Invalid MRI tensor');externalMRI={dims:tensor.dims,data:tensor.data};$('tensorText').value='';$('tensorText').placeholder='MRI tensor loaded in local memory ('+tensor.dims.join('×')+'). Type JSON here to replace it.';$('inputReport').textContent='MRI-derived tensor prepared locally. Verify model-specific normalization, orientation and channel layout before inference.';};
$('tensorText').addEventListener('input',()=>{externalMRI=null});

function renderGraph(){
 if(!graph)return;
 $('graphMeta').textContent=graph.totalNodes+' operators · '+graph.edges.length+' directed intermediate tensor dependencies · '+graph.initializers.length+' initializers · '+graph.inputs.length+' graph inputs · '+graph.outputs.length+' declared outputs. Node view limited to the first 120 operators.';
 $('graphNodes').replaceChildren();const count=Math.min(graph.nodes.length,120);window.SCEPT_GRAPH3D?.render(graph,selectedData);
 for(let i=0;i<count;i++){
  const node=graph.nodes[i],b=document.createElement('button');b.type='button';b.className='graphnode';
  const label=document.createElement('strong');label.textContent=node.op||'Unknown';
  const sub=document.createElement('small');sub.textContent=node.name||('#'+i);
  b.append(label,sub);
  b.onclick=()=>{
   $('graphDetails').textContent='Node #'+i+' · '+(node.name||node.op)+' · inputs: '+node.inputs.join(', ')+' · outputs: '+node.outputs.join(', ')+' · incoming operators: '+graph.edges.filter(e=>e.to===i).length+' · outgoing operators: '+graph.edges.filter(e=>e.from===i).length+' (structural graph edges only)';
  };
  $('graphNodes').append(b);
 }
}
