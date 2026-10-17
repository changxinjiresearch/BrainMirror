/* Preliminary research on an openly available MRI-trained EfficientNetB3 ONNX.
   This optional gate tests browser feasibility; not a clinical performance check.
   Dataset train/test leakage reported by model author. */
const {chromium}=require('playwright'),crypto=require('node:crypto');
(async()=>{
const source='https://huggingface.co/Udit013/brain-tumor-efficientnetb3/resolve/main/efficientnetb3.onnx';
const resp=await fetch(source,{signal:AbortSignal.timeout(120000)});
if(!resp.ok)throw Error('MRI-trained model HTTP '+resp.status);
const buffer=Buffer.from(await resp.arrayBuffer()),sha=crypto.createHash('sha256').update(buffer).digest('hex');
console.log('MRI-trained candidate ONNX bytes',buffer.byteLength,'source SHA256',sha);
if(buffer.byteLength>170000000)throw Error('MRI model too large for constrained browser smoke test');
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
const page=await browser.newPage();await page.route('http://127.0.0.1:8765/candidate-mri.onnx',route=>route.fulfill({status:200,contentType:'application/octet-stream',body:buffer}));await page.goto('http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
const meta=await page.evaluate(async()=>{
 const buf=new Uint8Array(await(await fetch('/candidate-mri.onnx')).arrayBuffer()),graph=SCEPT_INSPECT_ONNX(buf.buffer);
 ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
 ort.env.wasm.numThreads=1;
 const session=await ort.InferenceSession.create(buf,{executionProviders:['wasm']});
 const name=session.inputNames[0],m=graph.inputMetadata[name],shape=m?.shape;
 if(!shape||shape.some(x=>typeof x!=='number' || x<=0))throw Error('MRI model requires a model-specific dynamic input adapter: '+JSON.stringify(m));
 if(shape.reduce((a,b)=>a*b,1)>2000000)throw Error('MRI model input too large for Chrome smoke test');
 const arr=new Float32Array(shape.reduce((a,b)=>a*b,1));for(let i=0;i<arr.length;i++)arr[i]=i%256;
 const out=await session.run({[name]:new ort.Tensor('float32',arr,shape)});
 const outputs=Object.fromEntries(Object.entries(out).map(([n,t])=>[n,{shape:t.dims,first:Array.from(t.data.slice(0,10))}]));
 await session.release();
 return{input:name,metadata:m,outputs,numOperators:graph.totalNodes};
 });
 console.log('MRI-trained model browser compatibility PASS',JSON.stringify(meta).slice(0,1800));
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
