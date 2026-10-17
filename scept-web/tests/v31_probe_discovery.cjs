const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage();
  await page.goto('http://127.0.0.1:8765/v3.html',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.SCEPT_V3?.getSnapshot().model||document.getElementById('readyLabel').dataset.state==='error',null,{timeout:120000});
  const data=await page.evaluate(async()=>{
   const url='https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/resolve/ebffd7b9c0c53b51bada44116abd8f01aed41be6/onnx/model.onnx';
   const buf=await(await fetch(url)).arrayBuffer(),g=SCEPT_INSPECT_ONNX(buf);
   const patterns=['embeddings','attention/attention/query','attention/attention/key','attention/attention/value','attention/attention/Softmax','layer.6/intermediate','layer.6/output','layer.6/layernorm'];
   return Object.fromEntries(patterns.map(p=>[p,g.nodes.filter(n=>n.name.includes(p)).slice(0,12).map(n=>({name:n.name,op:n.op,out:n.outputs}))]));
  });
  for(const [k,v]of Object.entries(data))console.log('PROBE DISCOVERY',k,JSON.stringify(v));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
