const assert=require("node:assert/strict");
const {chromium}=require("playwright");
const http=require("node:http"),fs=require("node:fs"),path=require("node:path");
const root=path.resolve(__dirname,"..");
const artifacts=process.env.TEST_ARTIFACTS || path.join(root,"test-results");
fs.mkdirSync(artifacts,{recursive:true});
const server=http.createServer((req,res)=>{
 const target=path.resolve(root,"."+decodeURIComponent(req.url.split("?")[0]==="/"?"/index.html":req.url.split("?")[0]));
 if(!target.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
 fs.readFile(target,(error,data)=>{
  if(error){res.writeHead(404);res.end();return;}
  res.setHeader("Content-Type",target.endsWith(".js")?"application/javascript":target.endsWith(".css")?"text/css":"text/html");res.end(data);
 });
});
const checks=[];function pass(label){checks.push(label);console.log("PASS",label);}
(async()=>{
 await new Promise(r=>server.listen(0,"127.0.0.1",r));
 const browser=await chromium.launch({headless:true,...(process.env.BROWSER_CHANNEL?{channel:process.env.BROWSER_CHANNEL}:{}),
  args:["--use-fake-ui-for-media-stream","--use-fake-device-for-media-stream"]});
 try{
 const page=await browser.newPage({viewport:{width:390,height:844}});
 const errors=[];page.on("pageerror",e=>errors.push(e.message));
 page.on("dialog",d=>d.accept());
 await page.goto("http://127.0.0.1:"+server.address().port,{waitUntil:"networkidle",timeout:60000});
 await page.waitForFunction(()=>isModelReady && video.elt.readyState>=2,null,{timeout:60000});
 assert.equal(await page.evaluate(()=>HandModel.ENGINE.version),"0.10.8");
 assert.equal(await page.evaluate(()=>!!handLandmarker),true);
 pass("Real MediaPipe 0.10.8 model/GPU initialization and camera stream");
 await page.locator("#add-class-btn").click();
 const train=page.locator('.train-btn[data-id="ID1"]');
 assert.equal(await train.isDisabled(),true);
 assert.equal(await page.evaluate(()=>collectSample("ID1")),false);
 pass("No hand: training disabled and no empty sample collected");

 // Only replace the detection output after real model initialization.
 // Fixtures make the application state deterministic without claiming real gesture accuracy.
 await page.evaluate(()=>{
  window.fixture=Array.from({length:21},(_,i)=>({x:0.5+(i%5)*0.04,y:0.8-i*0.025,z:0}));
  handLandmarker.detectForVideo=()=>({landmarks:window.fixture?[window.fixture]:[]});
 });
 await page.waitForFunction(()=>handAvailable);
 await train.click();
 assert.equal(await page.evaluate(()=>trainingData.length),1);
 pass("Mouse tap adds one sample without a duplicate click");
 await page.waitForTimeout(100);
 await train.focus();await page.keyboard.press("Enter");
 assert.equal(await page.evaluate(()=>trainingData.length),2);
 pass("Keyboard activation adds one sample");
 await train.scrollIntoViewIfNeeded();let box=await train.boundingBox();
 await page.mouse.move(box.x+20,box.y+20);await page.mouse.down();await page.waitForTimeout(850);await page.mouse.up();
 const held=await page.evaluate(()=>trainingData.length);
 assert.ok(held>=4);
 await page.waitForTimeout(350);
 assert.equal(await page.evaluate(()=>trainingData.length),held);
 pass("Hold collects repeatedly and release stops without extra sample");
 const cdp=await page.context().newCDPSession(page);
 const point={x:Math.round(box.x+20),y:Math.round(box.y+20)};
 await cdp.send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[point]});
 await cdp.send("Input.dispatchTouchEvent",{type:"touchEnd",touchPoints:[]});
 await page.waitForTimeout(150);
 assert.equal(await page.evaluate(()=>trainingData.length),held+1);
 await cdp.send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[point]});
 await page.waitForTimeout(750);
 await cdp.send("Input.dispatchTouchEvent",{type:"touchEnd",touchPoints:[]});
 const touchCount=await page.evaluate(()=>trainingData.length);
 assert.ok(touchCount>=held+3);
 await page.waitForTimeout(300);
 assert.equal(await page.evaluate(()=>trainingData.length),touchCount);
 pass("Touch tap and hold preserve one-shot/repeat behavior");
 await page.mouse.move(box.x+20,box.y+20);await page.mouse.down();
 await train.dispatchEvent("pointercancel",{pointerId:1});
 await page.mouse.up();await page.waitForTimeout(400);
 assert.equal(await page.evaluate(()=>trainingData.length),touchCount);
 pass("Pointer cancellation ends acquisition");

 const dedup=await page.evaluate(()=>{
  lastSampleFrame=-1;
  const a=collectSample("ID1"),n=trainingData.length,b=collectSample("ID1");
  return {a,b,n,after:trainingData.length};
 });
 assert.equal(dedup.a,true);assert.equal(dedup.b,null);assert.equal(dedup.n,dedup.after);
 pass("A detection frame can only be collected once");
 await page.evaluate(()=>{video.elt.pause();});
 await page.waitForTimeout(650);
 assert.equal(await page.evaluate(()=>handAvailable),false);
 assert.equal(await page.evaluate(()=>collectSample("ID1")),false);
 await page.evaluate(()=>video.elt.play());
 await page.waitForFunction(()=>handAvailable);
 pass("Frozen camera invalidates cached landmarks and blocks training");

 // Simulate the BLE boundary; no physical device receives data.
 await page.evaluate(()=>{
  window.writes=[];
  isConnected=true;bluetoothDevice={gatt:{connected:true,disconnect(){}}};
  rxCharacteristic={writeValue:async bytes=>{window.writes.push(new TextDecoder().decode(bytes));}};
  startTracking();
 });
 await page.waitForFunction(()=>window.writes.some(x=>x==="ID1\n"));
 await page.evaluate(()=>{window.fixture=null;});
 await page.waitForTimeout(700);await page.evaluate(()=>sendQueue);
 assert.equal(await page.locator("#result-label").textContent(),"손 감지 안 됨");
 assert.match(await page.locator("#result-conf").textContent(), /stop 전송 완료/);
 const stops=await page.evaluate(()=>window.writes.filter(x=>x==="stop\n").length);
 assert.equal(stops,1);
 await page.waitForTimeout(400);
 assert.equal(await page.evaluate(()=>window.writes.filter(x=>x==="stop\n").length),1);
 pass("Hand loss clears result and emits one stop after grace period");
 await page.evaluate(()=>{
  window.fixture=Array.from({length:21},(_,i)=>({x:0.5+(i%5)*0.04,y:0.8-i*0.025,z:0}));
 });
 await page.waitForFunction(()=>document.getElementById("result-label").textContent==="ID1");
 await page.evaluate(()=>{stopTracking();});
 await page.evaluate(()=>sendQueue);
 await page.waitForTimeout(200);
 assert.equal(await page.locator("#result-label").textContent(),"중지됨");
 pass("Hand return resumes tracking; explicit stop cannot be overwritten");
 await page.evaluate(()=>{isConnected=false;rxCharacteristic=null;bluetoothDevice=null;});
 await page.locator("#add-class-btn").click();
 await page.locator("#add-class-btn").click();
 await page.waitForTimeout(100);
 await page.locator('.train-btn[data-id="ID3"]').click();
 await page.locator('.list-item[data-id="ID2"] .delete-btn').click();
 await page.locator("#add-class-btn").click();
 assert.deepEqual(await page.evaluate(()=>classIds),["ID1","ID3","ID4"]);
 pass("Automatic IDs, deletion gap and an empty ID are preserved");

 const before=await page.evaluate(()=>({
  prediction:HandModel.classify(trainingData,lastFeatures),query:[...lastFeatures],
  counts:HandModel.counts(trainingData),ids:[...classIds],next:nextClassId
 }));
 const downloadPromise=page.waitForEvent("download");
 await page.locator("#download-model-btn").click();
 const file=await downloadPromise;
 const filepath=path.join(artifacts,"hand-roundtrip.json");await file.saveAs(filepath);
 const saved=JSON.parse(fs.readFileSync(filepath,"utf8"));
 assert.equal(saved.format,"boundary-x-handpose-knn");assert.equal(saved.samples[0].features.length,40);
 await page.locator("#reset-model-btn").click();
 assert.equal(await page.evaluate(()=>trainingData.length),0);
 await page.locator("#model-file-input").setInputFiles(filepath);
 await page.waitForFunction(()=>!isBusy);
 const after=await page.evaluate(query=>({
  prediction:HandModel.classify(trainingData,query),counts:HandModel.counts(trainingData),ids:[...classIds],next:nextClassId
 }),before.query);
 assert.deepEqual(after.prediction,before.prediction);assert.deepEqual(after.counts,before.counts);
 assert.deepEqual(after.ids,before.ids);assert.equal(after.next,before.next);
 await page.locator("#add-class-btn").click();
 assert.equal(await page.evaluate(()=>classIds.at(-1)),"ID5");
 pass("Download/reset/import round trip preserves prediction, counts, empty IDs and next ID");

 const snapshot=await page.evaluate(()=>JSON.stringify({trainingData,classIds,nextClassId,isFlipped}));
 for(const [name,text] of [
  ["invalid","{"],
  ["image-model",JSON.stringify({...saved,format:"boundary-x-knn"})],
  ["wrong-version",JSON.stringify({...saved,version:2})],
  ["wrong-dimensions",JSON.stringify({...saved,samples:[{label:"ID1",features:[0]}]})],
  ["duplicate-id",JSON.stringify({...saved,classIds:["ID1","ID1"]})]
 ]){
  await page.locator("#model-file-input").setInputFiles({name:name+".json",mimeType:"application/json",buffer:Buffer.from(text)});
  await page.waitForFunction(()=>!isBusy);
  assert.match(await page.locator("#file-status").textContent(),/실패/);
  assert.equal(await page.evaluate(()=>JSON.stringify({trainingData,classIds,nextClassId,isFlipped})),snapshot);
 }
 pass("Invalid, image-model and incompatible files leave existing data intact");
 page.removeAllListeners("dialog");page.once("dialog",d=>d.dismiss());
 await page.locator("#model-file-input").setInputFiles(filepath);await page.waitForFunction(()=>!isBusy);
 assert.equal(await page.evaluate(()=>JSON.stringify({trainingData,classIds,nextClassId,isFlipped})),snapshot);
 assert.match(await page.locator("#file-status").textContent(),/취소/);
 page.on("dialog",d=>d.accept());
 pass("Import cancellation preserves current project");

 await page.evaluate(()=>{
  Object.defineProperty(navigator,"canShare",{configurable:true,value:()=>false});
 });
 const fallback=page.waitForEvent("download");await page.locator("#share-model-btn").click();await fallback;
 assert.match(await page.locator("#file-status").textContent(),/지원하지 않아/);
 await page.evaluate(()=>{
  Object.defineProperty(navigator,"canShare",{configurable:true,value:()=>true});
  Object.defineProperty(navigator,"share",{configurable:true,value:async payload=>{
   window.shared={type:payload.files[0].type,active:navigator.userActivation.isActive};
  }});
 });
 await page.locator("#share-model-btn").click();
 assert.deepEqual(await page.evaluate(()=>window.shared),{type:"application/json",active:true});
 await page.evaluate(()=>Object.defineProperty(navigator,"share",{configurable:true,value:async()=>{throw new DOMException("cancel","AbortError");}}));
 await page.locator("#share-model-btn").click();
 assert.match(await page.locator("#file-status").textContent(),/취소/);
 assert.equal(await page.locator("#download-model-btn").isEnabled(),true);
 pass("File sharing supports fallback, user activation and cancellation");

 const failure=await page.evaluate(async()=>{
  isConnected=true;rxCharacteristic={writeValue:async()=>{throw new Error("test failure");}};
  lastSentLabel="";lastSendTime=0;isTracking=true;
  await queueSend("ID1",trackingEpoch);
  const result={label:lastSentLabel,time:lastSendTime,message:byId("bluetooth-data-display").textContent};
  isTracking=false;isConnected=false;rxCharacteristic=null;
  return result;
 });
 assert.equal(failure.label,"");assert.equal(failure.time,0);assert.match(failure.message,/실패/);
 pass("Failed BLE writes do not update last successful transmission");
 const ordered=await page.evaluate(async()=>{
  const log=[];isConnected=true;isTracking=true;
  rxCharacteristic={writeValue:async bytes=>{await new Promise(r=>setTimeout(r,20));log.push(new TextDecoder().decode(bytes));}};
  const first=queueSend("ID1",trackingEpoch);
  await new Promise(r=>setTimeout(r,1));stopTracking();stopTracking();
  await first;await sendQueue;isConnected=false;rxCharacteristic=null;return log;
 });
 assert.deepEqual(ordered,["ID1\n","stop\n"]);
 pass("Stop is serialized after an in-flight prediction write");

 for(const [width,height] of [[320,740],[360,800],[390,844],[430,932],[768,1024],[844,390],[1280,900]]){
  await page.setViewportSize({width,height});await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(150);
  const layout=await page.evaluate(()=>{
   const header=document.querySelector("header").getBoundingClientRect();
   const back=document.querySelector(".back-button"),camera=document.querySelector(".canvas-container").getBoundingClientRect();
   const range=document.createRange();range.selectNodeContents(back);
   const label=document.createRange();label.selectNodeContents(document.querySelector(".train-text"));
   return {overflow:document.documentElement.scrollWidth>innerWidth,lines:range.getClientRects().length,
    trainLines:label.getClientRects().length,overlap:camera.top<header.bottom-1,
    ratio:camera.width/camera.height};
  });
  assert.equal(layout.overflow,false,width+" overflow");assert.equal(layout.lines,1,width+" wraps");
  assert.equal(layout.trainLines,1,width+" training wraps");assert.equal(layout.overlap,false,width+" overlap");
  assert.ok(Math.abs(layout.ratio-4/3)<0.02,width+" camera ratio");
  if([390,844,1280].includes(width))await page.screenshot({path:path.join(artifacts,"hand-layout-"+width+".png"),fullPage:true});
 }
 pass("Portrait/landscape layouts preserve 4:3 video and avoid wraps/overflow");
 assert.deepEqual(errors,[]);pass("No uncaught browser errors");
 fs.writeFileSync(path.join(artifacts,"results.json"),JSON.stringify({checks,errors},null,2));
 console.log("Completed",checks.length,"browser checks");
 }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});

