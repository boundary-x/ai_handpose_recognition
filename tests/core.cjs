const test = require("node:test");
const assert = require("node:assert/strict");
const model = require("../hand-model.js");
const points = Array.from({length:21},(_,i)=>({x:0.5+(i%5)*0.04,y:0.8-i*0.025}));
const features = model.extractFeatures(points);
const project = () => model.serialize(["ID1","ID3","ID4"],5,[
  {label:"ID1",features},
  {label:"ID3",features:features.map(x=>-x)}
],true);
test("40D features are invariant to translation and uniform scaling",()=>{
  const moved = model.extractFeatures(points.map(p=>({x:p.x*0.6+0.1,y:p.y*0.6-0.2})));
  assert.equal(features.length,40);
  features.forEach((v,i)=>assert.ok(Math.abs(v-moved[i])<1e-12));
  assert.ok(model.validFeatures(features));
  assert.ok(features.reduce((s,v)=>s+v*v,0)>1);
});
test("invalid/degenerate landmark sets cannot be collected",()=>{
  assert.equal(model.extractFeatures([]),null);
  assert.equal(model.extractFeatures(Array(21).fill({x:0,y:0})),null);
  assert.equal(model.extractFeatures(points.map((p,i)=>i? p:{x:NaN,y:0})),null);
});
test("KNN confidence uses the available neighbors and deterministic ties",()=>{
  assert.deepEqual(model.classify([{label:"ID1",features}],features),
    {label:"ID1",confidence:1,neighbors:1});
  const data=[{label:"ID3",features},{label:"ID1",features}];
  assert.equal(model.classify(data,features).label,"ID1");
  assert.equal(model.classify([...data].reverse(),features).label,"ID1");
});
test("JSON round trip preserves prediction, empty classes and next ID",()=>{
  const before=project(),after=model.parse(JSON.stringify(before));
  assert.deepEqual(model.classify(before.samples,features),model.classify(after.samples,features));
  assert.deepEqual(after.classIds,["ID1","ID3","ID4"]);
  assert.equal(after.nextClassId,5);
  assert.equal(after.settings.isFlipped,true);
});
test("file format cannot accept image KNN models or incompatible engines",()=>{
  assert.throws(()=>model.validate({...project(),format:"boundary-x-knn"}));
  assert.throws(()=>model.validate({...project(),version:2}));
  assert.throws(()=>model.validate({...project(),engine:{...model.ENGINE,k:3}}));
  assert.throws(()=>model.validate({...project(),engine:{...model.ENGINE,preprocessing:"other"}}));
});
test("malformed IDs, missing class references and bad coordinates are rejected",()=>{
  for(const mutate of [
    p=>{p.classIds=["ID1","ID1"];},
    p=>{p.nextClassId=1;},
    p=>{p.samples[0].label="ID99";},
    p=>{p.samples[0].features[0]=NaN;},
    p=>{p.samples[0].features[0]=null;},
    p=>{p.samples[0].features=Array(40).fill(0);},
    p=>{p.samples[0].features.pop();},
    p=>{p.settings.isFlipped="true";}
  ]){const p=project();mutate(p);assert.throws(()=>model.validate(p));}
});
test("per-class sample limit is enforced on import",()=>{
  const p=project();p.samples=Array.from({length:501},()=>({label:"ID1",features:[...features]}));
  assert.throws(()=>model.validate(p));
});
test("serialization does not alias live training data",()=>{
  const data=[{label:"ID1",features:[...features]}];
  const p=model.serialize(["ID1"],2,data,true);
  p.samples[0].features[0]=123;
  assert.notEqual(data[0].features[0],123);
});
