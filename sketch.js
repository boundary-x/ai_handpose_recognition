/**
 * sketch.js
 * Boundary X: AI Handpose + Finger Sync (Dynamic Resolution & Error Fix)
 */

// Bluetooth UUIDs
const UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_TX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

// Variables
let video;
let handpose;
let predictions = [];
let knnClassifier;

// System State
let currentMode = 'KNN'; 
let isConnected = false;
let isRunning = false;
let isModelReady = false;
let isFlipped = true; 
let bluetoothDevice, rxCharacteristic;

// KNN Data
let nextClassId = 1;
let idToNameMap = {};

// Finger Sync Data
let fingerAngles = { T:0, I:0, M:0, R:0, P:0 }; 
let selectedProtocol = 'analog'; 

// DOM Elements
let badge, btDisplay, statusUI;

function setup() {
  // 1. 캔버스를 먼저 생성 (초기엔 작게)
  const canvas = createCanvas(320, 240);
  canvas.parent('p5-container');

  // 2. 비디오 캡처 설정
  video = createCapture(VIDEO, () => {
    console.log("Video captured");
  });
  
  // 3. 비디오 메타데이터 로드 완료 시 캔버스 크기를 비디오에 맞춤 (여백 제거 핵심)
  video.elt.onloadedmetadata = () => {
    console.log(`Camera Source Size: ${video.width}x${video.height}`);
    resizeCanvas(video.width, video.height);
    video.size(video.width, video.height);
  };
  
  video.hide();

  // ML5 Handpose Setup
  console.log("Loading Handpose...");
  handpose = ml5.handpose(video, modelReady);
  handpose.on("predict", results => {
    predictions = results;
  });

  // KNN Setup
  knnClassifier = ml5.KNNClassifier();

  setupUI();
}

function modelReady() {
  console.log("Handpose Ready!");
  isModelReady = true;
  select('#camera-result-badge').html('준비 완료! 시작하세요.');
  setTimeout(() => select('#camera-result-badge').hide(), 2000);
}

function setupUI() {
  badge = select('#camera-result-badge');
  btDisplay = select('#bluetooth-data-display');
  statusUI = select('#bluetoothStatus');

  select('#tab-knn').mousePressed(() => switchMode('KNN'));
  select('#tab-sync').mousePressed(() => switchMode('SYNC'));

  const radios = document.getElementsByName('protocol');
  radios.forEach(r => {
    r.addEventListener('change', (e) => selectedProtocol = e.target.value);
  });

  select('#btn-connect').mousePressed(connectBluetooth);
  select('#btn-disconnect').mousePressed(disconnectBluetooth);
  
  select('#btn-start').mousePressed(() => {
     if(currentMode === 'KNN' && knnClassifier.getNumLabels() <= 0) {
         alert("학습 데이터가 없습니다. 먼저 제스처를 학습시켜주세요.");
         return;
     }
     isRunning = true;
     // KNN 모드면 분류 시작
     if (currentMode === 'KNN') classify();
  });
  
  select('#btn-stop').mousePressed(() => {
    isRunning = false;
    sendBluetoothData("stop");
    btDisplay.html("중지됨");
    btDisplay.style('color', '#EA4335');
  });
  select('#btn-flip').mousePressed(() => isFlipped = !isFlipped);

  select('#add-class-btn').mousePressed(addClass);
  select('#reset-model-btn').mousePressed(resetKNN);
}

function switchMode(mode) {
  currentMode = mode;
  select('#tab-knn').removeClass('active');
  select('#tab-sync').removeClass('active');
  isRunning = false; 
  
  if(mode === 'KNN') {
    select('#tab-knn').addClass('active');
    select('#panel-knn').style('display', 'block');
    select('#panel-sync').style('display', 'none');
  } else {
    select('#tab-sync').addClass('active');
    select('#panel-knn').style('display', 'none');
    select('#panel-sync').style('display', 'block');
  }
}

// === MAIN LOOP ===
function draw() {
  background(0);
  
  // 카메라 준비 전이면 대기
  if (!video || !video.width) return;

  push();
  if (isFlipped) {
    translate(width, 0);
    scale(-1, 1);
  }
  
  // 비디오 그리기 (꽉 차게)
  image(video, 0, 0, width, height);
  
  // 스켈레톤 그리기
  drawKeypoints();
  pop();

  // Finger Sync Logic (KNN은 classify() 함수에서 별도 루프로 돔)
  if (isModelReady && predictions.length > 0 && isRunning) {
    if (currentMode === 'SYNC') {
      const hand = predictions[0];
      analyzeFingers(hand.landmarks);
      processSyncProtocol();
    }
  }
}

function drawKeypoints() {
  if(predictions.length === 0) return;
  const hand = predictions[0];
  
  fill(0, 255, 0);
  noStroke();
  for (let j = 0; j < hand.landmarks.length; j++) {
    let keypoint = hand.landmarks[j];
    ellipse(keypoint[0], keypoint[1], 10, 10);
  }
}

// === KNN Logic ===
function addClass() {
  const name = select('#class-input').value();
  if(!name) { alert("이름을 입력해주세요."); return; }
  if(predictions.length === 0) { alert("손이 인식되지 않았습니다."); return; }
  
  const id = nextClassId++;
  idToNameMap[id] = name;
  
  const features = predictions[0].landmarks.flat();
  knnClassifier.addExample(features, id);
  
  const list = select('#class-list');
  list.elt.innerHTML += `<div class="train-btn-row"><span class="id-badge">ID ${id}</span> <b>${name}</b> (학습 완료)</div>`;
  select('#class-input').value('');
}

// KNN 분류 시작 함수 (루프)
function classify() {
  if (!isRunning || currentMode !== 'KNN') return;
  if (knnClassifier.getNumLabels() <= 0) return;
  if (predictions.length > 0) {
      const features = predictions[0].landmarks.flat();
      knnClassifier.classify(features, gotKNNResults);
  } else {
      // 손이 없으면 잠시 후 재시도
      requestAnimationFrame(classify);
  }
}

// [핵심 수정] 에러 핸들링이 강화된 결과 처리 함수
function gotKNNResults(err, result) {
  if (err) {
    console.error("KNN Error:", err);
    requestAnimationFrame(classify);
    return;
  }

  // 결과값이 유효한지 2중 체크
  if (result && result.confidencesByLabel) {
      const label = result.label;
      const conf = result.confidencesByLabel[label];

      if (label && conf > 0.8) {
          const name = idToNameMap[label] || "Unknown";
          select('#result-label').html(`ID ${label}: ${name}`);
          select('#result-confidence').html(Math.floor(conf * 100) + '%');

          if (frameCount % 10 === 0) {
              const data = `ID${label}`;
              sendBluetoothData(data);
              btDisplay.html("전송됨: " + data);
              btDisplay.style('color', '#0f0');
          }
      }
  }

  // 계속해서 분류 실행
  if (isRunning) {
      requestAnimationFrame(classify);
  }
}

function resetKNN() {
  if(confirm("모든 학습 데이터를 삭제하시겠습니까?")) {
      knnClassifier.clearAllLabels();
      select('#class-list').html('');
      nextClassId = 1;
      idToNameMap = {};
      select('#result-label').html("데이터 없음");
      isRunning = false;
  }
}

// === Finger Sync Logic ===
function analyzeFingers(landmarks) {
  fingerAngles.T = calculateBend(landmarks[2], landmarks[3], landmarks[4]);
  fingerAngles.I = calculateBend(landmarks[5], landmarks[6], landmarks[7]);
  fingerAngles.M = calculateBend(landmarks[9], landmarks[10], landmarks[11]);
  fingerAngles.R = calculateBend(landmarks[13], landmarks[14], landmarks[15]);
  fingerAngles.P = calculateBend(landmarks[17], landmarks[18], landmarks[19]);
  
  updateBar('T', fingerAngles.T);
  updateBar('I', fingerAngles.I);
  updateBar('M', fingerAngles.M);
  updateBar('R', fingerAngles.R);
  updateBar('P', fingerAngles.P);
}

function calculateBend(a, b, c) {
  const AB = createVector(a[0]-b[0], a[1]-b[1]);
  const CB = createVector(c[0]-b[0], c[1]-b[1]);
  const angle = p5.Vector.angleBetween(AB, CB); 
  const deg = degrees(angle); 
  let extension = map(deg, 80, 170, 0, 100, true);
  extension = constrain(extension, 0, 100); // 0~100 범위 강제
  return Math.floor(extension);
}

function updateBar(key, val) {
  select(`#bar-${key}`).style('width', val + '%');
  select(`#val-${key}`).html(val + '%');
}

function processSyncProtocol() {
  let msg = "";
  if (selectedProtocol === 'analog') {
    msg = `T${fingerAngles.T}I${fingerAngles.I}M${fingerAngles.M}R${fingerAngles.R}P${fingerAngles.P}`;
  } else if (selectedProtocol === 'digital') {
    const t = fingerAngles.T > 50 ? 1 : 0;
    const i = fingerAngles.I > 50 ? 1 : 0;
    const m = fingerAngles.M > 50 ? 1 : 0;
    const r = fingerAngles.R > 50 ? 1 : 0;
    const p = fingerAngles.P > 50 ? 1 : 0;
    msg = `T${t}I${i}M${m}R${r}P${p}`;
  } else if (selectedProtocol === 'count') {
    let count = 0;
    if(fingerAngles.T > 50) count++;
    if(fingerAngles.I > 50) count++;
    if(fingerAngles.M > 50) count++;
    if(fingerAngles.R > 50) count++;
    if(fingerAngles.P > 50) count++;
    msg = `${count}`;
  }

  if(frameCount % 10 === 0) { 
     sendBluetoothData(msg);
     btDisplay.html(msg);
     btDisplay.style('color', '#0f0');
  }
}

// === Bluetooth Logic ===
async function connectBluetooth() {
  try {
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: "BBC micro:bit" }],
      optionalServices: [UART_SERVICE_UUID]
    });
    const server = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE_UUID);
    rxCharacteristic = await service.getCharacteristic(UART_RX_CHARACTERISTIC_UUID);
    isConnected = true;
    statusUI.html("상태: 연결됨!");
    statusUI.addClass('status-connected');
  } catch (e) {
    statusUI.html("연결 취소/실패");
    console.log(e);
  }
}

function disconnectBluetooth() {
  if(bluetoothDevice && bluetoothDevice.gatt.connected) {
    bluetoothDevice.gatt.disconnect();
  }
  isConnected = false;
  statusUI.html("상태: 연결 해제");
  statusUI.removeClass('status-connected');
}

async function sendBluetoothData(str) {
  if(!isConnected || !rxCharacteristic) return;
  try {
    const encoder = new TextEncoder();
    await rxCharacteristic.writeValue(encoder.encode(str + "\n"));
  } catch(e) {
    console.log(e);
  }
}
