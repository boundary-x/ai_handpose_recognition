/**
 * sketch.js
 * Boundary X: AI Gesture Learning [Final Fixed Version]
 * Fix: UI updates even when Bluetooth is disconnected
 * Feature: Scale-Invariant Feature Extraction
 */

// Bluetooth UUIDs
const UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_TX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

let bluetoothDevice, rxCharacteristic, isConnected = false;
let bluetoothStatus = "연결 대기 중";
let isSendingData = false;
let lastSendTime = 0;

let video;
let handpose;
let knnClassifier;
let predictions = [];
let isModelReady = false;

// UI Elements
let classInput, addDataBtn, resetBtn;
let resultLabel, resultConf, btDataDisplay;
let trainingList, statusBadge;

// Data & Settings
let classes = {}; 
let isTraining = false;
let isFlipped = true; 
let isTracking = false;

function setup() {
  let canvas = createCanvas(320, 240);
  canvas.parent('p5-container');

  // 1. 비디오 설정
  video = createCapture(VIDEO);
  video.size(320, 240);
  video.hide();

  // 2. ML 설정
  handpose = ml5.handpose(video, modelReady);
  handpose.on("predict", results => {
    predictions = results;
  });
  knnClassifier = ml5.KNNClassifier();

  // 3. UI 설정
  setupUI();
}

function modelReady() {
  console.log("Handpose Ready");
  isModelReady = true;
  if(statusBadge) statusBadge.html("준비 완료! 제스처를 학습시키세요.");
}

function setupUI() {
  statusBadge = select('#status-badge');
  classInput = select('#class-input');
  addDataBtn = select('#add-data-btn');
  resetBtn = select('#reset-btn');
  trainingList = select('#training-list');
  resultLabel = select('#result-label');
  resultConf = select('#result-conf');
  btDataDisplay = select('#bluetooth-data-display');

  addDataBtn.mousePressed(() => isTraining = true);
  addDataBtn.mouseReleased(() => isTraining = false);
  resetBtn.mousePressed(clearAllModel);

  // --- [1. 기기 연결] ---
  let connectBtn = createButton("기기 연결");
  connectBtn.parent('bluetooth-control-buttons');
  connectBtn.addClass('start-button');
  connectBtn.mousePressed(connectBluetooth);

  let disconnectBtn = createButton("연결 해제");
  disconnectBtn.parent('bluetooth-control-buttons');
  disconnectBtn.addClass('stop-button');
  disconnectBtn.mousePressed(disconnectBluetooth);

  // --- [2. AI 인식 제어] ---
  let startTrackBtn = createButton("인식 시작");
  startTrackBtn.parent('recognition-control-buttons');
  startTrackBtn.addClass('start-button');
  startTrackBtn.mousePressed(() => { 
      isTracking = true; 
      btDataDisplay.html("데이터 분석 중...");
      btDataDisplay.style('color', '#0f0'); // 시작하면 녹색
  });

  let stopTrackBtn = createButton("인식 중지");
  stopTrackBtn.parent('recognition-control-buttons');
  stopTrackBtn.addClass('stop-button');
  stopTrackBtn.mousePressed(() => { 
      isTracking = false; 
      sendBluetoothData("stop");
      btDataDisplay.html("전송 중지됨");
      btDataDisplay.style('color', '#EA4335'); // 중지하면 빨간색
  });

  updateBluetoothStatusUI();
}

function draw() {
  background(0);

  // 1. 그리기 (거울 모드)
  push();
  if (isFlipped) {
      translate(width, 0);
      scale(-1, 1);
  }
  
  if (video.elt.readyState >= 2) {
      image(video, 0, 0, width, height);
      drawKeypoints(); 
  }
  pop();

  // 2. 로직 처리
  if (predictions.length > 0) {
    let hand = predictions[0];
    
    // [중요] 거리/크기 무시 정규화 적용된 함수 사용
    let features = extractRelativeFeatures(hand);

    if (isTraining) {
      let label = classInput.value().trim();
      if (label) addExample(features, label);
    } 
    else if (knnClassifier.getNumLabels() > 0) {
      classify(features);
    }
  }
}

/**
 * [업그레이드된 알고리즘]
 * 1. 손목 기준 상대 좌표 변환
 * 2. 스케일 정규화 (손 크기/거리 무시) -> 인식률 대폭 향상
 */
function extractRelativeFeatures(hand) {
  let features = [];
  let landmarks = hand.landmarks; 
  let wrist = landmarks[0]; 

  // 1. 가장 먼 거리 찾기 (손의 크기 측정)
  let maxDist = 0;
  for (let i = 1; i < landmarks.length; i++) {
      let dx = landmarks[i][0] - wrist[0];
      let dy = landmarks[i][1] - wrist[1];
      let dist = Math.sqrt(dx*dx + dy*dy);
      if (dist > maxDist) maxDist = dist;
  }
  
  // 0 나누기 방지
  if (maxDist < 1) maxDist = 1;

  // 2. 정규화된 상대 좌표 계산
  for (let i = 1; i < landmarks.length; i++) {
    let x = landmarks[i][0];
    let y = landmarks[i][1];
    
    // (좌표 - 손목) / 손크기
    let relativeX = (x - wrist[0]) / maxDist;
    let relativeY = (y - wrist[1]) / maxDist;
    
    features.push(relativeX);
    features.push(relativeY);
  }
  
  return features; 
}

function addExample(features, label) {
  knnClassifier.addExample(features, label);
  if (!classes[label]) classes[label] = 0;
  classes[label]++;
  updateListUI();
}

function classify(features) {
  knnClassifier.classify(features, (err, result) => {
    if (err) return;
    
    if (result.confidencesByLabel) {
      const label = result.label;
      const conf = result.confidencesByLabel[label];
      
      // 결과 라벨 업데이트
      resultLabel.html(label);
      resultConf.html(`정확도: ${(conf * 100).toFixed(0)}%`);
      
      if (conf > 0.85) resultLabel.style('color', '#00E676');
      else resultLabel.style('color', '#FFEB3B');

      // [핵심 수정] UI 업데이트와 전송 로직 분리
      if (isTracking) {
          // 1. 화면에는 무조건 표시 (연결 여부 상관없이)
          let displayMsg = `전송 데이터: ${label}`;
          if (!isConnected) displayMsg += " (연결 안됨)";
          
          btDataDisplay.html(displayMsg);
          btDataDisplay.style('color', '#00E676'); // Tracking 중이면 녹색

          // 2. 연결되어 있을 때만 실제 전송
          if (isConnected && millis() - lastSendTime > 100) {
              sendBluetoothData(label);
              lastSendTime = millis();
          }
      }
    }
  });
}

function drawKeypoints() {
  let videoWidth = video.elt.videoWidth || 640;
  let videoHeight = video.elt.videoHeight || 480;

  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      
      let x = map(keypoint[0], 0, videoWidth, 0, width);
      let y = map(keypoint[1], 0, videoHeight, 0, height);
      
      if (j === 0) fill(255, 0, 0); 
      else fill(0, 255, 0);         
      noStroke();
      ellipse(x, y, 8, 8);
    }
  }
}

function updateListUI() {
  trainingList.html("");
  if (Object.keys(classes).length === 0) {
      trainingList.html('<div class="empty-msg">아직 학습된 데이터가 없습니다.</div>');
      return;
  }

  for (let label in classes) {
    let li = createDiv().addClass('list-item');
    let left = createDiv().addClass('list-item-left');
    createSpan(label).parent(left);
    createSpan(`${classes[label]} data`).addClass('badge-count').parent(left);
    left.parent(li);

    let delBtn = createButton('X').addClass('delete-btn');
    delBtn.mousePressed(() => deleteClass(label));
    delBtn.parent(li);

    li.parent(trainingList);
  }
}

function deleteClass(label) {
    if(knnClassifier) {
        knnClassifier.clearLabel(label);
        delete classes[label];
        updateListUI();
        resultLabel.html("대기 중");
        resultConf.html("데이터 삭제됨");
    }
}

function clearAllModel() {
  knnClassifier.clearAllLabels();
  classes = {};
  updateListUI();
  resultLabel.html("대기 중");
  resultLabel.style('color', '#00E676');
  resultConf.html("데이터 없음");
}

/* --- Bluetooth Logic --- */
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
    bluetoothStatus = "연결됨: " + bluetoothDevice.name;
    updateBluetoothStatusUI(true);
  } catch (error) {
    console.error(error);
    bluetoothStatus = "연결 실패";
    updateBluetoothStatusUI(false, true);
  }
}

function disconnectBluetooth() {
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    bluetoothDevice.gatt.disconnect();
  }
  isConnected = false;
  bluetoothStatus = "연결 해제됨";
  rxCharacteristic = null;
  bluetoothDevice = null;
  updateBluetoothStatusUI(false);
}

function updateBluetoothStatusUI(connected = false, error = false) {
  const statusElement = select('#bluetoothStatus');
  if(statusElement) {
      statusElement.html(`상태: ${bluetoothStatus}`);
      statusElement.removeClass('status-connected');
      statusElement.removeClass('status-error');
      if (connected) statusElement.addClass('status-connected');
      else if (error) statusElement.addClass('status-error');
  }
}

async function sendBluetoothData(data) {
  if (!rxCharacteristic || !isConnected) return;
  if (isSendingData) return;
  try {
    isSendingData = true;
    const encoder = new TextEncoder();
    await rxCharacteristic.writeValue(encoder.encode(data + "\n"));
  } catch (error) {
    console.error(error);
  } finally {
    isSendingData = false;
  }
}
