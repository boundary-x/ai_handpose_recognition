/**
 * sketch.js
 * Boundary X: AI Hand Recognition [Gesture & Finger Sync]
 * Tech: ml5.handpose (21 Keypoints) + ml5.KNNClassifier
 */

// Bluetooth UUIDs
const UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_TX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

// Variables
let bluetoothDevice, rxCharacteristic, isConnected = false;
let isSendingData = false;
let lastSendTime = 0;

let video, handpose, knnClassifier;
let predictions = [];
let isModelReady = false;
let isTracking = false; // 데이터 전송 활성화 여부

// Current Mode: 'gesture' or 'finger'
let currentMode = 'gesture';
let fingerFormat = 'analog'; // 'analog' or 'digital'

// UI Elements
let statusBadge, btDisplayGesture, btDisplayFinger;
let gestureResultLabel, gestureConfidence;
let classInput, addClassBtn, gestureListContainer;
let canvas;

// ID Map for Gesture
let nextClassId = 1;
let idToNameMap = {};

function setup() {
  canvas = createCanvas(320, 240);
  canvas.parent('p5-container');
  
  // Handpose 로드
  handpose = ml5.handpose(video, modelReady);
  knnClassifier = ml5.KNNClassifier();

  // Handpose 이벤트
  handpose.on("predict", results => {
    predictions = results;
  });

  setupCamera();
  createUI();
}

function modelReady() {
  console.log("Handpose Model Ready!");
  isModelReady = true;
  select('#status-badge').html("준비 완료");
}

function setupCamera() {
  video = createCapture(VIDEO);
  video.size(320, 240);
  video.hide();
}

function createUI() {
  // 1. 모드 스위치 이벤트
  const modeRadios = document.getElementsByName('mode');
  modeRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
          currentMode = e.target.value;
          updateModeUI();
      });
  });

  // 2. 핑거 데이터 포맷 이벤트
  const formatRadios = document.getElementsByName('finger-format');
  formatRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
          fingerFormat = e.target.value;
      });
  });

  // 3. UI 요소 선택
  statusBadge = select('#status-badge');
  btDisplayGesture = select('#bt-display-gesture');
  btDisplayFinger = select('#bt-display-finger');
  gestureResultLabel = select('#gesture-result-label');
  gestureConfidence = select('#gesture-confidence');
  
  classInput = select('#class-input');
  addClassBtn = select('#add-example-btn');
  gestureListContainer = select('#gesture-list');

  // 4. 버튼 이벤트
  addClassBtn.mousePressed(addGestureExample);
  select('#reset-model-btn').mousePressed(resetModel);

  // 공통 버튼 생성
  createCommonButtons();
}

function updateModeUI() {
  if (currentMode === 'gesture') {
      select('#panel-gesture').style('display', 'block');
      select('#panel-finger').style('display', 'none');
      statusBadge.style('background-color', 'rgba(123, 31, 162, 0.8)'); // Purple
  } else {
      select('#panel-gesture').style('display', 'none');
      select('#panel-finger').style('display', 'block');
      statusBadge.style('background-color', 'rgba(245, 124, 0, 0.8)'); // Orange
  }
}

function createCommonButtons() {
  // 카메라 제어 (반전)
  let flipBtn = createButton("좌우 반전");
  flipBtn.parent('camera-control-buttons');
  flipBtn.addClass('start-button');
  flipBtn.mousePressed(() => {
     // p5.js capture doesn't support easy flip without drawing, handled in draw()
     // Here we just toggle a flag if needed, but drawing handles it usually.
     // For simplicity in this structure, we rely on draw() logic.
  });

  // 블루투스
  let connectBtn = createButton("기기 연결");
  connectBtn.parent('bluetooth-control-buttons');
  connectBtn.addClass('start-button');
  connectBtn.mousePressed(connectBluetooth);

  let disconnectBtn = createButton("연결 해제");
  disconnectBtn.parent('bluetooth-control-buttons');
  disconnectBtn.addClass('stop-button');
  disconnectBtn.mousePressed(disconnectBluetooth);

  // 인식 제어
  let startBtn = createButton("데이터 전송 시작");
  startBtn.parent('recognition-control-buttons');
  startBtn.addClass('start-button');
  startBtn.mousePressed(() => { isTracking = true; });

  let stopBtn = createButton("전송 중지");
  stopBtn.parent('recognition-control-buttons');
  stopBtn.addClass('stop-button');
  stopBtn.mousePressed(() => { isTracking = false; sendBluetoothData("stop"); });
}

// === MAIN LOOP ===

function draw() {
  background(0);
  
  // 비디오 그리기
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
  
  // 스켈레톤 그리기
  drawKeypoints();
  pop();

  if (predictions.length > 0) {
      let hand = predictions[0];
      
      // 모드별 로직 실행
      if (currentMode === 'gesture') {
          classifyGesture(hand);
      } else {
          processFingerSync(hand);
      }
      
      statusBadge.html(isTracking ? "전송 중..." : "인식 중 (대기)");
  } else {
      statusBadge.html("손을 보여주세요");
  }
}

// === LOGIC: KEYPOINTS DRAWING ===
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    
    // 관절 점
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
    // 뼈대 (간략)
    // ... (선 그리기 로직은 생략하거나 추가 가능)
  }
}

// === LOGIC: GESTURE LEARNING (KNN) ===

function addGestureExample() {
    if (predictions.length === 0) return;
    
    const className = classInput.value().trim();
    if (!className) return;

    // 현재 손의 좌표 특징 추출 (손목 기준 상대좌표로 변환하여 학습)
    const features = extractFeatures(predictions[0]);
    
    // ID 매핑 (없으면 생성)
    let labelId = Object.keys(idToNameMap).find(key => idToNameMap[key] === className);
    if (!labelId) {
        labelId = String(nextClassId++);
        idToNameMap[labelId] = className;
        addUIList(labelId, className);
    }

    knnClassifier.addExample(features, labelId);
    updateGestureCount(labelId);
}

function classifyGesture(hand) {
    if (knnClassifier.getNumLabels() <= 0) return;

    const features = extractFeatures(hand);
    knnClassifier.classify(features, (err, result) => {
        if (err) return;
        
        const labelId = result.label;
        const conf = result.confidencesByLabel[labelId];
        const name = idToNameMap[labelId] || "Unknown";

        gestureResultLabel.html(`${name} (ID:${labelId})`);
        gestureConfidence.html(`${(conf * 100).toFixed(0)}%`);

        if (isTracking && isConnected) {
             let data = `G${labelId}`;
             sendBluetoothData(data);
             btDisplayGesture.html(`전송됨: ${data}`);
        }
    });
}

// 좌표 정규화 (손목 기준)
function extractFeatures(hand) {
    let features = [];
    let wrist = hand.landmarks[0]; // 손목 좌표
    for (let i = 1; i < hand.landmarks.length; i++) {
        features.push(hand.landmarks[i][0] - wrist[0]); // x
        features.push(hand.landmarks[i][1] - wrist[1]); // y
    }
    return features;
}

// === LOGIC: FINGER SYNC ===

function processFingerSync(hand) {
    // 5개 손가락 상태 계산 (0~100)
    // 간단한 로직: 손가락 끝(Tip)과 손바닥(Palm/MCP) 사이 거리 기반
    // 랜드마크 인덱스: 엄지(4), 검지(8), 중지(12), 약지(16), 소지(20)
    // 기준점: 손목(0) 또는 각 손가락 시작점(MCP)
    
    let fingers = [];
    let tips = [4, 8, 12, 16, 20];
    let bases = [2, 5, 9, 13, 17]; // 대략적인 마디 기준점
    
    let fingerNames = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    let fingerValues = []; // 0~100 값
    let digitalValues = []; // 0 or 1

    for(let i=0; i<5; i++) {
        let tip = hand.landmarks[tips[i]];
        let base = hand.landmarks[bases[i]];
        
        // 거리 계산
        let dist = Math.hypot(tip[0] - base[0], tip[1] - base[1]);
        
        // 매핑 (손 크기에 따라 다름, 대략적인 튜닝 필요)
        // 엄지는 좀 짧아서 기준이 다름
        let maxDist = (i === 0) ? 60 : 100; 
        let minDist = (i === 0) ? 20 : 30;

        let val = map(dist, minDist, maxDist, 0, 100);
        val = constrain(val, 0, 100);
        
        fingerValues.push(Math.round(val));
        digitalValues.push(val > 50 ? 1 : 0);

        // UI 게이지 업데이트
        select(`#bar-${fingerNames[i]}`).style('width', `${val}%`);
    }

    // 데이터 전송
    if (isTracking && isConnected) {
        // 100ms 마다 전송 (너무 빠르면 렉걸림)
        if (millis() - lastSendTime > 100) {
            let dataStr = "";
            if (fingerFormat === 'analog') {
                // 예: F:100,0,50,0,0
                dataStr = `F:${fingerValues.join(',')}`;
            } else {
                // 예: F:10101
                dataStr = `F:${digitalValues.join('')}`;
            }
            sendBluetoothData(dataStr);
            btDisplayFinger.html(`전송됨: ${dataStr}`);
            lastSendTime = millis();
        }
    }
}

// === UI HELPERS ===

function addUIList(id, name) {
    const div = createDiv(`${id}: ${name} <span id="count-${id}" class="badge">0</span>`);
    div.parent(gestureListContainer);
    div.class('list-item'); // CSS 필요 시 추가
}

function updateGestureCount(id) {
    const count = knnClassifier.getCountByLabel()[id];
    const span = select(`#count-${id}`);
    if (span) span.html(`${count} data`);
}

function resetModel() {
    knnClassifier.clearAllLabels();
    idToNameMap = {};
    nextClassId = 1;
    gestureListContainer.html('');
    gestureResultLabel.html('데이터 없음');
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
    select('#bluetoothStatus').html("연결됨: " + bluetoothDevice.name).style('background', '#E6F4EA');
  } catch (error) {
    console.error(error);
  }
}

function disconnectBluetooth() {
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    bluetoothDevice.gatt.disconnect();
  }
  isConnected = false;
  select('#bluetoothStatus').html("연결 해제됨").style('background', '#F1F3F4');
}

async function sendBluetoothData(data) {
  if (!rxCharacteristic || !isConnected) return;
  try {
    const encoder = new TextEncoder();
    await rxCharacteristic.writeValue(encoder.encode(data + "\n"));
  } catch (error) {
    console.error(error);
  }
}
