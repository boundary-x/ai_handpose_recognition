/**
 * sketch.js
 * Boundary X: AI Gesture Learning [MediaPipe + 모바일 최적화 버전]
 * Feature: MediaPipe Hands 적용, 분석 횟수 제한(Throttling) 및 UI 렌더링 최적화
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
let knnClassifier;
let currentLandmarks = null; // MediaPipe 결과 저장
let isModelReady = false;

// 성능 최적화 변수 (프레임 드랍 방지)
let lastClassifyTime = 0;
let lastLabel = ""; 

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

  video = createCapture(VIDEO);
  video.size(320, 240);
  video.hide();

  // KNN 모델 준비 (데이터 학습용)
  knnClassifier = ml5.KNNClassifier();

  setupUI();
  
  // 구글 MediaPipe 설정 및 실행
  setupMediaPipe();
}

function setupMediaPipe() {
  const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  }});

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 0, // 0이 가장 가볍고 빠름 (모바일/태블릿용)
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  hands.onResults((results) => {
    // 손이 인식되면 좌표 데이터를 저장
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      currentLandmarks = results.multiHandLandmarks[0];
    } else {
      currentLandmarks = null;
    }
  });

  // 비디오 프레임을 MediaPipe로 전달
  const camera = new Camera(video.elt, {
    onFrame: async () => {
      await hands.send({image: video.elt});
    },
    width: 320,
    height: 240
  });
  camera.start();

  console.log("MediaPipe Ready");
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

  let connectBtn = createButton("기기 연결");
  connectBtn.parent('bluetooth-control-buttons');
  connectBtn.addClass('start-button');
  connectBtn.mousePressed(connectBluetooth);

  let disconnectBtn = createButton("연결 해제");
  disconnectBtn.parent('bluetooth-control-buttons');
  disconnectBtn.addClass('stop-button');
  disconnectBtn.mousePressed(disconnectBluetooth);

  let startTrackBtn = createButton("인식 시작");
  startTrackBtn.parent('recognition-control-buttons');
  startTrackBtn.addClass('start-button');
  startTrackBtn.mousePressed(() => { 
      isTracking = true; 
      btDataDisplay.html("데이터 분석 중...");
      btDataDisplay.style('color', '#0f0'); 
  });

  let stopTrackBtn = createButton("인식 중지");
  stopTrackBtn.parent('recognition-control-buttons');
  stopTrackBtn.addClass('stop-button');
  stopTrackBtn.mousePressed(() => { 
      isTracking = false; 
      sendBluetoothData("stop");
      btDataDisplay.html("전송 중지됨");
      btDataDisplay.style('color', '#EA4335'); 
  });

  updateBluetoothStatusUI();
}

function draw() {
  background(0);

  // 화면 그리기 (거울 모드)
  push();
  if (isFlipped) {
      translate(width, 0);
      scale(-1, 1);
  }
  
  if (video.elt.readyState >= 2) {
      image(video, 0, 0, width, height);
      if (currentLandmarks) {
          drawKeypoints(currentLandmarks);
      }
  }
  pop();

  // 손이 화면에 인식되었을 때
  if (currentLandmarks) {
    let features = extractRelativeFeatures(currentLandmarks);

    if (isTraining) {
      let label = classInput.value().trim();
      if (label) addExample(features, label);
    } 
    else if (knnClassifier.getNumLabels() > 0) {
      // ✨ 성능 최적화: 150ms 마다 한 번씩만 분석 실행 (과부하 방지)
      if (millis() - lastClassifyTime > 150) {
        classify(features);
        lastClassifyTime = millis();
      }
    }
  }
}

/**
 * 손목 기준 상대 좌표 변환 (거리 및 크기 무시)
 */
function extractRelativeFeatures(landmarks) {
  let features = [];
  let wrist = landmarks[0]; 

  let maxDist = 0;
  for (let i = 1; i < landmarks.length; i++) {
      let dx = landmarks[i].x - wrist.x;
      let dy = landmarks[i].y - wrist.y;
      let dist = Math.sqrt(dx*dx + dy*dy);
      if (dist > maxDist) maxDist = dist;
  }
  
  if (maxDist < 0.0001) maxDist = 0.0001;

  for (let i = 1; i < landmarks.length; i++) {
    // MediaPipe 좌표 구조(.x, .y)에 맞춰 변경됨
    let relativeX = (landmarks[i].x - wrist.x) / maxDist;
    let relativeY = (landmarks[i].y - wrist.y) / maxDist;
    
    features.push(relativeX);
    features.push(relativeY);
  }
  
  return features; 
}

function addExample(features, label) {
  knnClassifier.addExample(features, label);
  if (!classes[label]) classes[label] = 0;
  classes[label]++;
  
  // UI 업데이트 빈도 조절: 학습 데이터가 10의 배수일 때만 화면 갱신
  if (classes[label] % 10 === 0 || classes[label] === 1) {
      updateListUI();
  }
}

function classify(features) {
  knnClassifier.classify(features, (err, result) => {
    if (err) return;
    
    if (result.confidencesByLabel) {
      const label = result.label;
      const conf = result.confidencesByLabel[label];
      
      // ✨ UI 업데이트 최적화: 결과가 이전과 다를 때만 글씨 변경
      if (label !== lastLabel || conf < 0.85) {
          resultLabel.html(label);
          resultConf.html(`정확도: ${(conf * 100).toFixed(0)}%`);
          
          if (conf > 0.85) resultLabel.style('color', '#00E676');
          else resultLabel.style('color', '#FFEB3B');
          
          lastLabel = label;
      }

      if (isTracking) {
          if (isConnected && millis() - lastSendTime > 150) {
              sendBluetoothData(label);
              lastSendTime = millis();
              
              btDataDisplay.html(`전송 데이터: ${label}`);
              btDataDisplay.style('color', '#00E676');
          } else if (!isConnected) {
              btDataDisplay.html(`전송 데이터: ${label} (연결 안됨)`);
              btDataDisplay.style('color', '#FFEB3B');
          }
      }
    }
  });
}

function drawKeypoints(landmarks) {
  for (let j = 0; j < landmarks.length; j += 1) {
    // MediaPipe는 0~1 사이의 비율 값을 주므로 너비/높이를 곱해줌
    let x = landmarks[j].x * width;
    let y = landmarks[j].y * height;
    
    if (j === 0) fill(255, 0, 0); 
    else fill(0, 255, 0);         
    noStroke();
    ellipse(x, y, 8, 8);
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
        lastLabel = "";
    }
}

function clearAllModel() {
  knnClassifier.clearAllLabels();
  classes = {};
  updateListUI();
  resultLabel.html("대기 중");
  resultLabel.style('color', '#00E676');
  resultConf.html("데이터 없음");
  lastLabel = "";
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
