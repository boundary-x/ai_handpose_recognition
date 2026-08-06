/**
 * sketch.js
 * Boundary X: AI Gesture Learning [안정화 & 모바일 최적화 버전]
 * Feature: MediaPipe Hands 적용, Bluetooth 예외 완벽 처리, 큐(Queue) 전송 방식 적용
 */

// Bluetooth UUIDs
const UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_TX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

let bluetoothDevice, rxCharacteristic, isConnected = false;
let bluetoothStatus = "연결 대기 중";
let isSendingData = false;
let lastSendTime = 0;
let btQueue = []; // ✨ 데이터 유실 방지를 위한 큐 시스템

let video;
let knnClassifier;
let currentLandmarks = null;
let isModelReady = false;

// 성능 최적화 변수
let lastClassifyTime = 0;
let lastLabel = ""; 

// UI Elements
let classInput, addDataBtn, resetBtn;
let resultLabel, resultConf, btDataDisplay;
let trainingList, statusBadge;

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

  knnClassifier = ml5.KNNClassifier();

  setupUI();
  setupMediaPipe();
}

function setupMediaPipe() {
  const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  }});

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 0,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  hands.onResults((results) => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      currentLandmarks = results.multiHandLandmarks[0];
    } else {
      currentLandmarks = null;
    }
  });

  const camera = new Camera(video.elt, {
    onFrame: async () => {
      await hands.send({image: video.elt});
    },
    width: 320,
    height: 240
  });
  camera.start();

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

  // ✨ 개선: 조용히 무시되는 현상 수정 (모델 미준비/손 미인식 시 경고 UI)
  // pointerdown/up을 사용하여 모바일 터치와 마우스 클릭 모두 완벽 지원
  addDataBtn.elt.addEventListener('pointerdown', () => {
    if (!isModelReady) {
        statusBadge.html("🚨 AI 모델이 아직 준비되지 않았습니다!");
        statusBadge.style('background-color', '#EA4335');
        setTimeout(() => {
            statusBadge.html("준비 완료! 제스처를 학습시키세요.");
            statusBadge.style('background-color', 'rgba(0,0,0,0.7)');
        }, 2000);
        return;
    }
    if (!currentLandmarks) {
        resultLabel.html("손 인식 실패");
        resultLabel.style('color', '#EA4335');
        resultConf.html("화면에 손을 명확히 보여주세요");
        return;
    }
    isTraining = true;
  });
  
  addDataBtn.elt.addEventListener('pointerup', () => isTraining = false);
  addDataBtn.elt.addEventListener('pointerleave', () => isTraining = false);
  
  // 초기화 버튼 이벤트
  resetBtn.elt.addEventListener('click', clearAllModel);

  // --- 기기 연결 (순수 JS 이벤트로 팝업 차단 방지) ---
  let connectBtn = createButton("기기 연결");
  connectBtn.parent('bluetooth-control-buttons');
  connectBtn.addClass('start-button');
  connectBtn.elt.addEventListener('click', connectBluetooth);

  let disconnectBtn = createButton("연결 해제");
  disconnectBtn.parent('bluetooth-control-buttons');
  disconnectBtn.addClass('stop-button');
  disconnectBtn.elt.addEventListener('click', disconnectBluetooth);

  // --- 인식 제어 ---
  let startTrackBtn = createButton("인식 시작");
  startTrackBtn.parent('recognition-control-buttons');
  startTrackBtn.addClass('start-button');
  startTrackBtn.elt.addEventListener('click', () => { 
      isTracking = true; 
      btDataDisplay.html("데이터 분석 중...");
      btDataDisplay.style('color', '#0f0'); 
  });

  let stopTrackBtn = createButton("인식 중지");
  stopTrackBtn.parent('recognition-control-buttons');
  stopTrackBtn.addClass('stop-button');
  stopTrackBtn.elt.addEventListener('click', () => { 
      isTracking = false; 
      sendBluetoothData("stop", true); // ✨ 개선: 긴급 정지 플래그(true) 전송
      btDataDisplay.html("전송 중지됨");
      btDataDisplay.style('color', '#EA4335'); 
  });

  updateBluetoothStatusUI();
}

function draw() {
  background(0);

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

  if (currentLandmarks) {
    let features = extractRelativeFeatures(currentLandmarks);

    if (isTraining) {
      let label = classInput.value().trim();
      if (label) addExample(features, label);
    } 
    else if (knnClassifier.getNumLabels() > 0) {
      if (millis() - lastClassifyTime > 150) {
        classify(features);
        lastClassifyTime = millis();
      }
    }
  }
}

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

/* --- ✨ 완벽하게 개선된 Bluetooth Logic --- */
async function connectBluetooth() {
  try {
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: "BBC micro:bit" }],
      optionalServices: [UART_SERVICE_UUID]
    });
    
    // ✨ 개선: 기기 연결 단절(물리적 이탈, 전원 꺼짐 등) 이벤트 등록
    bluetoothDevice.addEventListener('gattserverdisconnected', onDisconnected);

    const server = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE_UUID);
    rxCharacteristic = await service.getCharacteristic(UART_RX_CHARACTERISTIC_UUID);
    
    isConnected = true;
    bluetoothStatus = "연결됨: " + bluetoothDevice.name;
    updateBluetoothStatusUI(true);
  } catch (error) {
    console.error("BT Connect Error:", error);
    bluetoothStatus = "연결 실패: " + error.message;
    updateBluetoothStatusUI(false, true);
  }
}

function disconnectBluetooth() {
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    bluetoothDevice.gatt.disconnect(); // 이 호출이 onDisconnected를 트리거합니다.
  } else {
    onDisconnected(); // 이미 끊긴 경우 강제 처리
  }
}

// ✨ 개선: 단절 이벤트 처리 함수
function onDisconnected() {
  isConnected = false;
  isSendingData = false;
  btQueue = []; // 대기 중인 전송 비우기
  bluetoothStatus = "연결 끊김 (기기 이탈)";
  rxCharacteristic = null;
  updateBluetoothStatusUI(false, true);
  
  if (btDataDisplay) {
      btDataDisplay.html("블루투스 연결이 끊어졌습니다.");
      btDataDisplay.style('color', '#EA4335');
  }
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

// ✨ 개선: Queue를 통한 데이터 유실 방지 및 타임아웃 처리
async function sendBluetoothData(data, isUrgent = false) {
  if (!rxCharacteristic || !isConnected) return;
  
  if (isUrgent) {
      btQueue = []; // 'stop' 같은 긴급 명령은 큐를 모두 비우고 최우선으로 등록
      btQueue.push(data);
  } else {
      // 일반 데이터는 너무 많이 쌓이면 딜레이가 발생하므로 오래된 것을 버림 (최대 3개 유지)
      if (btQueue.length > 3) btQueue.shift(); 
      btQueue.push(data);
  }
  
  processQueue();
}

async function processQueue() {
  if (isSendingData || btQueue.length === 0) return;
  
  isSendingData = true;
  let data = btQueue.shift();

  try {
    const encoder = new TextEncoder();
    const value = encoder.encode(data + "\n");
    
    // ✨ 개선: 무한 대기 방지용 타임아웃 2초(2000ms) 프로미스
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error("Timeout (응답 없음)")), 2000);
    });

    // Promise.race를 통해 전송과 타임아웃 중 먼저 끝나는 것을 처리
    await Promise.race([
        rxCharacteristic.writeValue(value),
        timeoutPromise
    ]);
    
  } catch (error) {
    console.error("BT Send Error:", error);
    // ✨ 개선: 에러 발생 시 콘솔뿐만 아니라 UI에도 즉각 피드백
    btDataDisplay.html(`전송 에러: ${error.message}`);
    btDataDisplay.style('color', '#EA4335'); 
  } finally {
    isSendingData = false;
    // 큐에 남은 데이터가 있다면 연속해서 처리
    if (btQueue.length > 0) {
        processQueue();
    }
  }
}
