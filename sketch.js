/**
 * sketch.js
 * Boundary X: AI 핸드포즈학습 [MediaPipe Edition]
 *
 * 변경사항:
 * - ml5.js Handpose(TF.js 1.x) → MediaPipe HandLandmarker(WASM/GPU)로 교체 → 프레임 드랍 해소
 * - 추론 루프를 p5.js draw()와 완전 분리 → 렌더링은 항상 부드럽게 유지
 * - 모델 미준비 시 학습 시도 → 명확한 안내 표시
 * - 학습 중 DOM 갱신 → 버튼을 뗄 때만 한 번 업데이트
 * - gattserverdisconnected 이벤트 처리 → 끊김 즉시 감지
 * - stop 명령 재시도(sendBluetoothDataReliable)
 * - writeValue 2초 타임아웃
 * - 전송 실패 UI 피드백
 * - 인식 중 연결 끊기면 자동 중지
 */

import { HandLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8";

// === Bluetooth UUIDs ===
const UART_SERVICE_UUID    = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_TX_UUID         = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_UUID         = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

// BLE 응답이 영영 안 올 때 강제로 실패 처리
function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("BLE write timeout")), ms))
  ]);
}

// === BLE State ===
let bluetoothDevice = null;
let rxCharacteristic = null;
let isConnected = false;
let bluetoothStatus = "연결 대기 중";
let isSendingData = false;
let isManualDisconnect = false;
let lastSendErrorTime = 0;
let lastSendTime = 0;
const SEND_INTERVAL = 100; // ms

// === MediaPipe State ===
let handLandmarker = null;
let isModelReady = false;

// 추론은 별도 루프에서 돌고, draw()는 이 결과를 읽기만 함
let lastLandmarks = null;      // 최신 추론 결과 (21개 랜드마크)
let isRunningInference = false; // 추론 중복 호출 방지
let lastVideoTime = -1;

// === KNN State ===
// features 배열의 배열을 {label, features}로 보관 (ml5 없이 직접 구현)
let trainingData = [];
const KNN_K = 5; // 최근접 이웃 수

// === App State ===
let video;
let classes = {};      // { label: count }
let isTraining = false;
let pendingTrainUpdate = false; // 학습 중 UI 업데이트 예약 플래그
let isFlipped = true;
let isTracking = false;

// === UI Elements ===
let classInput, addDataBtn, resetBtn;
let resultLabel, resultConf, btDataDisplay;
let trainingList, statusBadge;

// === MediaPipe 초기화 ===
async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
  isModelReady = true;
  if (statusBadge) statusBadge.html("준비 완료! 제스처를 학습시키세요.");
  console.log("MediaPipe HandLandmarker Ready");

  // 추론 루프 시작 (draw()와 독립적으로 돌아감)
  inferenceLoop();
}

// 추론 루프: draw()와 완전 분리 — 프레임 드랍 없이 카메라 화면이 부드럽게 유지됨
async function inferenceLoop() {
  if (!isModelReady || !video) {
    requestAnimationFrame(inferenceLoop);
    return;
  }

  const videoEl = video.elt;
  if (videoEl.readyState >= 2 && videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;
    try {
      const result = handLandmarker.detectForVideo(videoEl, performance.now());
      if (result.landmarks && result.landmarks.length > 0) {
        lastLandmarks = result.landmarks[0]; // 21개 {x,y,z} 객체 배열
      } else {
        lastLandmarks = null;
      }
    } catch (e) {
      console.error("HandLandmarker 추론 오류:", e);
    }
  }
  requestAnimationFrame(inferenceLoop);
}

// === p5.js Setup ===
function setup() {
  let canvas = createCanvas(320, 240);
  canvas.parent("p5-container");

  video = createCapture({ video: { facingMode: "user", width: 320, height: 240 }, audio: false });
  video.size(320, 240);
  video.hide();

  setupUI();
  initMediaPipe();
}

// === p5.js Draw (렌더링만 담당) ===
function draw() {
  background(0);

  // 1. 카메라 화면 그리기
  push();
  if (isFlipped) { translate(width, 0); scale(-1, 1); }
  if (video.elt.readyState >= 2) image(video, 0, 0, width, height);
  pop();

  // 2. 랜드마크 그리기
  if (lastLandmarks) drawLandmarks(lastLandmarks);

  // 3. 학습 / 인식 로직 (추론 결과가 있을 때만)
  if (!lastLandmarks) return;

  const features = extractFeatures(lastLandmarks);

  if (isTraining) {
    const label = classInput.value().trim();
    if (!isModelReady) {
      if (statusBadge) statusBadge.html("⚠️ 모델이 아직 로딩 중입니다. 잠시 후 다시 시도해주세요.");
      return;
    }
    if (label) {
      addExample(features, label);
      pendingTrainUpdate = true; // 버튼을 뗄 때 한 번만 DOM 갱신
    }
  } else if (isTracking && trainingData.length > 0) {
    classifyKNN(features);
  }
}

// === 특징 추출 (손목 기준 상대 좌표 + 스케일 정규화) ===
function extractFeatures(landmarks) {
  const wrist = landmarks[0];
  let maxDist = 0;

  for (let i = 1; i < landmarks.length; i++) {
    const dx = landmarks[i].x - wrist.x;
    const dy = landmarks[i].y - wrist.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > maxDist) maxDist = d;
  }
  if (maxDist < 0.001) maxDist = 0.001;

  const features = [];
  for (let i = 1; i < landmarks.length; i++) {
    features.push((landmarks[i].x - wrist.x) / maxDist);
    features.push((landmarks[i].y - wrist.y) / maxDist);
  }
  return features;
}

// === KNN 분류기 (직접 구현, ml5 의존성 제거) ===
function euclideanDistSq(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum;
}

function addExample(features, label) {
  trainingData.push({ label, features });
  if (!classes[label]) classes[label] = 0;
  classes[label]++;
  // DOM 갱신은 하지 않음 — 버튼 릴리즈 시 pendingTrainUpdate로 처리
}

function classifyKNN(features) {
  if (trainingData.length === 0) return;

  // 거리 계산 후 K개 최근접 이웃 투표
  const dists = trainingData.map(d => ({
    label: d.label,
    distSq: euclideanDistSq(features, d.features)
  }));
  dists.sort((a, b) => a.distSq - b.distSq);
  const kNearest = dists.slice(0, KNN_K);

  const votes = {};
  for (const n of kNearest) {
    votes[n.label] = (votes[n.label] || 0) + 1;
  }
  const label = Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);
  const conf = votes[label] / KNN_K;

  // UI 갱신
  resultLabel.html(label);
  resultConf.html(`정확도: ${(conf * 100).toFixed(0)}%`);
  resultLabel.style("color", conf >= 0.85 ? "#00E676" : "#FFEB3B");

  let displayMsg = `전송 데이터: ${label}`;
  if (!isConnected) displayMsg += " (연결 안됨)";
  btDataDisplay.html(displayMsg);
  btDataDisplay.style("color", "#00E676");

  // BLE 전송 (스로틀)
  if (isConnected && millis() - lastSendTime > SEND_INTERVAL) {
    sendBluetoothData(label);
    lastSendTime = millis();
  }
}

// === 랜드마크 시각화 ===
function drawLandmarks(landmarks) {
  // 손가락 연결선 (MediaPipe HandLandmarker 연결 인덱스)
  const connections = [
    [0,1],[1,2],[2,3],[3,4],       // 엄지
    [0,5],[5,6],[6,7],[7,8],       // 검지
    [0,9],[9,10],[10,11],[11,12],  // 중지
    [0,13],[13,14],[14,15],[15,16],// 약지
    [0,17],[17,18],[18,19],[19,20],// 소지
    [5,9],[9,13],[13,17]           // 손바닥 가로
  ];

  stroke(0, 200, 0);
  strokeWeight(2);
  for (const [a, b] of connections) {
    let ax = landmarks[a].x * width;
    let ay = landmarks[a].y * height;
    let bx = landmarks[b].x * width;
    let by = landmarks[b].y * height;
    if (isFlipped) { ax = width - ax; bx = width - bx; }
    line(ax, ay, bx, by);
  }

  noStroke();
  for (let i = 0; i < landmarks.length; i++) {
    let x = landmarks[i].x * width;
    let y = landmarks[i].y * height;
    if (isFlipped) x = width - x;
    fill(i === 0 ? color(255, 0, 0) : color(0, 255, 0));
    ellipse(x, y, 7, 7);
  }
}

// === 학습 목록 UI ===
function updateListUI() {
  trainingList.html("");
  if (Object.keys(classes).length === 0) {
    trainingList.html('<div class="empty-msg">아직 학습된 데이터가 없습니다.</div>');
    return;
  }
  for (const label in classes) {
    const li = createDiv().addClass("list-item");
    const left = createDiv().addClass("list-item-left");
    createSpan(label).parent(left);
    createSpan(`${classes[label]} data`).addClass("badge-count").parent(left);
    left.parent(li);
    const delBtn = createButton("X").addClass("delete-btn");
    delBtn.mousePressed(() => deleteClass(label));
    delBtn.parent(li);
    li.parent(trainingList);
  }
}

function deleteClass(label) {
  trainingData = trainingData.filter(d => d.label !== label);
  delete classes[label];
  updateListUI();
  resultLabel.html("대기 중");
  resultConf.html("데이터 삭제됨");
}

function clearAllModel() {
  trainingData = [];
  classes = {};
  updateListUI();
  resultLabel.html("대기 중");
  resultLabel.style("color", "#00E676");
  resultConf.html("데이터 없음");
}

// === UI 설정 ===
function setupUI() {
  statusBadge  = select("#status-badge");
  classInput   = select("#class-input");
  addDataBtn   = select("#add-data-btn");
  resetBtn     = select("#reset-btn");
  trainingList = select("#training-list");
  resultLabel  = select("#result-label");
  resultConf   = select("#result-conf");
  btDataDisplay = select("#bluetooth-data-display");

  addDataBtn.mousePressed(() => { isTraining = true; });
  addDataBtn.mouseReleased(() => {
    isTraining = false;
    if (pendingTrainUpdate) {
      updateListUI(); // 버튼을 뗄 때 한 번만 DOM 갱신
      pendingTrainUpdate = false;
    }
  });
  resetBtn.mousePressed(clearAllModel);

  // 블루투스 버튼
  const connectBtn = createButton("기기 연결");
  connectBtn.parent("bluetooth-control-buttons").addClass("start-button");
  connectBtn.mousePressed(connectBluetooth);

  const disconnectBtn = createButton("연결 해제");
  disconnectBtn.parent("bluetooth-control-buttons").addClass("stop-button");
  disconnectBtn.mousePressed(disconnectBluetooth);

  // 인식 제어 버튼
  const startTrackBtn = createButton("인식 시작");
  startTrackBtn.parent("recognition-control-buttons").addClass("start-button");
  startTrackBtn.mousePressed(() => {
    isTracking = true;
    btDataDisplay.html("데이터 분석 중...");
    btDataDisplay.style("color", "#0f0");
  });

  const stopTrackBtn = createButton("인식 중지");
  stopTrackBtn.parent("recognition-control-buttons").addClass("stop-button");
  stopTrackBtn.mousePressed(() => stopTracking());

  updateBluetoothStatusUI();
}

// 인식을 멈추고 stop 신호 전송
async function stopTracking(sendStopSignal = true) {
  isTracking = false;
  btDataDisplay.html("전송 중지됨");
  btDataDisplay.style("color", "#EA4335");

  if (!sendStopSignal) return;

  const sent = await sendBluetoothDataReliable("stop");
  if (!sent && isConnected) {
    btDataDisplay.html("⚠️ 정지 신호 전송 실패 - 연결을 확인해주세요");
    btDataDisplay.style("color", "#EA4335");
  }
}

// === Bluetooth Logic ===
async function connectBluetooth() {
  try {
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: "BBC micro:bit" }],
      optionalServices: [UART_SERVICE_UUID]
    });
    const server  = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE_UUID);
    rxCharacteristic = await service.getCharacteristic(UART_RX_UUID);

    // 예기치 않게 끊겼을 때도 상태를 동기화
    bluetoothDevice.addEventListener("gattserverdisconnected", onDisconnected);

    isConnected = true;
    bluetoothStatus = "연결됨: " + bluetoothDevice.name;
    updateBluetoothStatusUI(true);
  } catch (error) {
    console.error(error);
    bluetoothStatus = "연결 실패";
    updateBluetoothStatusUI(false, true);
  }
}

// 수동 해제든 예기치 않은 끊김이든 이 함수 하나로 상태를 정리
function onDisconnected() {
  isConnected = false;
  rxCharacteristic = null;
  bluetoothDevice = null;

  // 인식 중이었으면 자동 중지 (이미 끊겼으므로 stop 전송 시도 불필요)
  const wasTracking = isTracking;
  if (isTracking) stopTracking(false);

  if (isManualDisconnect) {
    bluetoothStatus = "연결 해제됨";
    updateBluetoothStatusUI(false);
  } else {
    bluetoothStatus = "연결이 끊어졌습니다. 다시 연결해주세요.";
    updateBluetoothStatusUI(false, true);
  }

  if (wasTracking) {
    btDataDisplay.html(
      isManualDisconnect
        ? "연결 해제로 인식이 중지되었습니다"
        : "⚠️ 연결이 끊어져 인식이 자동으로 중지되었습니다"
    );
    btDataDisplay.style("color", isManualDisconnect ? "#888" : "#EA4335");
  }
  isManualDisconnect = false;
}

function disconnectBluetooth() {
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    isManualDisconnect = true;
    bluetoothDevice.gatt.disconnect();
  } else {
    isConnected = false;
    bluetoothStatus = "연결 해제됨";
    rxCharacteristic = null;
    bluetoothDevice = null;
    updateBluetoothStatusUI(false);
  }
}

function updateBluetoothStatusUI(connected = false, error = false) {
  const el = select("#bluetoothStatus");
  if (el) {
    el.html(`상태: ${bluetoothStatus}`);
    el.removeClass("status-connected").removeClass("status-error");
    if (connected) el.addClass("status-connected");
    else if (error) el.addClass("status-error");
  }
}

// 성공하면 true, 스킵되거나 실패하면 false를 반환
async function sendBluetoothData(data) {
  if (!rxCharacteristic || !isConnected) return false;
  if (isSendingData) return false;
  try {
    isSendingData = true;
    const encoder = new TextEncoder();
    // writeValue가 끝내 응답하지 않는 경우를 대비해 2초 타임아웃
    await withTimeout(
      rxCharacteristic.writeValue(encoder.encode(data + "\n")), 2000
    );
    return true;
  } catch (error) {
    console.error(error);
    const now = Date.now();
    if (now - lastSendErrorTime > 3000) {
      lastSendErrorTime = now;
      const el = select("#bluetoothStatus");
      if (el) {
        el.html("⚠️ 데이터 전송 실패 - 연결 상태를 확인해주세요")
          .removeClass("status-connected").addClass("status-error");
      }
    }
    return false;
  } finally {
    isSendingData = false;
  }
}

// stop처럼 반드시 전달되어야 하는 명령을 위한 재시도 버전
async function sendBluetoothDataReliable(data, maxRetries = 5, retryDelayMs = 80) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    if (await sendBluetoothData(data)) return true;
    await new Promise(r => setTimeout(r, retryDelayMs));
  }
  console.error(`전송 재시도 실패: ${data}`);
  return false;
}
