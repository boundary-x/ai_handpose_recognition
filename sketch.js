/**
 * sketch.js
 * Boundary X: AI 핸드포즈학습 [Pure JS - p5.js]
 */

// === Bluetooth UUIDs ===
const UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_UUID      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

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
const SEND_INTERVAL = 100;

// === MediaPipe State ===
let handLandmarker = null;
let isModelReady = false;
let lastLandmarks = null;
let lastVideoTime = -1;

// === KNN State ===
let trainingData = [];
const KNN_K = 5;

// === App State ===
let classes = {};
let isTraining = false;
let lastTrainTime = 0;
const TRAIN_INTERVAL = 200;
let isFlipped = true;
let isTracking = false;

// === Canvas / Video ===
let canvas, ctx, videoEl;

// =============================================
// 초기화
// =============================================
async function init() {
  // 캔버스 설정
  canvas = document.getElementById("handpose-canvas");
  ctx = canvas.getContext("2d");

  // 카메라 설정
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 320, height: 240 },
      audio: false
    });
    videoEl = document.getElementById("hidden-video");
    videoEl.srcObject = stream;
    videoEl.play();
  } catch (e) {
    console.error("카메라 오류:", e);
    setStatus("❌ 카메라를 열 수 없습니다. 권한을 확인해주세요.");
    return;
  }

  // MediaPipe 초기화
  await initMediaPipe();

  // 렌더 루프 시작
  renderLoop();
}

async function initMediaPipe() {
  try {
    setStatus("MediaPipe 라이브러리 로딩 중...");
    const m = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8");
    const vision = await m.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
    );
    handLandmarker = await m.HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 1
    });
    isModelReady = true;
    setStatus("✅ 준비 완료! 제스처를 학습시키세요.");
    console.log("MediaPipe HandLandmarker Ready");
  } catch (e) {
    console.error("MediaPipe 초기화 실패:", e);
    setStatus("❌ 모델 로드 실패. 페이지를 새로고침 해주세요.");
  }
}

// =============================================
// 렌더 + 추론 루프 (하나의 루프로 통합)
// =============================================
function renderLoop() {
  requestAnimationFrame(renderLoop);

  if (!videoEl || videoEl.readyState < 2) return;

  // 1. 캔버스에 카메라 그리기 (거울 모드)
  ctx.save();
  if (isFlipped) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  // 2. 추론
  if (isModelReady && videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;
    try {
      const result = handLandmarker.detectForVideo(videoEl, performance.now());
      lastLandmarks = (result.landmarks && result.landmarks.length > 0)
        ? result.landmarks[0] : null;
    } catch (e) { console.error(e); }
  }

  // 3. 랜드마크 그리기
  if (lastLandmarks) drawLandmarks(lastLandmarks);

  // 4. 학습 / 인식
  if (!lastLandmarks) return;
  const features = extractFeatures(lastLandmarks);

  if (isTraining) {
    if (!isModelReady) { setStatus("⚠️ 모델 로딩 중입니다."); return; }
    const label = document.getElementById("class-input").value.trim();
    if (label && performance.now() - lastTrainTime > TRAIN_INTERVAL) {
      addExample(features, label);
      lastTrainTime = performance.now();
    }
  } else if (isTracking && trainingData.length > 0) {
    classifyKNN(features);
  }
}

// =============================================
// 특징 추출
// =============================================
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

// =============================================
// KNN
// =============================================
function addExample(features, label) {
  trainingData.push({ label, features });
  if (!classes[label]) classes[label] = 0;
  classes[label]++;

  // 해당 배지만 실시간 갱신
  const badge = document.querySelector(`.badge-label[data-label="${label}"] .badge-count`);
  if (badge) {
    badge.innerText = `${classes[label]} data`;
  } else {
    updateListUI();
  }
}

function euclideanDistSq(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum;
}

function classifyKNN(features) {
  if (trainingData.length === 0) return;
  const dists = trainingData.map(d => ({
    label: d.label,
    distSq: euclideanDistSq(features, d.features)
  }));
  dists.sort((a, b) => a.distSq - b.distSq);
  const kNearest = dists.slice(0, KNN_K);
  const votes = {};
  for (const n of kNearest) votes[n.label] = (votes[n.label] || 0) + 1;
  const label = Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);
  const conf = votes[label] / KNN_K;

  const resultLabel = document.getElementById("result-label");
  const resultConf  = document.getElementById("result-conf");
  const btDisplay   = document.getElementById("bluetooth-data-display");

  if (resultLabel) { resultLabel.textContent = label; resultLabel.style.color = conf >= 0.85 ? "#00E676" : "#FFEB3B"; }
  if (resultConf)  resultConf.textContent = `정확도: ${(conf * 100).toFixed(0)}%`;

  let msg = `전송 데이터: ${label}`;
  if (!isConnected) msg += " (연결 안됨)";
  if (btDisplay) { btDisplay.textContent = msg; btDisplay.style.color = "#00E676"; }

  if (isConnected && Date.now() - lastSendTime > SEND_INTERVAL) {
    sendBluetoothData(label);
    lastSendTime = Date.now();
  }
}

// =============================================
// 랜드마크 시각화
// =============================================
function drawLandmarks(landmarks) {
  const W = canvas.width, H = canvas.height;
  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17]
  ];

  ctx.strokeStyle = "rgb(0,200,0)";
  ctx.lineWidth = 2;
  for (const [a, b] of connections) {
    let ax = landmarks[a].x * W, ay = landmarks[a].y * H;
    let bx = landmarks[b].x * W, by = landmarks[b].y * H;
    if (isFlipped) { ax = W - ax; bx = W - bx; }
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }

  for (let i = 0; i < landmarks.length; i++) {
    let x = landmarks[i].x * W;
    let y = landmarks[i].y * H;
    if (isFlipped) x = W - x;
    ctx.fillStyle = i === 0 ? "rgb(255,0,0)" : "rgb(0,255,0)";
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
  }
}

// =============================================
// UI
// =============================================
function setStatus(msg) {
  const el = document.getElementById("status-badge");
  if (el) el.textContent = msg;
}

function updateListUI() {
  const list = document.getElementById("training-list");
  if (!list) return;
  list.innerHTML = "";
  if (Object.keys(classes).length === 0) {
    list.innerHTML = '<div class="empty-msg">아직 학습된 데이터가 없습니다.</div>';
    return;
  }
  for (const label in classes) {
    const li   = document.createElement("div"); li.className = "list-item";
    const left = document.createElement("div"); left.className = "list-item-left badge-label"; left.dataset.label = label;
    const nameSpan  = document.createElement("span"); nameSpan.textContent = label;
    const countSpan = document.createElement("span"); countSpan.className = "badge-count"; countSpan.textContent = `${classes[label]} data`;
    left.appendChild(nameSpan); left.appendChild(countSpan); li.appendChild(left);
    const delBtn = document.createElement("button"); delBtn.className = "delete-btn"; delBtn.textContent = "X";
    delBtn.addEventListener("click", () => deleteClass(label));
    li.appendChild(delBtn);
    list.appendChild(li);
  }
}

function deleteClass(label) {
  trainingData = trainingData.filter(d => d.label !== label);
  delete classes[label];
  updateListUI();
  const rl = document.getElementById("result-label");
  const rc = document.getElementById("result-conf");
  if (rl) rl.textContent = "대기 중";
  if (rc) rc.textContent = "데이터 삭제됨";
}

// =============================================
// 전역 함수 (버튼에서 호출)
// =============================================
function clearAllModel() {
  trainingData = []; classes = {};
  updateListUI();
  const rl = document.getElementById("result-label");
  const rc = document.getElementById("result-conf");
  if (rl) { rl.textContent = "대기 중"; rl.style.color = "#00E676"; }
  if (rc) rc.textContent = "데이터 없음";
}

function startTraining() { isTraining = true; }
function stopTraining()  { isTraining = false; }

function startTracking() {
  isTracking = true;
  const el = document.getElementById("bluetooth-data-display");
  if (el) { el.textContent = "데이터 분석 중..."; el.style.color = "#0f0"; }
}

async function stopTracking(sendStopSignal = true) {
  isTracking = false;
  const el = document.getElementById("bluetooth-data-display");
  if (el) { el.textContent = "전송 중지됨"; el.style.color = "#EA4335"; }
  if (!sendStopSignal) return;
  const sent = await sendBluetoothDataReliable("stop");
  if (!sent && isConnected && el) el.textContent = "⚠️ 정지 신호 전송 실패";
}

// =============================================
// Bluetooth
// =============================================
async function connectBluetooth() {
  try {
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: "BBC micro:bit" }],
      optionalServices: [UART_SERVICE_UUID]
    });
    const server  = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE_UUID);
    rxCharacteristic = await service.getCharacteristic(UART_RX_UUID);
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

function disconnectBluetooth() {
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    isManualDisconnect = true;
    bluetoothDevice.gatt.disconnect();
  } else {
    isConnected = false; rxCharacteristic = null; bluetoothDevice = null;
    bluetoothStatus = "연결 해제됨";
    updateBluetoothStatusUI(false);
  }
}

function onDisconnected() {
  isConnected = false; rxCharacteristic = null; bluetoothDevice = null;
  const wasTracking = isTracking;
  if (isTracking) stopTracking(false);
  bluetoothStatus = isManualDisconnect ? "연결 해제됨" : "연결이 끊어졌습니다. 다시 연결해주세요.";
  updateBluetoothStatusUI(false, !isManualDisconnect);
  const el = document.getElementById("bluetooth-data-display");
  if (wasTracking && el) {
    el.textContent = isManualDisconnect ? "연결 해제로 인식이 중지되었습니다" : "⚠️ 연결이 끊어져 인식이 자동으로 중지되었습니다";
    el.style.color = isManualDisconnect ? "#888" : "#EA4335";
  }
  isManualDisconnect = false;
}

function updateBluetoothStatusUI(connected = false, error = false) {
  const el = document.getElementById("bluetoothStatus");
  if (!el) return;
  el.textContent = `상태: ${bluetoothStatus}`;
  el.classList.remove("status-connected", "status-error");
  if (connected) el.classList.add("status-connected");
  else if (error) el.classList.add("status-error");
}

async function sendBluetoothData(data) {
  if (!rxCharacteristic || !isConnected) return false;
  if (isSendingData) return false;
  try {
    isSendingData = true;
    await withTimeout(rxCharacteristic.writeValue(new TextEncoder().encode(data + "\n")), 2000);
    return true;
  } catch (error) {
    console.error(error);
    const now = Date.now();
    if (now - lastSendErrorTime > 3000) {
      lastSendErrorTime = now;
      bluetoothStatus = "⚠️ 데이터 전송 실패";
      updateBluetoothStatusUI(false, true);
    }
    return false;
  } finally {
    isSendingData = false;
  }
}

async function sendBluetoothDataReliable(data, maxRetries = 5, retryDelayMs = 80) {
  for (let i = 0; i < maxRetries; i++) {
    if (await sendBluetoothData(data)) return true;
    await new Promise(r => setTimeout(r, retryDelayMs));
  }
  return false;
}

// 페이지 로드 시 초기화
window.addEventListener("load", init);
