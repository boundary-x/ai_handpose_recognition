/* Boundary X — MediaPipe hand landmarks + custom KNN. */
const UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";
const SEND_INTERVAL = 100;
const HAND_FRESH_MS = 500;
let video, handLandmarker;
let isModelReady = false, modelLoadFailed = false;
let lastLandmarks = null, lastFeatures = null, lastVideoTime = -1;
let frameId = 0, lastSampleFrame = -1, lastHandSeenAt = -Infinity;
let handAvailable = false, handLossSent = false;
let handLossStatus = "";
let trainingData = [], classIds = [], nextClassId = 1;
let isTracking = false, isBusy = false, isFlipped = true;
let trackingEpoch = 0, training;
let bluetoothDevice = null, rxCharacteristic = null;
let isConnected = false, isConnecting = false, isManualDisconnect = false;
let bluetoothStatus = "연결 대기 중", lastSentLabel = "", lastSendTime = 0;
let sendQueue = Promise.resolve(), predictionSendPending = false;
const byId = id => document.getElementById(id);
const setText = (id, text) => { byId(id).textContent = text; };
const fileStatus = message => setText("file-status", message);
const trainingStatus = message => setText("training-status", message);

function setup() {
  const canvas = createCanvas(320, 240);
  canvas.parent("p5-container");
  video = createCapture({video: {facingMode: "user", width: 320, height: 240}, audio: false});
  video.size(320, 240);
  video.elt.setAttribute("playsinline", "");
  video.elt.muted = true;
  video.hide();
  createUI();
  initMediaPipe();
}
function createUI() {
  training = TrainingInput.createController({
    collect: collectSample,
    onStart: () => { if (isTracking) stopTracking(); }
  });
  byId("add-class-btn").addEventListener("click", addClass);
  byId("download-model-btn").addEventListener("click", downloadModel);
  byId("share-model-btn").addEventListener("click", shareModel);
  byId("import-model-btn").addEventListener("click", () => {
    training.stop(); byId("model-file-input").click();
  });
  byId("model-file-input").addEventListener("change", event => importModel(event.target.files[0]));
  const button = (id, label, parent, handler, style = "start-button") => {
    const node = document.createElement("button");
    node.type = "button"; node.id = id; node.className = style; node.textContent = label;
    node.addEventListener("click", handler); byId(parent).appendChild(node);
  };
  button("reset-model-btn", "🗑️ 데이터 초기화", "reset-btn-container", clearAllModel, "reset-button");
  button("connect-btn", "AI 로딩 중...", "bluetooth-control-buttons", connectBluetooth);
  button("disconnect-btn", "연결 해제", "bluetooth-control-buttons", disconnectBluetooth, "stop-button");
  button("start-track-btn", "인식 시작", "recognition-control-buttons", startTracking);
  button("stop-track-btn", "인식 중지", "recognition-control-buttons", () => stopTracking(), "stop-button");
  const suspend = () => {
    training.stop(); stopTracking(); invalidateHand();
  };
  window.addEventListener("blur", suspend);
  window.addEventListener("pagehide", suspend);
  document.addEventListener("visibilitychange", () => { if (document.hidden) suspend(); });
  const header = document.querySelector("header");
  const updateHeader = () => document.documentElement.style.setProperty("--header-height", header.getBoundingClientRect().height + "px");
  new ResizeObserver(updateHeader).observe(header);
  updateHeader(); renderClasses(); updateControls();
}
function updateControls() {
  document.querySelectorAll(".train-btn").forEach(button => {
    button.disabled = isBusy || !isModelReady || !handAvailable;
  });
  document.querySelectorAll(".delete-btn, #add-class-btn, #reset-model-btn, #import-model-btn")
    .forEach(button => { button.disabled = isBusy; });
  byId("download-model-btn").disabled = isBusy || !trainingData.length;
  byId("share-model-btn").disabled = isBusy || !trainingData.length;
  byId("start-track-btn").disabled = isBusy || !isModelReady || !trainingData.length;
  byId("connect-btn").disabled = !isModelReady || isConnecting || isConnected;
  byId("connect-btn").textContent = isConnected ? "연결됨" : isConnecting ? "연결 중..." : isModelReady ? "기기 연결" : modelLoadFailed ? "AI 로드 실패" : "AI 로딩 중...";
}
async function initMediaPipe() {
  try {
    setText("status-badge", "MediaPipe 로딩 중...");
    const m = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8");
    const vision = await m.FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm");
    handLandmarker = await m.HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO", numHands: 1
    });
    isModelReady = true;
    setText("status-badge", "손을 카메라에 비춰주세요");
    updateControls();
    inferenceLoop();
  } catch (error) {
    console.error(error);
    modelLoadFailed = true; isModelReady = false;
    setText("status-badge", "모델 로드 실패");
    trainingStatus("모델을 불러오지 못했습니다. 연결 상태를 확인한 뒤 새로고침해주세요.");
    updateControls();
  }
}
function inferenceLoop() {
  if (isModelReady && video && handLandmarker && !document.hidden) {
    const input = video.elt;
    if (input.readyState >= 2 && input.currentTime !== lastVideoTime) {
      lastVideoTime = input.currentTime;
      try {
        const result = handLandmarker.detectForVideo(input, performance.now());
        handleHandResult(result.landmarks && result.landmarks[0]);
      } catch (error) {
        console.error(error);
        invalidateHand();
      }
    }
  }
  requestAnimationFrame(inferenceLoop);
}
function handleHandResult(landmarks) {
  const features = HandModel.extractFeatures(landmarks);
  if (!features) { invalidateHand(); return; }
  frameId++;
  lastLandmarks = landmarks; lastFeatures = features;
  lastHandSeenAt = performance.now(); handLossSent = false; handLossStatus = "";
  if (!handAvailable) {
    handAvailable = true;
    setText("status-badge", "손 감지됨");
    updateControls();
  }
  // Classify once per fresh camera result, not once per render of cached landmarks.
  if (isTracking && !isBusy) showPrediction(HandModel.classify(trainingData, features));
}
function invalidateHand() {
  lastLandmarks = null; lastFeatures = null;
  if (handAvailable) {
    handAvailable = false;
    training.stop();
    trainingStatus("손이 감지되지 않아 수집을 중단했습니다. 손을 비춘 뒤 다시 눌러주세요.");
    updateControls();
  }
  if (isModelReady) setText("status-badge", "손을 카메라에 비춰주세요");
}
function checkHandFreshness() {
  if (handAvailable && performance.now() - lastHandSeenAt > HAND_FRESH_MS) invalidateHand();
  if (isTracking && !handAvailable) {
    setText("result-label", "손 감지 안 됨");
    setText("result-conf", (handLossStatus || "0.5초 동안 손이 없으면 stop을 전송합니다.") + " 손을 비추면 인식을 다시 시작합니다.");
    if (!handLossSent && performance.now() - lastHandSeenAt > HAND_FRESH_MS) {
      handLossSent = true;
      const epoch = trackingEpoch, lossFrame = frameId;
      handLossStatus = isConnected ? "stop 전송 중…" : "stop 전송 대기 · 기기 연결 필요";
      sendStop(epoch).then(sent => {
        if (epoch !== trackingEpoch || lossFrame !== frameId || !isTracking || handAvailable) return;
        handLossStatus = sent ? "stop 전송 완료" :
          (isConnected ? "stop 전송 실패 · 연결을 확인해주세요." : "stop 전송 안 됨 · 기기 연결 필요");
      });
    }
  }
}
function draw() {
  background(0);
  push();
  if (isFlipped) { translate(width, 0); scale(-1, 1); }
  if (video && video.elt.readyState >= 2) image(video, 0, 0, width, height);
  pop();
  if (training) checkHandFreshness();
  if (lastLandmarks) drawLandmarks(lastLandmarks);
}
function addClass() {
  if (isBusy) return;
  training.stop();
  if (classIds.length >= HandModel.MAX_CLASSES || nextClassId >= Number.MAX_SAFE_INTEGER - 1) {
    trainingStatus("ID는 최대 100개까지 추가할 수 있습니다."); return;
  }
  const id = "ID" + nextClassId++;
  classIds.push(id); renderClasses(); updateControls();
}
function renderClasses() {
  const list = byId("training-list"); list.replaceChildren();
  if (!classIds.length) {
    const empty = document.createElement("div");
    empty.className = "empty-msg"; empty.textContent = "아직 학습 ID가 없습니다.";
    list.appendChild(empty); return;
  }
  const counts = HandModel.counts(trainingData);
  for (const id of classIds) {
    const row = document.createElement("div"); row.className = "list-item train-btn-row"; row.dataset.id = id;
    const button = document.createElement("button");
    button.type = "button"; button.className = "train-btn"; button.dataset.id = id;
    button.setAttribute("aria-label", id + " 학습: 짧게 누르면 1개, 길게 누르면 연속 수집");
    for (const [name, text] of [["id-badge", id], ["train-text", "학습하기"], ["badge-count train-count", (counts[id] || 0) + "개"]]) {
      const span = document.createElement("span"); span.className = name; span.textContent = text;
      button.appendChild(span);
    }
    training.bind(button, id);
    const remove = document.createElement("button");
    remove.type = "button"; remove.className = "delete-btn delete-class-btn"; remove.textContent = "×";
    remove.setAttribute("aria-label", id + " 삭제");
    remove.addEventListener("click", () => deleteClass(id));
    row.append(button, remove); list.appendChild(row);
  }
}
function collectSample(id) {
  if (isBusy || !isModelReady || !classIds.includes(id)) return false;
  if (!handAvailable || !lastFeatures || performance.now() - lastHandSeenAt > HAND_FRESH_MS) {
    trainingStatus("손을 카메라에 비춘 뒤 다시 눌러주세요."); return false;
  }
  if (isTracking) stopTracking();
  if (lastSampleFrame === frameId) return null;
  const count = HandModel.counts(trainingData)[id] || 0;
  if (count >= HandModel.MAX_PER_CLASS || trainingData.length >= HandModel.MAX_SAMPLES) {
    trainingStatus("ID당 500개, 전체 2,000개까지 학습할 수 있습니다."); return false;
  }
  trainingData.push({label: id, features: [...lastFeatures]});
  lastSampleFrame = frameId;
  const badge = document.querySelector('.train-btn[data-id="' + id + '"] .badge-count');
  if (badge) badge.textContent = (count + 1) + "개";
  trainingStatus(id + " · " + (count + 1) + "개 수집됨");
  updateControls();
  return true;
}
function stopForChange() {
  training.stop(); stopTracking();
}
function deleteClass(id) {
  if (isBusy) return;
  training.stop();
  if (!confirm(id + "와 해당 학습 데이터를 삭제할까요?")) return;
  stopForChange();
  trainingData = trainingData.filter(sample => sample.label !== id);
  classIds = classIds.filter(label => label !== id);
  renderClasses(); updateControls();
  setText("result-label", "대기 중"); setText("result-conf", "데이터 변경됨");
  trainingStatus(id + "를 삭제했습니다. 다른 ID는 유지됩니다.");
}
function clearAllModel() {
  if (isBusy) return;
  training.stop();
  if (!confirm("모든 ID와 학습 데이터를 초기화할까요?")) return;
  stopForChange();
  trainingData = []; classIds = []; nextClassId = 1; lastSampleFrame = -1;
  renderClasses(); updateControls();
  setText("result-label", "대기 중"); setText("result-conf", "데이터 없음");
  trainingStatus("초기화했습니다. ID1부터 추가할 수 있습니다.");
  fileStatus("학습한 뒤 모델을 다운로드하거나 공유하세요.");
}
function makeModelFile() {
  training.stop();
  const project = HandModel.serialize(classIds, nextClassId, trainingData, isFlipped);
  const file = new File([JSON.stringify(project)], "boundary-x-handpose-" +
    new Date().toISOString().replace(/[:.]/g, "-") + ".json", {type: "application/json"});
  if (file.size > HandModel.MAX_BYTES) throw new Error("파일이 8MiB를 초과합니다. 학습 데이터를 줄여주세요.");
  return file;
}
function downloadFile(file) {
  const url = URL.createObjectURL(file), link = document.createElement("a");
  link.href = url; link.download = file.name; link.hidden = true;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
function downloadModel() {
  if (isBusy) return;
  try { downloadFile(makeModelFile()); fileStatus("JSON 다운로드를 요청했습니다. 브라우저의 다운로드 또는 파일 앱을 확인해주세요."); }
  catch (error) { fileStatus(error.message); }
}
async function shareModel() {
  if (isBusy) return;
  try {
    const file = makeModelFile();
    if (!navigator.share || !navigator.canShare || !navigator.canShare({files: [file]})) {
      downloadFile(file);
      fileStatus("JSON 파일 공유를 지원하지 않아 다운로드했습니다. 파일을 메일 등에 첨부해주세요.");
      return;
    }
    isBusy = true; updateControls();
    await navigator.share({files: [file], title: "Boundary X 핸드 포즈 모델"});
    fileStatus("공유 앱에 파일을 전달했습니다. 최종 전송은 선택한 앱에서 확인해주세요.");
  } catch (error) {
    fileStatus(error.name === "AbortError" ? "공유를 취소했습니다."
      : "파일 공유에 실패했습니다. 모델 다운로드로 저장한 뒤 첨부해주세요.");
  } finally { isBusy = false; updateControls(); }
}
async function importModel(file) {
  if (!file || isBusy) return;
  isBusy = true; training.stop(); updateControls();
  try {
    if (file.size > HandModel.MAX_BYTES) throw new Error("8MiB 이하의 JSON 파일을 선택해주세요.");
    const project = HandModel.parse(await file.text());
    // Prepare independent state before touching the current project.
    const samples = project.samples.map(sample => ({label: sample.label, features: [...sample.features]}));
    const ids = [...project.classIds];
    if (classIds.length && !confirm("가져오면 현재 ID와 학습 데이터를 교체합니다. 계속할까요?")) {
      fileStatus("가져오기를 취소했습니다. 기존 데이터는 유지됩니다."); return;
    }
    stopForChange();
    trainingData = samples; classIds = ids; nextClassId = project.nextClassId;
    isFlipped = project.settings.isFlipped; lastSampleFrame = -1;
    renderClasses();
    setText("result-label", "모델 준비됨"); setText("result-conf", "인식 시작을 눌러주세요.");
    trainingStatus("모델을 가져왔습니다. 손을 비추고 ID별 학습을 이어갈 수 있습니다.");
    fileStatus(classIds.length + "개 ID · " + trainingData.length + "개 샘플을 가져왔습니다.");
  } catch (error) {
    fileStatus("가져오기 실패: " + error.message);
  } finally {
    byId("model-file-input").value = "";
    isBusy = false; updateControls();
  }
}
function startTracking() {
  if (isBusy || isTracking || !isModelReady || !trainingData.length) return;
  training.stop();
  trackingEpoch++; isTracking = true; handLossSent = false; handLossStatus = "";
  lastSentLabel = ""; lastSendTime = 0;
  setText("result-label", "손 감지 대기"); setText("result-conf", "");
}
function stopTracking(sendStopSignal = true) {
  const active = isTracking;
  isTracking = false;
  if (active) trackingEpoch++;
  if (active) {
    setText("result-label", "중지됨"); setText("result-conf", "");
    if (sendStopSignal) sendStop(trackingEpoch);
  }
}
function showPrediction(result) {
  if (!result || !isTracking) return;
  setText("result-label", result.label);
  setText("result-conf", "신뢰도: " + (result.confidence * 100).toFixed(0) + "%" +
    (result.neighbors < 5 ? " · 샘플 부족 (" + result.neighbors + "/5)" : ""));
  if (!isConnected) { setText("bluetooth-data-display", "전송 대기: 기기 연결 필요"); return; }
  if (!predictionSendPending &&
      (result.label !== lastSentLabel || Date.now() - lastSendTime > SEND_INTERVAL)) {
    predictionSendPending = true;
    queueSend(result.label, trackingEpoch).finally(() => { predictionSendPending = false; });
  }
}
function sendStop(epoch) {
  return queueSend("stop", epoch, 5);
}
function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error("BLE write timeout")), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
function queueSend(data, epoch, attempts = 1) {
  const target = rxCharacteristic, device = bluetoothDevice;
  const operation = sendQueue.then(async () => {
    if (data !== "stop" && (epoch !== trackingEpoch || !isTracking)) return false;
    if (!isConnected || !target || target !== rxCharacteristic) {
      if (epoch === trackingEpoch) setText("bluetooth-data-display", "전송 대기: 기기 연결 필요");
      return false;
    }
    for (let i = 0; i < attempts; i++) {
      if (!isConnected || target !== rxCharacteristic) return false;
      try {
        await withTimeout(target.writeValue(new TextEncoder().encode(data + "\n")), 2000);
        if (epoch === trackingEpoch) {
          lastSentLabel = data; lastSendTime = Date.now();
          setText("bluetooth-data-display", "전송됨: " + data);
        }
        return true;
      } catch (error) {
        if (error.message === "BLE write timeout") {
          // A timeout does not cancel the underlying write. Disconnect before allowing another.
          if (device && device.gatt.connected) device.gatt.disconnect();
          isConnected = false; rxCharacteristic = null;
          stopTracking(false);
          bluetoothStatus = "전송 시간 초과. 다시 연결해주세요."; updateBluetoothStatusUI(); updateControls();
          setText("bluetooth-data-display", "전송 실패: 다시 연결해주세요.");
          return false;
        }
        if (i + 1 < attempts) await new Promise(resolve => setTimeout(resolve, 80));
      }
    }
    if (epoch === trackingEpoch) {
      setText("bluetooth-data-display", data === "stop" ? "정지 신호 전송 실패: 연결을 확인해주세요." : "전송 실패: 연결을 확인해주세요.");
    }
    return false;
  });
  sendQueue = operation.catch(() => false);
  return sendQueue;
}
async function connectBluetooth() {
  if (isConnecting || isConnected) return;
  isConnecting = true; updateControls();
  try {
    if (!navigator.bluetooth) throw new Error("이 브라우저는 블루투스를 지원하지 않습니다.");
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      filters: [{namePrefix: "BBC micro:bit"}], optionalServices: [UART_SERVICE_UUID]
    });
    bluetoothDevice.addEventListener("gattserverdisconnected", onDisconnected);
    const server = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE_UUID);
    rxCharacteristic = await service.getCharacteristic(UART_RX_UUID);
    isConnected = true; lastSentLabel = ""; lastSendTime = 0;
    bluetoothStatus = "연결됨: " + bluetoothDevice.name;
  } catch (error) {
    bluetoothStatus = "연결 실패: " + error.message;
  } finally { isConnecting = false; updateBluetoothStatusUI(); updateControls(); }
}
function disconnectBluetooth() {
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    isManualDisconnect = true; bluetoothDevice.gatt.disconnect();
  } else onDisconnected();
}
function onDisconnected(event) {
  if (event && event.target !== bluetoothDevice) return;
  isConnected = false; rxCharacteristic = null; bluetoothDevice = null;
  const wasTracking = isTracking;
  stopTracking(false);
  bluetoothStatus = isManualDisconnect ? "연결 해제됨" : "연결이 끊어졌습니다. 다시 연결해주세요.";
  isManualDisconnect = false;
  updateBluetoothStatusUI(); updateControls();
  setText("bluetooth-data-display", wasTracking ? "연결 해제로 인식이 중지되었습니다." : "전송 대기: 기기 연결 필요");
}
function updateBluetoothStatusUI() {
  setText("bluetoothStatus", "상태: " + bluetoothStatus);
  byId("bluetoothStatus").classList.toggle("status-connected", isConnected);
}
function drawLandmarks(landmarks) {
  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17]
  ];
  stroke(0, 200, 0); strokeWeight(2);
  for (const [a, b] of connections) {
    let ax = landmarks[a].x * width, ay = landmarks[a].y * height;
    let bx = landmarks[b].x * width, by = landmarks[b].y * height;
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
