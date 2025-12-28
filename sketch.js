/**
 * sketch.js
 * Boundary X: AI Handpose (Corrected Drawing & Sizing)
 * Features: Skeleton Drawing Fix, 4:3 Ratio, Robust Recognition
 */

// Bluetooth UUIDs
const UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const UART_TX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const UART_RX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

let bluetoothDevice = null;
let rxCharacteristic = null;
let txCharacteristic = null;
let isConnected = false;
let bluetoothStatus = "연결 대기 중";
let isSendingData = false;

// ML Variables
let video;
let handpose;
let knnClassifier;
let predictions = []; 
let isModelReady = false;
let isPredicting = false; 

// ID Mapping
let nextClassId = 1; 
let idToNameMap = {}; 

// DOM Elements
let classInput, addClassBtn, classListContainer, resetBtn;
let resultLabel, resultConfidence, btDataDisplay;
let cameraResultBadge; 
let flipButton, switchCameraButton, connectBluetoothButton, disconnectBluetoothButton;
let startRecognitionButton, stopRecognitionButton; 
let canvas;

// Camera
let facingMode = "user";
let isFlipped = true; // 기본적으로 거울 모드
let isVideoLoaded = false;

function setup() {
  // [설정] 4:3 비율 캔버스 생성 (640x480)
  canvas = createCanvas(640, 480);
  canvas.parent('p5-container');
  
  // Handpose 초기화
  console.log("Loading Handpose model...");
  handpose = ml5.handpose(video, modelReady);
  
  // Handpose 감지 이벤트
  handpose.on("predict", results => {
    predictions = results;
  });

  // KNN 초기화
  knnClassifier = ml5.KNNClassifier();

  setupCamera();
  createUI();
}

function modelReady() {
  console.log("Handpose Model Loaded!");
  isModelReady = true;
  if(cameraResultBadge) cameraResultBadge.html("모델 로드 완료");
}

function setupCamera() {
  let constraints = {
    video: {
      facingMode: facingMode,
      width: 640,
      height: 480
    },
    audio: false
  };
  video = createCapture(constraints);
  video.size(640, 480);
  video.hide();

  let videoLoadCheck = setInterval(() => {
    if (video.elt.readyState >= 2 && video.width > 0) {
      isVideoLoaded = true;
      clearInterval(videoLoadCheck);
      console.log(`Video Stream Ready: ${video.width}x${video.height}`);
    }
  }, 100);
}

function stopVideo() {
    if (video) {
        if (video.elt.srcObject) {
            video.elt.srcObject.getTracks().forEach(track => track.stop());
        }
        video.remove();
        video = null;
    }
}

// === [핵심] 손 그리기 (좌표 변환 포함) ===
function draw() {
  background(0);

  if (!isVideoLoaded) {
      fill(255);
      textAlign(CENTER);
      text("카메라 로딩 중...", width/2, height/2);
      return;
  }

  push(); // 변환 시작
  
  if (isFlipped) {
    translate(width, 0);
    scale(-1, 1);
  }
  
  // 1. 비디오 그리기
  image(video, 0, 0, width, height);
  
  // 2. [수정] 스켈레톤 그리기를 push/pop 안으로 이동
  // 이제 좌표계가 반전된 상태에서 그려지므로 영상과 일치합니다.
  drawKeypoints();
  drawSkeleton();
  
  pop(); // 변환 종료
}

function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0); // 초록색 점
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
  }
}

function drawSkeleton() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    const annotations = prediction.annotations;
    stroke(0, 255, 0); // 초록색 선
    strokeWeight(3);

    const parts = [
        annotations.thumb, annotations.indexFinger, 
        annotations.middleFinger, annotations.ringFinger, annotations.pinky
    ];

    parts.forEach(part => {
        for (let j = 0; j < part.length - 1; j++) {
             line(part[j][0], part[j][1], part[j+1][0], part[j+1][1]);
        }
    });
  }
}

// === [핵심 알고리즘] 손 데이터 정규화 ===
function getStandardizedLandmarks(landmarks) {
    // 1. 기준점: 손목(0번)
    const wrist = landmarks[0];
    const wristX = wrist[0];
    const wristY = wrist[1];
    
    // 2. 기준 벡터: 손목(0) -> 중지 뿌리(9)
    const middleBase = landmarks[9];
    const baseDeltaX = middleBase[0] - wristX;
    const baseDeltaY = middleBase[1] - wristY;
    
    // 크기(Scale): 손목~중지뿌리 거리
    const scale = Math.sqrt(baseDeltaX * baseDeltaX + baseDeltaY * baseDeltaY);
    
    // 각도(Angle): 수직 정렬을 위한 회전 각도
    const angle = Math.atan2(baseDeltaY, baseDeltaX);
    const rotationAngle = -Math.PI / 2 - angle; 
    
    const cos = Math.cos(rotationAngle);
    const sin = Math.sin(rotationAngle);

    let standardizedFeatures = [];

    // 3. 변환 수행
    for (let i = 0; i < landmarks.length; i++) {
        // (1) 평행이동
        let dx = landmarks[i][0] - wristX;
        let dy = landmarks[i][1] - wristY;
        
        // (2) 크기 정규화
        dx /= scale;
        dy /= scale;
        
        // (3) 회전 변환
        let nx = dx * cos - dy * sin;
        let ny = dx * sin + dy * cos;
        
        let nz = (landmarks[i][2] || 0) / scale;

        standardizedFeatures.push(nx, ny, nz);
    }

    return standardizedFeatures;
}

// === UI 생성 및 이벤트 ===
function createUI() {
  classInput = select('#class-input');
  addClassBtn = select('#add-class-btn');
  classListContainer = select('#class-list');
  resetBtn = select('#reset-model-btn');
  
  resultLabel = select('#result-label');
  resultConfidence = select('#result-confidence');
  btDataDisplay = select('#bluetooth-data-display');

  cameraResultBadge = select('#camera-result-badge');

  addClassBtn.mousePressed(addNewClass);
  classInput.elt.addEventListener("keypress", (e) => {
      if (e.key === "Enter") addNewClass();
  });
  
  resetBtn.mousePressed(resetModel);

  flipButton = createButton("좌우 반전");
  flipButton.parent('camera-control-buttons');
  flipButton.addClass('start-button');
  flipButton.mousePressed(() => isFlipped = !isFlipped);

  switchCameraButton = createButton("전후방 전환");
  switchCameraButton.parent('camera-control-buttons');
  switchCameraButton.addClass('start-button');
  switchCameraButton.mousePressed(switchCamera);

  connectBluetoothButton = createButton("기기 연결");
  connectBluetoothButton.parent('bluetooth-control-buttons');
  connectBluetoothButton.addClass('start-button');
  connectBluetoothButton.mousePressed(connectBluetooth);

  disconnectBluetoothButton = createButton("연결 해제");
  disconnectBluetoothButton.parent('bluetooth-control-buttons');
  disconnectBluetoothButton.addClass('stop-button');
  disconnectBluetoothButton.mousePressed(disconnectBluetooth);

  startRecognitionButton = createButton("인식 시작");
  startRecognitionButton.parent('recognition-control-buttons');
  startRecognitionButton.addClass('start-button');
  startRecognitionButton.mousePressed(startClassify);

  stopRecognitionButton = createButton("인식 중지");
  stopRecognitionButton.parent('recognition-control-buttons');
  stopRecognitionButton.addClass('stop-button');
  stopRecognitionButton.mousePressed(stopClassify);

  updateBluetoothStatusUI();
}

function switchCamera() {
  stopVideo();
  isVideoLoaded = false;
  facingMode = facingMode === "user" ? "environment" : "user";
  // 후면 카메라는 반전 안 함
  isFlipped = (facingMode === "user"); 
  setTimeout(setupCamera, 500);
}

// === 로직: 데이터 학습 ===

function addNewClass() {
    const className = classInput.value().trim();
    if (className === "") {
        alert("이름을 입력해주세요.");
        return;
    }

    const currentId = String(nextClassId++);
    idToNameMap[currentId] = className; 

    const row = createDiv('');
    row.addClass('train-btn-row');
    row.parent(classListContainer);

    const trainBtn = createButton(
        `<span class="id-badge">ID ${currentId}</span>
         <span class="train-text">${className}</span>`
    );
    trainBtn.addClass('train-btn');
    trainBtn.parent(row);
    
    const countBadge = createSpan('0 data');
    countBadge.addClass('train-count');
    countBadge.parent(trainBtn);

    trainBtn.mousePressed(() => {
        addExample(currentId); 
        trainBtn.style('background', '#e0e0e0');
        setTimeout(() => trainBtn.style('background', '#f8f9fa'), 100);
    });

    const delBtn = createButton('×');
    delBtn.addClass('delete-class-btn');
    delBtn.parent(row);
    delBtn.mousePressed(() => {
        if(confirm(`[ID ${currentId}: ${className}] 클래스를 삭제하시겠습니까?`)) {
            knnClassifier.clearLabel(currentId);
            row.remove();
        }
    });

    classInput.value('');
}

function addExample(labelId) {
    if (!isModelReady || !isVideoLoaded) {
      alert("모델 로딩 중입니다. 잠시만 기다려주세요.");
      return;
    }
    
    if (predictions.length > 0) {
        const landmarks = predictions[0].landmarks;
        const features = getStandardizedLandmarks(landmarks);
        
        knnClassifier.addExample(features, labelId);
        updateButtonCount(labelId);
        console.log(`학습됨 (ID ${labelId}): 정규화 완료`);
    } else {
        alert("손이 감지되지 않았습니다. 화면에 손을 비춰주세요.");
    }
}

function updateButtonCount(labelId) {
    const count = knnClassifier.getCountByLabel()[labelId];
    const buttons = document.querySelectorAll('.train-btn');
    buttons.forEach(btn => {
        if (btn.innerText.includes(`ID ${labelId}`)) {
            const badge = btn.querySelector('.train-count');
            if(badge) badge.innerText = `${count} data`;
        }
    });
}

function resetModel() {
    if(confirm("모든 학습 데이터를 삭제하시겠습니까?")) {
        knnClassifier.clearAllLabels();
        idToNameMap = {};
        nextClassId = 1;
        classListContainer.html(''); 
        resultLabel.html("데이터 없음");
        resultConfidence.html("");
        btDataDisplay.html("전송 데이터: 대기 중...");
        btDataDisplay.style('color', '#666');
        
        stopClassify(); 
    }
}

// === 로직: 분류 및 전송 ===

function startClassify() {
    if (knnClassifier.getNumLabels() <= 0) {
        alert("먼저 학습 데이터를 추가해주세요!");
        return;
    }
    if (!isPredicting) {
        isPredicting = true;
        cameraResultBadge.style('display', 'block');
        cameraResultBadge.html('인식 중...');
        classify(); 
    }
}

function stopClassify() {
    isPredicting = false;
    resultLabel.html("중지됨");
    resultLabel.style('color', '#666');
    resultConfidence.html("");
    
    sendBluetoothData("stop");
    btDataDisplay.html("전송됨: stop");
    btDataDisplay.style('color', '#EA4335');

    if(cameraResultBadge) {
        cameraResultBadge.style('display', 'none');
        cameraResultBadge.removeClass('high-confidence');
    }
}

function classify() {
    if (!isPredicting) return;
    
    if (predictions.length > 0) {
         const landmarks = predictions[0].landmarks;
         const features = getStandardizedLandmarks(landmarks);
         knnClassifier.classify(features, gotResults);
    } else {
        requestAnimationFrame(classify);
    }
}

function gotResults(error, result) {
    if (error) {
        console.error(error);
        return;
    }

    if (result.confidencesByLabel) {
        const labelId = result.label;
        const confidence = result.confidencesByLabel[labelId] * 100;
        const name = idToNameMap[labelId] || "알 수 없음";

        resultLabel.html(`ID ${labelId} (${name})`);
        resultLabel.style('color', '#000');
        resultConfidence.html(`정확도: ${confidence.toFixed(0)}%`);

        const badgeText = `ID ${labelId} (${name}) | ${confidence.toFixed(0)}%`;
        cameraResultBadge.html(badgeText);
        
        if (confidence > 60) {
            cameraResultBadge.addClass('high-confidence');
            let dataToSend = `ID${labelId}`;
            sendBluetoothData(dataToSend);
            btDataDisplay.html(`전송됨: ${dataToSend}`);
            btDataDisplay.style('color', '#0f0');
        } else {
            cameraResultBadge.removeClass('high-confidence');
        }
    }

    if (isPredicting) {
        requestAnimationFrame(classify); 
    }
}

/* --- Bluetooth Logic (기존 동일) --- */
async function connectBluetooth() {
  try {
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: "BBC micro:bit" }],
      optionalServices: [UART_SERVICE_UUID]
    });
    const server = await bluetoothDevice.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE_UUID);
    rxCharacteristic = await service.getCharacteristic(UART_RX_CHARACTERISTIC_UUID);
    txCharacteristic = await service.getCharacteristic(UART_TX_CHARACTERISTIC_UUID);
    isConnected = true;
    bluetoothStatus = "연결됨: " + bluetoothDevice.name;
    updateBluetoothStatusUI(true);
  } catch (error) {
    console.error("Connection failed", error);
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
  txCharacteristic = null;
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
    console.error("Error sending data:", error);
  } finally {
    isSendingData = false;
  }
}
