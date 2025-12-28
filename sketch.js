/**
 * sketch.js
 * Boundary X: AI Gesture Learning [Wrist-Centric Logic]
 * * Algorithm:
 * 1. Detect Hand Keypoints (21 points)
 * 2. Set Wrist(0) as Origin (0,0)
 * 3. Calculate Relative Coordinates for other 20 points
 * 4. Train/Predict using KNN
 */

let video;
let handpose;
let knnClassifier;
let predictions = [];
let isModelReady = false;

// UI Variables
let classInput, addDataBtn, resetBtn;
let resultLabel, resultConf, rawDataDisplay;
let trainingList;
let statusBadge;

// Data Management
let classes = {}; // { "LabelName": count }
let isTraining = false; // "Hold to train" status

function setup() {
  let canvas = createCanvas(320, 240);
  canvas.parent('p5-container');

  // 1. Handpose 설정
  video = createCapture(VIDEO);
  video.size(320, 240);
  video.hide();

  handpose = ml5.handpose(video, modelReady);
  handpose.on("predict", results => {
    predictions = results;
  });

  // 2. KNN 설정
  knnClassifier = ml5.KNNClassifier();

  // 3. UI 설정
  setupUI();
}

function modelReady() {
  console.log("Handpose Model Ready");
  isModelReady = true;
  if(statusBadge) {
      statusBadge.html("준비 완료! 제스처를 학습시키세요.");
      statusBadge.style('background', 'rgba(0,0,0,0.7)');
  }
}

function setupUI() {
  statusBadge = select('#status-badge');
  classInput = select('#class-input');
  addDataBtn = select('#add-data-btn');
  resetBtn = select('#reset-btn');
  trainingList = select('#training-list');
  
  resultLabel = select('#result-label');
  resultConf = select('#result-conf');
  rawDataDisplay = select('#raw-data-display');

  // 버튼 누르고 있을 때 계속 학습 (Data Collection Loop)
  addDataBtn.mousePressed(() => isTraining = true);
  addDataBtn.mouseReleased(() => isTraining = false);
  
  resetBtn.mousePressed(clearModel);
}

// === 메인 루프 ===
function draw() {
  background(0);

  // 1. 비디오 & 스켈레톤 그리기
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
  drawKeypoints();
  pop();

  // 2. 데이터 처리 및 학습/분류
  if (predictions.length > 0) {
    let hand = predictions[0];
    
    // [핵심] 손목 기준 상대 좌표 추출
    let features = extractRelativeFeatures(hand);
    
    // 디버깅: 데이터 값 확인 (앞 3개만 표시)
    rawDataDisplay.html(`Features[40]: [${features.slice(0,3).map(n=>n.toFixed(0))}...]`);

    // A. 학습 모드 (버튼 누르는 중)
    if (isTraining) {
      let label = classInput.value().trim();
      if (label) {
        addExample(features, label);
      }
    } 
    // B. 분류 모드 (학습 데이터가 있고 학습중이 아닐 때)
    else if (knnClassifier.getNumLabels() > 0) {
      classify(features);
    }
  } else {
    rawDataDisplay.html("손을 보여주세요.");
  }
}

/**
 * [핵심 알고리즘] 손목 기준 상대 좌표 변환 함수
 * 모든 관절 좌표에서 손목(Wrist) 좌표를 뺍니다.
 * 결과: 손의 절대 위치와 상관없이 '손 모양'만 데이터로 남음.
 */
function extractRelativeFeatures(hand) {
  let features = [];
  let landmarks = hand.landmarks; // [x, y, z] 배열 21개
  let wrist = landmarks[0]; // 0번 인덱스가 손목 (기준점)

  // 손목(0번)을 제외한 나머지 20개 점(1~20번)만 사용
  for (let i = 1; i < landmarks.length; i++) {
    let x = landmarks[i][0];
    let y = landmarks[i][1];
    // z축은 웹캠 환경에서 노이즈가 심하므로 x, y만 사용하여 2D 형상 인식에 집중
    
    // 상대 좌표 계산 (Relative Coordinates)
    let relativeX = x - wrist[0];
    let relativeY = y - wrist[1];
    
    features.push(relativeX);
    features.push(relativeY);
  }
  
  return features; // 총 40개 데이터 (20개 점 * 2축)
}

// 학습 데이터 추가
function addExample(features, label) {
  knnClassifier.addExample(features, label);
  
  // 카운트 업데이트
  if (!classes[label]) classes[label] = 0;
  classes[label]++;
  
  updateListUI();
}

// 분류 실행
function classify(features) {
  knnClassifier.classify(features, (err, result) => {
    if (err) {
      console.error(err);
      return;
    }
    
    if (result.confidencesByLabel) {
      const label = result.label;
      const conf = result.confidencesByLabel[label];
      
      resultLabel.html(label);
      resultConf.html(`정확도: ${(conf * 100).toFixed(0)}%`);
      
      // 글자색 변경 (인식률 시각화)
      if (conf > 0.85) {
          resultLabel.style('color', '#00E676'); // Green (Good)
      } else {
          resultLabel.style('color', '#FFEB3B'); // Yellow (Weak)
      }
    }
  });
}

// 스켈레톤 그리기
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    
    // 관절 점 찍기
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      
      if (j === 0) fill(255, 0, 0); // 손목은 빨간색
      else fill(0, 255, 0);         // 나머지는 초록색
      
      noStroke();
      ellipse(keypoint[0], keypoint[1], 8, 8);
    }
  }
}

// 리스트 UI 업데이트
function updateListUI() {
  trainingList.html("");
  if (Object.keys(classes).length === 0) {
      trainingList.html('<div class="empty-msg">아직 학습된 데이터가 없습니다.</div>');
      return;
  }

  for (let label in classes) {
    let div = createDiv(`
      <span>${label}</span>
      <span class="badge-count">${classes[label]} data</span>
    `);
    div.addClass('list-item');
    div.parent(trainingList);
  }
}

// 모델 초기화
function clearModel() {
  knnClassifier.clearAllLabels();
  classes = {};
  updateListUI();
  resultLabel.html("대기 중");
  resultLabel.style('color', '#00E676');
  resultConf.html("데이터 없음");
}
