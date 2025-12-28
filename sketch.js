/**
 * sketch.js
 * Boundary X: AI Gesture Learning [Resolution Fixed]
 * Fix: Mapped coordinates from Video source to Canvas size
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
let classes = {}; 
let isTraining = false; 

function setup() {
  let canvas = createCanvas(320, 240);
  canvas.parent('p5-container');

  // 1. 비디오 설정
  video = createCapture(VIDEO);
  video.size(320, 240); // 요청 사이즈
  video.hide();

  // 2. Handpose 설정
  handpose = ml5.handpose(video, modelReady);
  handpose.on("predict", results => {
    predictions = results;
  });

  // 3. KNN 설정
  knnClassifier = ml5.KNNClassifier();

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

  addDataBtn.mousePressed(() => isTraining = true);
  addDataBtn.mouseReleased(() => isTraining = false);
  resetBtn.mousePressed(clearModel);
}

// === 메인 루프 ===
function draw() {
  background(0);

  // [핵심 수정] 좌표계 변환 그룹 시작
  push();
  // 1. 캔버스 전체를 거울처럼 좌우 반전시킵니다.
  translate(width, 0);
  scale(-1, 1);

  // 2. 비디오 그리기 (이제 반전된 상태로 그려짐)
  // 비디오가 로드된 상태일 때만 그립니다.
  if (video.elt.readyState >= 2) {
      image(video, 0, 0, width, height);
      
      // 3. 스켈레톤 그리기 (같은 반전된 좌표계 위에서 그림)
      drawKeypoints();
  }
  pop(); // 좌표계 변환 그룹 끝

  // 4. 데이터 처리 및 학습/분류
  if (predictions.length > 0) {
    let hand = predictions[0];
    
    // 손목 기준 상대 좌표 추출
    let features = extractRelativeFeatures(hand);
    
    rawDataDisplay.html(`Features[40]: [${features.slice(0,3).map(n=>n.toFixed(0))}...]`);

    if (isTraining) {
      let label = classInput.value().trim();
      if (label) addExample(features, label);
    } 
    else if (knnClassifier.getNumLabels() > 0) {
      classify(features);
    }
  } else {
    rawDataDisplay.html("손을 보여주세요.");
  }
}

/**
 * 손목 기준 상대 좌표 변환 함수
 */
function extractRelativeFeatures(hand) {
  let features = [];
  let landmarks = hand.landmarks; 
  let wrist = landmarks[0]; 

  for (let i = 1; i < landmarks.length; i++) {
    let x = landmarks[i][0];
    let y = landmarks[i][1];
    
    let relativeX = x - wrist[0];
    let relativeY = y - wrist[1];
    
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
    if (err) { console.error(err); return; }
    
    if (result.confidencesByLabel) {
      const label = result.label;
      const conf = result.confidencesByLabel[label];
      
      resultLabel.html(label);
      resultConf.html(`정확도: ${(conf * 100).toFixed(0)}%`);
      
      if (conf > 0.85) resultLabel.style('color', '#00E676');
      else resultLabel.style('color', '#FFEB3B');
    }
  });
}

// [핵심 수정] 스켈레톤 그리기 (좌표 매핑 적용)
function drawKeypoints() {
  // 비디오 원본 크기 가져오기 (없으면 기본값)
  let videoWidth = video.elt.videoWidth || 640;
  let videoHeight = video.elt.videoHeight || 480;

  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      
      // [중요] 원본 비디오 좌표(keypoint)를 캔버스 크기(width, height)로 비율 변환(Map)
      let x = map(keypoint[0], 0, videoWidth, 0, width);
      let y = map(keypoint[1], 0, videoHeight, 0, height);
      
      if (j === 0) fill(255, 0, 0); 
      else fill(0, 255, 0);         
      
      noStroke();
      // 이미 위에서 전체 화면을 반전(scale(-1,1))했으므로
      // 여기서는 그냥 변환된 x, y를 그대로 찍으면 위치가 일치합니다.
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
    let div = createDiv(`
      <span>${label}</span>
      <span class="badge-count">${classes[label]} data</span>
    `);
    div.addClass('list-item');
    div.parent(trainingList);
  }
}

function clearModel() {
  knnClassifier.clearAllLabels();
  classes = {};
  updateListUI();
  resultLabel.html("대기 중");
  resultLabel.style('color', '#00E676');
  resultConf.html("데이터 없음");
}
