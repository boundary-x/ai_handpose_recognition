/* In-app help. Tours only explain the UI; they never operate training or Bluetooth. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const support = $('support-card');
  const example = 'https://makecode.microbit.org/57559-53483-63617-50743';
  support.innerHTML = `
    <summary><span><strong>사용 가이드 및 지원</strong><small>사용법 · 예제 코드 · 문제 해결</small></span><span class="support-chevron" aria-hidden="true">⌄</span></summary>
    <div class="support-content">
      <p class="support-intro">처음이라면 화면 안내부터 시작해보세요.</p>
      <div class="support-actions">
        <button type="button" data-tour="all" class="support-primary">사용법 둘러보기 <span aria-hidden="true">→</span></button>
      </div>
      <details class="support-section" id="help-examples"><summary>마이크로비트 예제 코드</summary><div class="support-answer example-codes">
        <div class="example-code"><a href="https://makecode.microbit.org/S49771-77509-50114-72682" target="_blank" rel="noopener noreferrer">블루투스 이름 확인 코드 ↗</a><p>연결할 마이크로비트의 장치 이름을 확인합니다. 마이크로비트의 LED 매트릭스에 출력되는 이름(알파벳 소문자 5자리)을 확인한 뒤 아래 프로젝트 코드를 다운로드하세요.</p></div>
        <div class="example-code"><a id="project-example-link" href="${example}" target="_blank" rel="noopener noreferrer">프로젝트 예제 코드 ↗</a><p><strong>조건문에 ID1을 직접 입력해 코드를 완성하세요.</strong> 다른 학습 ID도 사용하려면 해당 ID에 맞는 조건과 동작을 추가하세요.</p></div>
        <p class="support-caption">stop을 받았을 때 기기를 멈추는 동작도 설정하세요. 다운로드하면 마이크로비트에 있던 이전 코드가 교체됩니다.</p>
      </div></details>
      <details class="support-section" id="help-troubleshooting"><summary>문제 해결 <span class="support-meta">증상별 안내</span></summary><div class="support-answer support-faq">
        <details id="help-camera"><summary>카메라가 켜지지 않아요</summary><p>주소창의 사이트 권한에서 카메라 사용을 허용하세요. 다른 앱이 카메라를 사용 중이면 종료하고 다시 시도하세요. 새로고침 전에는 수집한 모델을 다운로드해두세요.</p></details>
        <details id="help-hand"><summary>손이 감지되지 않거나 학습 버튼이 비활성화돼요</summary><p>모델 로딩이 끝날 때까지 기다린 뒤 손 전체가 화면 안에 들어오도록 비춰주세요. 밝은 곳에서 손과 배경을 구분하면 도움이 됩니다. ‘손 감지됨’ 상태에서 ID별 학습 버튼을 사용할 수 있습니다.</p></details>
        <details id="help-connection"><summary>블루투스 연결이 안 되거나 기기가 움직이지 않아요</summary><p>마이크로비트 전원과 예제 코드 다운로드 여부를 확인하세요. 다른 브라우저나 앱에 연결되어 있다면 연결을 해제하세요. 기기 선택 창이 열리지 않으면 사용 중인 브라우저의 Web Bluetooth 지원 여부를 확인하세요.</p><p>앱 아래의 ‘전송됨’ 표시와 예제 코드의 수신 문자열을 비교하세요. ID1과 id1은 다릅니다. stop에 대한 정지 동작도 코드에 있어야 합니다.</p></details>
        <details id="help-accuracy"><summary>다른 손 모양으로 인식돼요</summary><p>구분하기 쉬운 손 모양부터 시작하세요. 각 ID에 비슷한 개수의 샘플을 모으고, 손 위치와 거리를 조금씩 바꾸어 수집하세요. 현재 학습 방식은 손의 회전이나 좌우 손을 자동으로 같은 동작으로 처리하지 않습니다.</p></details>
        <details id="help-files"><summary>가져오기나 파일 공유가 안 돼요</summary><p>이 핸드포즈 앱에서 다운로드한 JSON 파일을 사용하세요. 이미지 분류 앱의 모델은 가져올 수 없습니다. 파일 공유를 지원하지 않는 환경에서는 모델 다운로드를 사용하고 저장한 파일을 직접 첨부하세요.</p></details>
      </div></details>
      <details class="support-section" id="help-updates"><summary>업데이트 노트 <span class="support-meta">최근 변경</span></summary><div class="support-answer">
        <p class="support-release">사용 가이드 및 지원 추가</p><ul><li>화면 하이라이트 안내와 예제 코드 바로가기</li><li>앱 안에서 확인하는 문제 해결 및 업데이트 노트</li></ul>
        <p class="support-release">현재 버전의 주요 개선</p><ul><li>ID1부터 자동으로 추가되는 학습 ID</li><li>짧게 눌러 1개 수집, 길게 눌러 연속 수집</li><li>모델 다운로드 · 파일 공유 · 가져오기</li><li>모바일 레이아웃과 이미지 분류 앱의 디자인 통일</li><li>손이 0.5초 동안 감지되지 않으면 stop 전송</li></ul>
      </div></details>
      <a class="support-original" href="https://boundaryx.io/ai/?bmode=view&idx=169856960" target="_blank" rel="noopener noreferrer">개념 설명 · 프로젝트 아이디어 보기 ↗</a>
    </div>`;

  const tours = {
    learn: [
      ['#p5-container', '손을 카메라에 비춰주세요', '카메라 권한을 허용하고 모델 로딩을 기다리세요. 손 전체가 보이면 관절 표시와 ‘손 감지됨’ 상태가 나타납니다.'],
      ['#add-class-btn', '학습 ID를 추가하세요', '이 버튼을 누르면 ID1부터 자동으로 만들어집니다. ID마다 서로 다른 손 모양을 정해주세요.'],
      ['#training-list', '짧게 누르거나 길게 누르세요', 'ID별 ‘학습하기’를 짧게 누르면 샘플 1개, 길게 누르면 연속으로 수집합니다. 손이 감지될 때 사용할 수 있습니다.'],
      ['#add-class-btn', '다른 손 모양도 학습하세요', 'ID2를 추가해 다른 손 모양을 수집하세요. 각 ID에 비슷한 개수의 샘플을 모으면 비교하기 좋습니다.'],
      ['#recognition-control-buttons', '인식을 시작하세요', '학습한 뒤 ‘인식 시작’을 누르세요. 마이크로비트가 연결되지 않아도 화면에서 인식 결과를 확인할 수 있습니다.'],
      ['#result-card .data-info-box', '결과를 확인하세요', '손 모양을 바꾸면서 ID와 신뢰도를 확인하세요. 손이 사라지면 인식을 멈추고, 다시 비추면 자동으로 이어집니다.']
    ],
    files: [
      ['#download-model-btn', '학습한 모델을 보관하세요', '모델 다운로드로 ID와 손 좌표 학습 데이터를 JSON 파일에 저장합니다. 원본 카메라 영상은 포함되지 않습니다.'],
      ['#share-model-btn', '파일을 공유하세요', '지원하는 기기에서는 공유 창에서 앱을 선택할 수 있습니다. 공유가 지원되지 않으면 다운로드한 파일을 직접 첨부하세요.'],
      ['#import-model-btn', '이어서 학습하세요', '이 핸드포즈 앱에서 저장한 JSON을 선택하세요. 가져오면 현재 ID와 데이터가 교체되므로 필요한 모델은 먼저 저장하세요.']
    ],
    device: [
      ['#project-example-link', '프로젝트 예제를 완성하세요', '지원 카드에서 이름 확인 코드와 프로젝트 예제를 열 수 있습니다. 프로젝트 조건문에 ID1을 직접 입력하고 원하는 동작을 넣은 뒤 다운로드하세요.'],
      ['#bluetooth-control-buttons', '마이크로비트를 연결하세요', '전원을 켠 뒤 ‘기기 연결’에서 내 장치를 선택하세요. 예제의 블루투스 페어링 설정을 확인하고, 연결이 어렵다면 ‘연결이 안 되나요?’를 참고하세요.'],
      ['#recognition-control-buttons', '인식 결과를 전송하세요', '학습한 뒤 ‘인식 시작’을 누르면 인식한 ID가 전송됩니다. ‘인식 중지’를 누르면 stop을 전송합니다.'],
      ['#bluetooth-data-display', '실제 전송 상태를 확인하세요', '이 영역에서 전송 성공 여부를 확인하세요. 손이 0.5초 동안 사라져도 stop을 전송하며, 기기 코드에서 이를 처리해야 멈춥니다.']
    ]
  };
  const allSteps = [...tours.learn, ...tours.device, ...tours.files];
  const chapters = [{label:'학습', start:0}, {label:'기기 연결', start:tours.learn.length}, {label:'저장·가져오기', start:tours.learn.length + tours.device.length}];
  const dialog = document.createElement('dialog');
  dialog.id = 'guide-dialog';
  dialog.setAttribute('aria-labelledby', 'guide-title');
  dialog.setAttribute('aria-describedby', 'guide-description');
  dialog.innerHTML = `<div id="guide-spotlight" aria-hidden="true"></div><section id="guide-panel"><div class="guide-topline"><span id="guide-progress"></span><button id="guide-close" type="button" aria-label="화면 안내 종료">닫기 ×</button></div><nav class="guide-chapters" aria-label="안내 구간">${chapters.map((chapter, i) => `<button type="button" data-chapter="${i}" aria-pressed="false">${chapter.label}</button>`).join('')}</nav><div aria-live="polite" aria-atomic="true"><h2 id="guide-title"></h2><p id="guide-description"></p></div><p class="guide-caption">화면 안내입니다. 닫은 뒤 직접 눌러보세요.</p><button id="guide-skip-device" type="button" hidden>기기 연결 건너뛰기 →</button><div class="guide-navigation"><button id="guide-prev" type="button">이전</button><button id="guide-next" type="button">다음</button></div></section>`;
  document.body.appendChild(dialog);
  let steps = [], index = 0, target = null, opener = null, originalScroll = 0, pendingFrame = 0;

  let examplesWereOpen = false;

  function openHelp(section) {
    support.open = true;
    if (section) {
      $('help-troubleshooting').open = true;
      $(section).open = true;
    }
    const heading = (section ? $(section) : support).querySelector('summary');
    heading.scrollIntoView({block: 'center', behavior: 'instant'});
    heading.focus({preventScroll: true});
  }
  document.querySelectorAll('[data-help]').forEach(button => button.addEventListener('click', () => openHelp(button.dataset.help || null)));

  function renderStep() {
    const [selector, title, description] = steps[index];
    if (selector === '#project-example-link') $('help-examples').open = true;
    target = document.querySelector(selector);
    const chapterIndex = index < chapters[1].start ? 0 : index < chapters[2].start ? 1 : 2;
    dialog.querySelectorAll('[data-chapter]').forEach((button, i) => button.setAttribute('aria-pressed', String(i === chapterIndex)));
    $('guide-skip-device').hidden = chapterIndex !== 1;
    $('guide-progress').textContent = `${chapters[chapterIndex].label}${chapterIndex === 1 ? ' · 선택' : ''} · ${index + 1} / ${steps.length}`;
    $('guide-title').textContent = title;
    $('guide-description').textContent = description;
    if (selector === '#training-list' && !target.querySelector('.train-btn')) {
      $('guide-title').textContent = 'ID별 학습 버튼이 표시될 자리예요';
      $('guide-description').textContent = 'ID를 추가하면 이곳에 ‘학습하기’ 버튼이 나타납니다. 손이 감지될 때 짧게 눌러 1개, 길게 눌러 연속 수집할 수 있습니다.';
    }
    $('guide-prev').disabled = index === 0;
    $('guide-next').textContent = index === steps.length - 1 ? '안내 마치기' : '다음';
    if (target) target.scrollIntoView({block: 'center', behavior: 'instant'});
    positionGuide(true);
  }

  function positionGuide(reveal = false) {
    if (!dialog.open) return;
    const panel = $('guide-panel'), spot = $('guide-spotlight');
    const width = window.innerWidth, height = window.innerHeight, gap = 16;
    panel.style.width = Math.min(360, width - 24) + 'px';
    const ph = panel.getBoundingClientRect().height, pw = panel.getBoundingClientRect().width;
    const headerBottom = document.querySelector('header').getBoundingClientRect().bottom;
    let r = target ? target.getBoundingClientRect() : null;
    // Narrow screens reserve the lower area for the explanation. A temporary bottom
    // spacer allows the last control to scroll above it without altering saved data.
    const narrow = width < 700;
    if (reveal && r && narrow) {
      const top = Math.max(12, headerBottom + 16);
      window.scrollBy({top: r.top - top, behavior: 'instant'});
      r = target.getBoundingClientRect();
    }
    let x = width - pw - 12, y = height - ph - 12;
    if (r && !narrow) {
      const candidates = [
        [r.left - pw - gap, Math.max(12, Math.min(r.top, height - ph - 12))],
        [r.right + gap, Math.max(12, Math.min(r.top, height - ph - 12))],
        [Math.max(12, Math.min(r.left, width - pw - 12)), r.bottom + gap],
        [Math.max(12, Math.min(r.left, width - pw - 12)), r.top - ph - gap]
      ];
      const fit = candidates.find(([cx, cy]) => cx >= 12 && cy >= 12 && cx + pw <= width - 12 && cy + ph <= height - 12);
      if (fit) [x,y] = fit;
    }
    panel.style.left = x + 'px'; panel.style.top = Math.max(12, y) + 'px';
    if (r) {
      const top = Math.max(4, r.top - 5), left = Math.max(4, r.left - 5);
      const bottom = Math.min(height - 4, narrow ? y - 12 : height - 4, r.bottom + 5);
      spot.hidden = bottom <= top || r.right <= 0 || r.left >= width;
      Object.assign(spot.style, {left: left + 'px', top: top + 'px', width: Math.max(0, Math.min(width - 4, r.right + 5) - left) + 'px', height: Math.max(0, bottom - top) + 'px'});
    } else spot.hidden = true;
  }
  function startTour(kind, button) {
    if (kind !== 'all') return;
    opener = button; originalScroll = window.scrollY;
    steps = allSteps; index = 0;
    examplesWereOpen = $('help-examples').open;
    document.body.classList.add('guide-active');
    dialog.showModal();
    renderStep();
    $('guide-next').focus({preventScroll:true});
  }
  support.querySelectorAll('[data-tour]').forEach(button => button.addEventListener('click', () => startTour(button.dataset.tour, button)));
  $('guide-prev').addEventListener('click', () => { if (index > 0) { index--; renderStep(); } });
  $('guide-next').addEventListener('click', () => { if (index === steps.length - 1) dialog.close(); else { index++; renderStep(); } });
  dialog.querySelectorAll('[data-chapter]').forEach(button => button.addEventListener('click', () => { index = chapters[Number(button.dataset.chapter)].start; renderStep(); }));
  $('guide-skip-device').addEventListener('click', () => { index = chapters[2].start; renderStep(); $('guide-next').focus({preventScroll:true}); });
  $('guide-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    document.body.classList.remove('guide-active');
    $('help-examples').open = examplesWereOpen;
    window.scrollTo({top:originalScroll, behavior:'instant'});
    if (opener) opener.focus({preventScroll:true});
  });
  const reposition = () => {
    if (!dialog.open || pendingFrame) return;
    pendingFrame = requestAnimationFrame(() => { pendingFrame = 0; positionGuide(); });
  };
  window.addEventListener('resize', () => { if (dialog.open) renderStep(); });
  window.addEventListener('scroll', reposition, {passive:true});
  if (location.hash === '#support-card') requestAnimationFrame(() => openHelp());
})();

