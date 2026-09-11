/* Pure feature extraction, KNN, and portable hand-pose project files. */
(function(root) {
  "use strict";
  const FORMAT = "boundary-x-handpose-knn";
  const MAX_CLASSES = 100, MAX_PER_CLASS = 500, MAX_SAMPLES = 2000;
  const MAX_BYTES = 8 * 1024 * 1024;
  const ENGINE = Object.freeze({
    library: "@mediapipe/tasks-vision", version: "0.10.8",
    model: "hand_landmarker/float16/1", numHands: 1,
    features: 40, preprocessing: "wrist-relative-xy-max-radius-v1",
    distance: "euclidean-squared", k: 5,
    handedness: "not-canonicalized", mirror: "display-only"
  });
  function check(value, message) { if (!value) throw new Error(message); }
  function extractFeatures(points) {
    if (!Array.isArray(points) || points.length !== 21 ||
        points.some(p => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y))) return null;
    const wrist = points[0];
    let radius = 0;
    const values = [];
    for (let i = 1; i < points.length; i++) {
      const x = points[i].x - wrist.x, y = points[i].y - wrist.y;
      radius = Math.max(radius, Math.hypot(x, y));
      values.push(x, y);
    }
    if (!Number.isFinite(radius) || radius < 0.001) return null;
    return values.map(value => value / radius);
  }
  function validFeatures(values) {
    if (!Array.isArray(values) || values.length !== 40 ||
        values.some(v => typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > 1.00001)) return false;
    let radius = 0;
    for (let i = 0; i < 40; i += 2) radius = Math.max(radius, Math.hypot(values[i], values[i + 1]));
    return Math.abs(radius - 1) < 0.0001;
  }
  function classify(data, query) {
    if (!data.length || !validFeatures(query)) return null;
    const nearest = data.map(sample => ({
      label: sample.label,
      distance: sample.features.reduce((sum, value, i) => sum + (value - query[i]) ** 2, 0)
    })).sort((a,b) => a.distance - b.distance).slice(0, ENGINE.k);
    const votes = new Map();
    for (const sample of nearest) {
      const vote = votes.get(sample.label) || {label: sample.label, count: 0, distance: 0};
      vote.count++; vote.distance += sample.distance; votes.set(sample.label, vote);
    }
    // Deterministic ties: vote count, summed distance, then numeric ID.
    const best = [...votes.values()].sort((a,b) => b.count-a.count || a.distance-b.distance ||
      Number(a.label.slice(2))-Number(b.label.slice(2)))[0];
    return {label: best.label, confidence: best.count / nearest.length, neighbors: nearest.length};
  }
  function counts(data) {
    const result = Object.create(null);
    for (const sample of data) result[sample.label] = (result[sample.label] || 0) + 1;
    return result;
  }
  function validate(project) {
    check(project && project.format === FORMAT && project.version === 1,
      "이 핸드 포즈 앱에서 내보낸 모델 JSON(버전 1)을 선택해주세요.");
    check(project.engine && Object.keys(ENGINE).every(key => project.engine[key] === ENGINE[key]),
      "손 특징 추출 방식이나 모델 버전이 현재 앱과 다릅니다.");
    check(project.settings && typeof project.settings.isFlipped === "boolean", "화면 반전 설정이 올바르지 않습니다.");
    check(Array.isArray(project.classIds) && project.classIds.length > 0 && project.classIds.length <= MAX_CLASSES,
      "ID 목록이 올바르지 않습니다.");
    const ids = new Set();
    let highest = 0;
    for (const id of project.classIds) {
      check(typeof id === "string" && /^ID[1-9]\d*$/.test(id) &&
        Number.isSafeInteger(Number(id.slice(2))) && !ids.has(id), "잘못되었거나 중복된 ID가 있습니다.");
      highest = Math.max(highest, Number(id.slice(2))); ids.add(id);
    }
    check(Number.isSafeInteger(project.nextClassId) && project.nextClassId > highest &&
      project.nextClassId < Number.MAX_SAFE_INTEGER, "다음 ID 번호가 올바르지 않습니다.");
    check(Array.isArray(project.samples) && project.samples.length > 0 &&
      project.samples.length <= MAX_SAMPLES, "학습 데이터는 1~2,000개여야 합니다.");
    const totals = new Map();
    for (const sample of project.samples) {
      check(sample && ids.has(sample.label), "존재하지 않는 ID의 학습 데이터가 있습니다.");
      check(validFeatures(sample.features), "손 좌표 데이터가 손상되었거나 40차원 특징이 아닙니다.");
      const total = (totals.get(sample.label) || 0) + 1;
      check(total <= MAX_PER_CLASS, "ID당 학습 데이터는 500개까지 지원합니다.");
      totals.set(sample.label, total);
    }
    return project;
  }
  function serialize(classIds, nextClassId, samples, isFlipped) {
    return validate({
      format: FORMAT, version: 1, createdAt: new Date().toISOString(),
      engine: {...ENGINE}, settings: {isFlipped}, classIds: [...classIds], nextClassId,
      samples: samples.map(sample => ({label: sample.label, features: [...sample.features]}))
    });
  }
  function parse(text) {
    let project;
    try { project = JSON.parse(text); }
    catch (_) { throw new Error("JSON 파일을 읽을 수 없습니다. 파일이 손상되었는지 확인해주세요."); }
    return validate(project);
  }
  const api = {ENGINE, MAX_CLASSES, MAX_PER_CLASS, MAX_SAMPLES, MAX_BYTES, extractFeatures,
    validFeatures, classify, counts, validate, serialize, parse};
  root.HandModel = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);

