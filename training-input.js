/* One tap = one sample; hold = repeated samples. No synthetic-click duplicates. */
(function (root) {
  "use strict";
  function createController({ collect, onStart = () => {}, onStop = () => {}, delay = 350, interval = 200 }) {
    let active = null;
    function stop() {
      if (!active) return;
      const old = active;
      active = null;
      clearTimeout(old.timer);
      old.button.classList.remove("is-training");
      if (old.button.hasPointerCapture(old.pointerId)) old.button.releasePointerCapture(old.pointerId);
      onStop();
    }
    function repeat(state) {
      if (active !== state) return;
      state.holding = true;
      state.button.classList.add("is-training");
      // null means no newer hand result yet: skip this tick without ending the hold.
      if (collect(state.id) === false) { stop(); return; }
      state.timer = setTimeout(() => repeat(state), interval);
    }
    function bind(button, id) {
      button.addEventListener("pointerdown", event => {
        if (button.disabled || active || event.isPrimary === false || event.button !== 0) return;
        const state = { button, id, pointerId: event.pointerId, holding: false, x: event.clientX, y: event.clientY };
        active = state;
        button.setPointerCapture(event.pointerId);
        onStart();
        state.timer = setTimeout(() => repeat(state), delay);
      });
      button.addEventListener("pointermove", event => {
        if (!active || active.button !== button || active.pointerId !== event.pointerId) return;
        const rect = button.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right ||
            event.clientY < rect.top || event.clientY > rect.bottom ||
            (!active.holding && Math.hypot(event.clientX - active.x, event.clientY - active.y) > 12)) stop();
      });
      button.addEventListener("pointerup", event => {
        if (!active || active.button !== button || active.pointerId !== event.pointerId) return;
        const tap = !active.holding;
        stop();
        if (tap) collect(id);
      });
      for (const name of ["pointercancel", "lostpointercapture"]) {
        button.addEventListener(name, () => { if (active && active.button === button) stop(); });
      }
      button.addEventListener("click", event => {
        // Keyboard/assistive activation only; pointer taps were handled on pointerup.
        if (event.detail === 0 && !button.disabled && !active) collect(id);
      });
      button.addEventListener("contextmenu", event => event.preventDefault());
    }
    return { bind, stop };
  }
  root.TrainingInput = { createController };
  if (typeof module !== "undefined" && module.exports) module.exports = root.TrainingInput;
})(typeof globalThis !== "undefined" ? globalThis : window);
