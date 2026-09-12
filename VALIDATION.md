# Validation Notes

Date: 2026-09-12

## Implementation

- Retained MediaPipe Tasks Vision 0.10.8, p5.js 1.6.0, the one-hand detector, the custom KNN, k = 5, and the 100 ms transmission interval.
- Added automatic IDs, per-ID tap/hold collection, JSON download, OS file sharing with a download fallback, and validated project import.
- Added a hand-pose-specific file format with 40-dimensional features and preprocessing metadata.
- Preserved empty IDs, the next ID, the display mirror setting, and sample order during restoration.
- Added fresh-detection checks and blocked repeated collection of the same detection result.
- Cleared stale landmarks and stopped acquisition when the hand disappears or the camera stalls.
- During active tracking, sent one stop after 500 ms without a fresh hand; a returning hand resumes tracking.
- Corrected confidence when fewer than five neighbors exist and made vote ties deterministic.
- Serialized UART writes, preserved stop retries, and updated successful-send state only after successful writes.
- Adjusted the header, back button, training rows, and sticky offsets while preserving the 4:3 camera view.
- Added an English README distinguishing web users from developers who run Node.js tests.
- Removed the redundant p5.dom addon reference; the retained p5.js 1.6.0 supplies its DOM functionality.

## Automated Checks

### Core: 8 passing tests

Run with `npm run test:core`.

1. Forty-dimensional features remain invariant to translation and uniform scaling.
2. Invalid and degenerate landmarks cannot be collected.
3. KNN confidence uses available neighbors; ties are deterministic.
4. JSON round trips preserve predictions, empty classes, and next ID.
5. Image KNN files and incompatible engine/preprocessing versions are rejected.
6. Invalid IDs, references, dimensions, and coordinate values are rejected.
7. Per-class import limits are enforced.
8. Serialization does not retain aliases to live feature arrays.

### Browser: 20 passing checks

Run with `npm run test:browser`.

Tested in headless Microsoft Edge on Windows, using Playwright and a simulated camera.
The actual MediaPipe library, WASM/GPU detector, and hand model initialized successfully.
After initialization, deterministic landmark fixtures replaced the detector's output for interaction tests.

1. Actual MediaPipe and camera initialization.
2. Disabled collection with no detected hand.
3. One sample per mouse tap.
4. Keyboard collection.
5. Hold repetition and release.
6. Touch tap and touch hold without duplicate mouse events.
7. Pointer cancellation.
8. One collection per detection frame.
9. Frozen-camera freshness expiry.
10. Hand-loss display and a single stop signal.
11. Resuming after hand return and explicit stop behavior.
12. Automatic IDs and deletion gaps.
13. Download/reset/import prediction and sample-count equivalence.
14. Invalid or incompatible files leave existing data unchanged.
15. Import cancellation retains the current project.
16. File-sharing fallback, active user gesture, and cancellation.
17. Failed Bluetooth writes do not update successful-send state.
18. Stop follows an in-flight prediction write.
19. No horizontal overflow, back-button wrapping, or training-label wrapping at widths 320, 360, 390, 430, 768, 844, and 1280; preserved 4:3 camera geometry.
20. No uncaught browser errors.

Screenshots at 390, 844, and 1280 pixels were visually inspected.

### Additional error-path checks

- A stop write succeeded on the third attempt after two simulated failures.
- An unresolved write timed out, disconnected the simulated link, and stopped tracking.
- Blocking the model download produced a load-failure message, disabled recognition, and kept file import available.

## Limitations

- Real gesture-recognition accuracy was not measured. Landmark fixtures test application behavior, not detector accuracy.
- Physical smartphone sharing sheets and email attachment delivery were not exercised. The sharing boundary was mocked.
- No physical micro:bit was connected. UART order, errors, retries, and timeout behavior used simulated characteristics.
- Long-duration performance on every mobile device was not tested.
- Left/right hand equivalence and rotation invariance are not provided by the current feature definition.

## Shared design verification

The image KNN app's control styles are reused for buttons, training rows, cards, header, and responsive spacing. The hand camera retains its 4:3 aspect ratio. After the visual update, all 8 core tests and 20 browser checks passed again; mobile and desktop screenshots were inspected.

