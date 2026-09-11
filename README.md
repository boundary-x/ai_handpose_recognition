# 👋 Boundary X - AI Hand Pose Recognition (KNN)

**Boundary X - AI Hand Pose Recognition** is a web-based application that lets users train and recognize hand gestures directly in the browser using **MediaPipe HandLandmarker** and a **custom K-Nearest Neighbors (KNN) classifier**.

It connects to **BBC Micro:bit** through the **Web Bluetooth API** to control hardware using recognized gesture IDs.

**No Node.js installation is required to use the published web app.** Open the HTTPS site in a compatible browser and allow camera access. Internet access is needed to load the libraries and hand-detection model. Bluetooth control requires a browser that supports Web Bluetooth.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-Web-blue)
![Tech](https://img.shields.io/badge/Stack-p5.js%20%7C%20MediaPipe%20%7C%20KNN-orange)

## ✨ Key Features

### 1. 🎓 In-Browser Hand Pose Training

- **Automatic IDs:** Add gesture classes as `ID1`, `ID2`, `ID3`, and so on. Deleted IDs are not reused; a full reset starts again at ID1.
- **Tap or Hold:** A short click or tap collects one sample. Holding for approximately **350 ms** starts collection at approximately **200 ms** intervals, depending on device and camera performance.
- **Fresh Hand Samples:** Collects only valid, recent hand detections. Each detection result can be collected once; a new video frame is required for another sample.
- **Collection Controls:** Releasing or canceling the input, leaving the button, losing the hand, or leaving the page stops collection. Keyboard activation collects one sample.
- **Normalized Features:** Uses **40 features**: the x/y coordinates of 20 finger landmarks relative to the wrist, normalized by the maximum wrist-to-landmark distance.
- **Live Feedback:** Displays sample counts, recognized IDs, and KNN confidence. When fewer than five samples are available, confidence uses the actual neighbor count and the UI indicates insufficient samples.
- **Limits:** Supports up to **100 IDs**, **500 samples per ID**, and **2,000 samples in total**.

### 2. 💾 Model Download, Sharing & Import

- **JSON Download:** Saves learned hand features, class IDs, the next ID number, engine settings, and the display mirror setting in one file.
- **File Sharing:** Opens the device's share sheet for an available app, such as an email app. Unsupported JSON file sharing falls back to a download; canceling sharing does not download a file.
- **Model Import:** Restores files exported by this app. Import replaces the current project after confirmation when IDs already exist, and recognition is started manually afterward.
- **Validated Restore:** Rejects malformed or incompatible files before replacing the current data. Empty IDs and the ID sequence are preserved.

> Files contain learned hand features, not photos, video, or the MediaPipe model itself. Only this app's **version 1 JSON format**, up to **8 MiB**, is supported. Image-classification KNN files are not compatible. Download the project before refreshing or closing the page; there is no automatic persistent storage.

### 3. 🔗 Wireless Control (Web Bluetooth API)

- **Direct Connection:** Uses the **Nordic UART Service** to connect to a compatible micro:bit.
- **ID-Based Transmission:** Sends the displayed gesture ID. Existing device programs that match custom gesture names must be updated to match `ID1`, `ID2`, and so on.
- **Transmission Timing:** Sends changed results or resends after more than **100 ms** since the last successful write. The existing behavior of sending predictions without a confidence threshold is retained.
- **Reliable Stop:** Serializes writes and retries failed stop signals up to five attempts. A write timeout disconnects the link to avoid overlapping unresolved writes.
- **Hand Loss:** Stops collecting immediately when detection is lost. During active recognition, about **500 ms** without a fresh hand result triggers one `stop` signal. Recognition resumes when a hand returns; explicit stop or disconnection requires starting again.
- **Accurate Status:** Updates the last successful ID and timestamp only after a completed Bluetooth write.

### 4. 📱 Responsive & Sticky UI

- **Mobile Portrait:** Keeps the **4:3** camera view below the header while scrolling through the controls.
- **Mobile Landscape:** Displays the camera and controls side by side on supported viewport sizes.
- **Responsive Header:** Keeps the back button on one line and adjusts the camera offset to the actual header height.
- **Landmark Overlay:** Displays the detected hand skeleton on the mirrored camera preview.

---

## 📡 Communication Protocol

The app sends the recognized **gesture ID**, followed by a newline character (`\n`), via Bluetooth UART.

**Data Format:**

```text
ID{ClassNumber}\n
```

**Examples:**

- **When ID1 is recognized:** `ID1\n`
- **When ID2 is recognized:** `ID2\n`
- **When active recognition stops or hand loss exceeds the grace period:** `stop\n`

---

## 🛠️ Developer Testing (Optional)

**Web app users do not need Node.js, npm, or Playwright.** These tools are only required for the developer tests below.

```sh
npm install
npx playwright install chromium
npm test
```

For an existing Microsoft Edge installation on Windows, set `$env:BROWSER_CHANNEL = 'msedge'` and run `npm test`.

- `npm run test:core`: Feature normalization, KNN voting, and JSON validation tests.
- `npm run test:browser`: Loads the actual MediaPipe model, then uses landmark fixtures to test UI behavior. Sharing and Bluetooth are mocked; no files or data are sent to real apps or devices.

Tests require internet access for browser libraries and model assets. Reports and screenshots are saved in `test-results/`, or the directory selected by `TEST_ARTIFACTS`. See [Validation Notes](VALIDATION.md) for coverage and limitations.

The app is a static website with **no build step**. Host `index.html`, `style.css`, `sketch.js`, `hand-model.js`, and `training-input.js` together over HTTPS.

**Feature Compatibility:** The detector handles one hand. Features use x/y only; z coordinates and handedness labels are not used. Mirroring affects the display only. Left/right hands and rotated poses are not automatically canonicalized. KNN uses squared Euclidean distance with `k = 5`; tied votes are resolved by total distance and then numeric ID.

---

**Tech Stack:**

- **Frontend:** HTML5, CSS3
- **Creative Coding:** p5.js **1.6.0** (Canvas, Video Handling)
- **Hand Detection:** MediaPipe Tasks Vision **0.10.8**, HandLandmarker (GPU, one hand)
- **Classification:** Custom JavaScript KNN
- **Model Storage:** JSON Download / Import
- **File Sharing:** Web Share API, where supported
- **Connectivity:** Web Bluetooth API (BLE)

**License:**

- Copyright © 2024 Boundary X Co. All rights reserved.
- All rights to the source code and design of this project belong to BoundaryX.
- Web: [boundaryx.io](https://boundaryx.io)
- Contact: [boundaryx.io/contact](https://boundaryx.io/contact)

