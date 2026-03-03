/**
 * Fire & Smoke Detector — Camera Frontend Logic
 */

// Ganti URL ini dengan URL Hugging Face Space kamu setelah deploy
// Contoh: "https://username-fire-smoke-detector.hf.space"
// Untuk development lokal, gunakan: "http://localhost:7860"
const API_URL = "http://localhost:7860";

// These will be set after DOM loads
let webcamVideo, overlayCanvas, captureCanvas, cameraPrompt;
let startCameraBtn;
let statsPanel, statusBadge;
let fireCount, smokeCount, fpsCounter;
let captureCtx, overlayCtx;

let stream = null;
let isDetecting = false;
let framesThisSecond = 0;
let lastFpsUpdate = 0;

// ========================
// INITIALIZATION
// ========================

document.addEventListener("DOMContentLoaded", () => {
    // Get DOM elements after page loads
    webcamVideo = document.getElementById("webcamVideo");
    webcamVideo.setAttribute("playsinline", ""); // Set via JS to avoid Firefox compat warning in HTML
    overlayCanvas = document.getElementById("overlayCanvas");
    captureCanvas = document.getElementById("captureCanvas");
    cameraPrompt = document.getElementById("cameraPrompt");
    startCameraBtn = document.getElementById("startCameraBtn");
    statsPanel = document.getElementById("statsPanel");
    statusBadge = document.getElementById("statusBadge");
    fireCount = document.getElementById("fireCount");
    smokeCount = document.getElementById("smokeCount");
    fpsCounter = document.getElementById("fpsCounter");

    captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });
    overlayCtx = overlayCanvas.getContext("2d");

    checkApiHealth();

    // Event listeners
    startCameraBtn.addEventListener("click", startCamera);

    window.addEventListener("resize", resizeCanvas);

    console.log("[FireDetector] Initialized successfully");
});

// ========================
// API HEALTH CHECK
// ========================

async function checkApiHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();

        if (data.status === "healthy") {
            statusBadge.classList.add("connected");
            statusBadge.classList.remove("error");
            statusBadge.querySelector(".status-text").textContent = "API Connected";
            console.log("[FireDetector] API connected, model loaded:", data.model_loaded);
        }
    } catch (err) {
        statusBadge.classList.add("error");
        statusBadge.classList.remove("connected");
        statusBadge.querySelector(".status-text").textContent = "API Offline";
        console.error("[FireDetector] API health check failed:", err);
    }
}

// ========================
// CAMERA HANDLING
// ========================

function resizeCanvas() {
    if (!webcamVideo || !webcamVideo.videoWidth) return;

    const rect = webcamVideo.getBoundingClientRect();
    overlayCanvas.width = rect.width;
    overlayCanvas.height = rect.height;

    captureCanvas.width = webcamVideo.videoWidth;
    captureCanvas.height = webcamVideo.videoHeight;

    console.log(`[FireDetector] Canvas resized: overlay=${rect.width}x${rect.height}, capture=${webcamVideo.videoWidth}x${webcamVideo.videoHeight}`);
}

async function startCamera() {
    try {
        startCameraBtn.disabled = true;
        startCameraBtn.querySelector("span").textContent = "Starting...";

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        webcamVideo.srcObject = stream;

        webcamVideo.onloadedmetadata = () => {
            webcamVideo.play();
            resizeCanvas();

            // Hide prompt, show stats
            cameraPrompt.classList.add("hidden");
            statsPanel.style.display = "flex";

            console.log("[FireDetector] Camera started, beginning detection loop");

            // Start Detection Loop
            startDetectionLoop();
        };

    } catch (err) {
        console.error("[FireDetector] Camera access error:", err);
        alert("Could not access camera. Please ensure you have granted permission.");
        startCameraBtn.disabled = false;
        startCameraBtn.querySelector("span").textContent = "Start Camera";
    }
}

// ========================
// DETECTION LOOP
// ========================

function startDetectionLoop() {
    isDetecting = true;
    lastFpsUpdate = performance.now();
    framesThisSecond = 0;
    console.log("[FireDetector] Detection loop started");
    detectFrame();
}

function calculateFPS() {
    const now = performance.now();
    framesThisSecond++;

    if (now - lastFpsUpdate >= 1000) {
        if (fpsCounter) fpsCounter.textContent = framesThisSecond;
        framesThisSecond = 0;
        lastFpsUpdate = now;
    }
}

async function detectFrame() {
    if (!isDetecting) return;

    // Wait until video has valid dimensions
    if (!webcamVideo.videoWidth || !webcamVideo.videoHeight) {
        requestAnimationFrame(detectFrame);
        return;
    }

    // Ensure captureCanvas is correctly sized
    if (captureCanvas.width !== webcamVideo.videoWidth) {
        captureCanvas.width = webcamVideo.videoWidth;
        captureCanvas.height = webcamVideo.videoHeight;
    }

    // Draw current video frame to hidden capture canvas
    captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);

    try {
        // Convert canvas to blob (JPEG for speed)
        const blob = await new Promise(resolve => captureCanvas.toBlob(resolve, "image/jpeg", 0.75));

        if (!blob) {
            console.warn("[FireDetector] Failed to create blob from canvas");
            if (isDetecting) requestAnimationFrame(detectFrame);
            return;
        }

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        // Use backend default threshold
        const response = await fetch(
            `${API_URL}/detect`,
            {
                method: "POST",
                body: formData,
            }
        );

        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                drawDetections(data);

                // Update stats
                fireCount.textContent = data.class_summary.fire || 0;
                smokeCount.textContent = data.class_summary.smoke || 0;

                calculateFPS();
            }
        } else {
            const errText = await response.text();
            console.error("[FireDetector] API error:", response.status, errText);
        }
    } catch (error) {
        console.error("[FireDetector] Detection fetch error:", error.message);
    }

    // Schedule next frame
    if (isDetecting) {
        // Small delay to not overwhelm CPU (especially since we're also training)
        setTimeout(() => requestAnimationFrame(detectFrame), 100);
    }
}

// ========================
// DRAWING (WHITE BOX DETECTOR)
// ========================

function drawDetections(data) {
    if (!overlayCanvas.width || !overlayCanvas.height) resizeCanvas();

    // Clear previous drawings
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!data.detections || data.detections.length === 0) return;

    // Calculate scaling factors between video source and displayed canvas
    const scaleX = overlayCanvas.width / captureCanvas.width;
    const scaleY = overlayCanvas.height / captureCanvas.height;

    // Setup drawing style: White Bounding Boxes
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = "#FFFFFF";
    overlayCtx.font = "bold 16px Inter, sans-serif";

    data.detections.forEach(det => {
        // Scale coordinates to fit screen
        const x = det.bbox.x1 * scaleX;
        const y = det.bbox.y1 * scaleY;
        const width = (det.bbox.x2 - det.bbox.x1) * scaleX;
        const height = (det.bbox.y2 - det.bbox.y1) * scaleY;

        // Draw White Box
        overlayCtx.beginPath();
        overlayCtx.rect(x, y, width, height);
        overlayCtx.stroke();

        // Label Text
        const text = `${det.class.toUpperCase()} ${(det.confidence * 100).toFixed(0)}%`;
        const textWidth = overlayCtx.measureText(text).width;

        // Draw Text Background (White pill)
        overlayCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
        const labelY = Math.max(0, y - 28);
        overlayCtx.fillRect(x, labelY, textWidth + 16, 26);

        // Draw text (Black on white)
        overlayCtx.fillStyle = "#000000";
        overlayCtx.fillText(text, x + 8, labelY + 18);
    });
}
