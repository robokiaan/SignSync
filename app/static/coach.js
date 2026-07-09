// SignSync ISL - Real-time AI Coach Feedback Engine
//
// A SINGLE MediaPipe Holistic model instance serves two jobs, switched via
// `holisticMode`:
//   'ref'  - priming: run the reference video through the model once to build
//            the reference feature sequence (then cached per sign).
//   'user' - coaching: run the live webcam and score the learner via DTW.
// This replaces the old design that ran a second Holistic model on the
// reference video every session (and accumulated frames across <video loop>).

let webcamStream = null;
let holisticModel = null;      // the one and only Holistic instance
let cameraHelper = null;

let isWebcamActive = false;

// Sustained-scoring + smoothing state (reset per sign in resetDTWSequences)
let smoothedScore = null;      // EMA of the instantaneous score (not a running max)
let prevUserSmoothed = null;   // last smoothed user frame, for feature EMA
let refUsesRight = false;      // does the reference sign actually use each hand?
let refUsesLeft = false;

let canvasElement, canvasCtx;

// DTW sequence buffers + reference cache
let refSequence = [];
let userSequence = [];
const refCache = {};           // signName -> reference feature sequence (in-memory)
let holisticMode = "idle";     // 'idle' | 'ref' | 'user'
let refReadyPromise = Promise.resolve();
let activePrimeId = 0;         // guards against overlapping reference primings

const SAMPLE_FPS = 5; // reference sampling rate; matches the live user throttle (200ms)

const MEDIAPIPE_OPTIONS = {
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
};

function resetCoachState() {
    document.getElementById("coach-score-display").textContent = "--%";
    document.getElementById("coach-feedback-status").textContent = "Idle";
    document.getElementById("coach-feedback-message").textContent = "Activate your webcam to begin real-time gesture analysis.";

    canvasElement = document.getElementById("webcam-canvas");
    canvasCtx = canvasElement.getContext("2d");
}

// Lazily create the single shared Holistic model.
function ensureHolisticModel() {
    if (holisticModel) return holisticModel;
    holisticModel = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });
    holisticModel.setOptions(MEDIAPIPE_OPTIONS);
    holisticModel.onResults(onHolisticResults);
    return holisticModel;
}

// Single results dispatcher for both reference priming and live coaching.
function onHolisticResults(results) {
    if (holisticMode === "ref") {
        refSequence.push(extractFrameFeatures(results));
        return;
    }
    if (holisticMode !== "user" || !isWebcamActive || !canvasElement || !canvasCtx) return;

    resizeCanvasIfNeeded();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    drawCustomSkeleton(results);

    if (activeSign) analyzeFeedback(results);
}

// ---------------------------------------------------------------------------
// Reference priming: run the reference video through the model exactly once.
// ---------------------------------------------------------------------------
async function primeReference(signName) {
    const myId = ++activePrimeId;

    if (refCache[signName]) {
        refSequence = refCache[signName];
        computeRefHandUsage();
        return;
    }

    const video = document.getElementById("practice-ref-video");
    if (!video) { refSequence = []; return; }

    ensureHolisticModel();
    await waitForVideoReady(video);
    if (myId !== activePrimeId) return; // superseded by a newer sign selection

    document.getElementById("coach-feedback-status").textContent = "Analyzing reference...";
    refSequence = [];
    const previousMode = holisticMode;
    holisticMode = "ref";
    try {
        await sampleVideoThroughModel(video, myId);
    } finally {
        // Restore to coaching if the webcam is live, otherwise idle.
        holisticMode = isWebcamActive ? "user" : "idle";
    }

    if (myId === activePrimeId) {
        refCache[signName] = refSequence.slice();
        computeRefHandUsage();
        // Return the video to a clean looping state for the learner to watch.
        video.loop = true;
        try { video.currentTime = 0; } catch (e) { /* ignore */ }
    }
}

// A hand "counts" for a sign if it's detected in a meaningful fraction of the
// reference frames. Used to penalize the learner for omitting a required hand.
function computeRefHandUsage() {
    refUsesRight = false;
    refUsesLeft = false;
    if (!refSequence.length) return;
    let r = 0, l = 0;
    for (const f of refSequence) {
        if (f.visibility.rightHand) r++;
        if (f.visibility.leftHand) l++;
    }
    refUsesRight = r / refSequence.length >= 0.3;
    refUsesLeft = l / refSequence.length >= 0.3;
}

function waitForVideoReady(video) {
    return new Promise((resolve) => {
        if (video.readyState >= 2) { resolve(); return; }
        video.addEventListener("loadeddata", () => resolve(), { once: true });
    });
}

// Step through the video by seeking to fixed timestamps (every 1/SAMPLE_FPS
// seconds), fully awaiting each frame before advancing. This is deterministic
// and does NOT depend on real-time playback, so it can't race with the model's
// async init or per-frame latency (a play-based sampler drops frames when the
// short clip ends before the first send() resolves).
async function sampleVideoThroughModel(video, myId) {
    video.pause();
    video.loop = false;

    let duration = video.duration;
    if (!isFinite(duration) || duration <= 0) duration = 10; // safety fallback

    const step = 1 / SAMPLE_FPS; // 0.2s to match the live 5 FPS user rate
    for (let t = 0; t < duration - 1e-3; t += step) {
        if (myId !== activePrimeId) return;
        await seekVideo(video, t);
        if (myId !== activePrimeId) return;
        try {
            await holisticModel.send({ image: video });
        } catch (err) {
            console.error("Reference priming frame error:", err);
        }
    }
}

function seekVideo(video, time) {
    return new Promise((resolve) => {
        const onSeeked = () => {
            video.removeEventListener("seeked", onSeeked);
            resolve();
        };
        video.addEventListener("seeked", onSeeked);
        try {
            video.currentTime = time;
        } catch (e) {
            video.removeEventListener("seeked", onSeeked);
            resolve();
        }
    });
}

// ---------------------------------------------------------------------------
// Webcam control
// ---------------------------------------------------------------------------
async function toggleWebcam() {
    const btn = document.getElementById("btn-toggle-camera");

    if (isWebcamActive) {
        stopWebcamStream();
        btn.textContent = "Start Camera";
        btn.style.background = "";
        resetCoachState();
        return;
    }

    document.getElementById("coach-feedback-status").textContent = "Starting camera...";
    btn.textContent = "Stop Camera";
    btn.style.background = "var(--danger)";

    try {
        const videoElement = document.getElementById("webcam-raw");
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
        });
        videoElement.srcObject = webcamStream;

        ensureHolisticModel();

        // Make sure the reference sequence is primed before we score anything.
        document.getElementById("coach-feedback-status").textContent = "Preparing reference...";
        await refReadyPromise;

        isWebcamActive = true;
        holisticMode = "user";
        document.getElementById("coach-feedback-status").textContent = "Coach Active";

        let lastProcessTime = 0;
        cameraHelper = new Camera(videoElement, {
            onFrame: async () => {
                if (!isWebcamActive || holisticMode !== "user") return;
                const now = performance.now();
                if (now - lastProcessTime >= 200) { // throttle to 5 FPS
                    lastProcessTime = now;
                    await holisticModel.send({ image: videoElement });
                }
            },
            width: 640,
            height: 480,
        });
        cameraHelper.start();
    } catch (err) {
        console.error("Camera startup failed:", err);
        document.getElementById("coach-feedback-status").textContent = "Camera Error";
        document.getElementById("coach-feedback-message").textContent = "Could not access your webcam. Check browser permissions.";
        btn.textContent = "Start Camera";
        btn.style.background = "";
        isWebcamActive = false;
        holisticMode = "idle";
    }
}

function stopWebcamStream() {
    isWebcamActive = false;
    if (holisticMode === "user") holisticMode = "idle";

    if (cameraHelper) {
        cameraHelper.stop();
        cameraHelper = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        webcamStream = null;
    }

    const videoElement = document.getElementById("webcam-raw");
    if (videoElement) videoElement.srcObject = null;

    if (canvasCtx && canvasElement) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
}

// Only reallocate the canvas backing store when its display size actually
// changes (reassigning width/height every frame clears + reallocates it).
function resizeCanvasIfNeeded() {
    const w = canvasElement.clientWidth;
    const h = canvasElement.clientHeight;
    if (canvasElement.width !== w || canvasElement.height !== h) {
        canvasElement.width = w;
        canvasElement.height = h;
    }
}

// ---------------------------------------------------------------------------
// Skeleton overlay
// ---------------------------------------------------------------------------
function drawCustomSkeleton(results) {
    const ctx = canvasCtx;
    const w = canvasElement.width;
    const h = canvasElement.height;

    const project = (lm) => ({ x: lm.x * w, y: lm.y * h });

    const drawLine = (pt1, pt2, color, thickness = 2) => {
        if (!pt1 || !pt2) return;
        const p1 = project(pt1);
        const p2 = project(pt2);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.shadowBlur = 4;
        ctx.shadowColor = color;
        ctx.stroke();
        ctx.shadowBlur = 0;
    };

    const drawJoint = (pt, color, radius = 4) => {
        if (!pt) return;
        const p = project(pt);
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
    };

    const pose = results.poseLandmarks;
    if (pose) {
        const leftSh = pose[11], rightSh = pose[12];
        const leftEl = pose[13], rightEl = pose[14];
        const leftWr = pose[15], rightWr = pose[16];

        drawLine(leftSh, rightSh, "#4f46e5", 3);
        drawLine(leftSh, leftEl, "#6366f1", 3);
        drawLine(leftEl, leftWr, "#06b6d4", 3);
        drawLine(rightSh, rightEl, "#6366f1", 3);
        drawLine(rightEl, rightWr, "#06b6d4", 3);

        [leftSh, rightSh, leftEl, rightEl, leftWr, rightWr].forEach((p) => drawJoint(p, "#14b8a6", 5));
    }

    if (results.leftHandLandmarks) drawHandBones(results.leftHandLandmarks, drawLine, drawJoint);
    if (results.rightHandLandmarks) drawHandBones(results.rightHandLandmarks, drawLine, drawJoint);
}

function drawHandBones(landmarks, drawLine, drawJoint) {
    const wrist = landmarks[0];
    const colorLine = "#10b981";
    const colorJoint = "#67e8f9";

    for (let f = 0; f < 5; f++) {
        const start = 1 + (f * 4);
        drawLine(wrist, landmarks[start], colorLine, 2);
        for (let j = 0; j < 3; j++) {
            drawLine(landmarks[start + j], landmarks[start + j + 1], colorLine, 2);
        }
    }
    landmarks.forEach((p) => drawJoint(p, colorJoint, 3));
}

// ---------------------------------------------------------------------------
// DTW gesture analysis
// ---------------------------------------------------------------------------
// 26-D feature layout (all components normalized to ~[0,1]):
//   0-4   right finger bend (thumb..pinky)     gate: rightHand
//   5-7   right palm-normal x,y,z              gate: rightHand
//   8     right finger spread                  gate: rightHand
//   9-13  left finger bend                     gate: leftHand
//   14-16 left palm-normal x,y,z               gate: leftHand
//   17    left finger spread                   gate: leftHand
//   18    right elbow angle                    gate: pose
//   19    left elbow angle                     gate: pose
//   20    right shoulder (arm elevation)       gate: pose
//   21    left shoulder (arm elevation)        gate: pose
//   22-23 right wrist location x,y             gate: rightHand + pose
//   24-25 left wrist location x,y              gate: leftHand + pose
const FEATURE_GROUPS = [
    { start: 0, end: 9, needRight: true },
    { start: 9, end: 18, needLeft: true },
    { start: 18, end: 22, needPose: true },
    { start: 22, end: 24, needRight: true, needPose: true },
    { start: 24, end: 26, needLeft: true, needPose: true },
];
const FEATURE_DIM = 26;

// Per-feature coaching message, indexed to match the layout above.
const FEATURE_FEEDBACK = (() => {
    const m = new Array(FEATURE_DIM);
    const fingers = ["thumb", "index finger", "middle finger", "ring finger", "pinky finger"];
    for (const [side, base] of [["right", 0], ["left", 9]]) {
        for (let i = 0; i < 5; i++) m[base + i] = `Adjust your ${side} ${fingers[i]} shape.`;
        m[base + 5] = m[base + 6] = m[base + 7] = `Rotate your ${side} palm to match the reference orientation.`;
        m[base + 8] = `Adjust the spread between your ${side} fingers.`;
    }
    m[18] = "Check your right arm or elbow bend.";
    m[19] = "Check your left arm or elbow bend.";
    m[20] = "Adjust how high you raise your right arm.";
    m[21] = "Adjust how high you raise your left arm.";
    m[22] = m[23] = "Move your right hand to the correct position.";
    m[24] = m[25] = "Move your left hand to the correct position.";
    return m;
})();

// Is feature index `k` mutually observed by both frames (per its group's gate)?
function featureVisible(k, uVis, rVis) {
    for (const g of FEATURE_GROUPS) {
        if (k >= g.start && k < g.end) {
            if (g.needRight && !(uVis.rightHand && rVis.rightHand)) return false;
            if (g.needLeft && !(uVis.leftHand && rVis.leftHand)) return false;
            if (g.needPose && !(uVis.pose && rVis.pose)) return false;
            return true;
        }
    }
    return false;
}

function clamp01(v) {
    return v < 0 ? 0 : (v > 1 ? 1 : v);
}

let lastSpokenTime = 0;
let lastSpokenText = "";

function speakFeedback(text) {
    const now = performance.now();
    if (now - lastSpokenTime >= 3500 && text !== lastSpokenText) {
        lastSpokenTime = now;
        lastSpokenText = text;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        window.speechSynthesis.speak(utterance);
    }
}

function resetDTWSequences() {
    refSequence = [];
    userSequence = [];
    smoothedScore = null;
    prevUserSmoothed = null;
    refUsesRight = false;
    refUsesLeft = false;
}

function getMaskedVectorDistance(userFrame, refFrame) {
    // No hands visible at all => force a maximum mismatch (can't "pass" by just showing a face).
    if (!userFrame.visibility.rightHand && !userFrame.visibility.leftHand) {
        return 1.0;
    }

    const uVis = userFrame.visibility;
    const rVis = refFrame.visibility;
    const uf = userFrame.features;
    const rf = refFrame.features;

    let sum = 0;
    let count = 0;
    for (const g of FEATURE_GROUPS) {
        if (g.needRight && !(uVis.rightHand && rVis.rightHand)) continue;
        if (g.needLeft && !(uVis.leftHand && rVis.leftHand)) continue;
        if (g.needPose && !(uVis.pose && rVis.pose)) continue;
        for (let i = g.start; i < g.end; i++) {
            const d = uf[i] - rf[i];
            sum += d * d;
            count++;
        }
    }

    if (count === 0) return 1.0;
    return Math.sqrt(sum / count);
}

// Light per-feature EMA on the live user stream to damp landmark jitter.
// Resets whenever the visibility profile changes (don't blend across a hand
// appearing/disappearing, or you'd average real values with 0.5 defaults).
function smoothUserFrame(frame) {
    const a = 0.5;
    if (!prevUserSmoothed || !sameVisibility(prevUserSmoothed.visibility, frame.visibility)) {
        prevUserSmoothed = { features: frame.features.slice(), visibility: frame.visibility };
        return { features: frame.features.slice(), visibility: frame.visibility };
    }
    const f = frame.features.map((v, i) => a * v + (1 - a) * prevUserSmoothed.features[i]);
    prevUserSmoothed = { features: f.slice(), visibility: frame.visibility };
    return { features: f, visibility: frame.visibility };
}

function sameVisibility(a, b) {
    return a.rightHand === b.rightHand && a.leftHand === b.leftHand && a.pose === b.pose;
}

function analyzeFeedback(results) {
    userSequence.push(smoothUserFrame(extractFrameFeatures(results)));

    // Sliding window sized to the reference bounds cost/memory over a session.
    const cap = Math.max(20, Math.ceil(refSequence.length * 1.5));
    if (userSequence.length > cap) {
        userSequence.splice(0, userSequence.length - cap);
    }

    if (refSequence.length < 5 || userSequence.length < 5) {
        document.getElementById("coach-feedback-status").textContent = "Recording movement...";
        document.getElementById("coach-feedback-message").textContent = "Keep replicating the sign movements as shown in the reference video.";
        return;
    }

    const { cost, coverage } = calculateDTW(userSequence, refSequence);

    // Base quality from the warp cost.
    const threshold = 0.30;
    const quality = cost < threshold ? (1 - cost / threshold) : 0; // 0..1

    // Coverage gate: reward only attempts that traverse most of the reference,
    // so a short lucky fragment (open-ended DTW) can't score high.
    const coverageFactor = Math.min(1, coverage / 0.6);

    // Hand penalty: don't give credit for omitting a hand the sign requires.
    let userRight = 0, userLeft = 0;
    for (const f of userSequence) {
        if (f.visibility.rightHand) userRight++;
        if (f.visibility.leftHand) userLeft++;
    }
    const userRightFrac = userRight / userSequence.length;
    const userLeftFrac = userLeft / userSequence.length;
    let handFactor = 1;
    if (refUsesRight && userRightFrac < 0.3) handFactor *= 0.4;
    if (refUsesLeft && userLeftFrac < 0.3) handFactor *= 0.4;

    const instant = 100 * quality * coverageFactor * handFactor;

    // Sustained score: EMA, so it reflects held correctness rather than a single
    // best-ever frame (and it decays if the learner stops performing the sign).
    smoothedScore = (smoothedScore === null) ? instant : (0.7 * smoothedScore + 0.3 * instant);
    const displayScore = Math.round(clamp01(smoothedScore / 100) * 99);

    // Diagnostic: largest deviating joint against the DTW-aligned reference frame.
    const userFeaturesObj = userSequence[userSequence.length - 1];
    const matchedRefFeaturesObj = refSequence[nearestRefFrame(userFeaturesObj)];
    let maxDiff = 0;
    let maxDiffIdx = -1;
    for (let k = 0; k < userFeaturesObj.features.length; k++) {
        if (!featureVisible(k, userFeaturesObj.visibility, matchedRefFeaturesObj.visibility)) continue;
        const diff = Math.abs(userFeaturesObj.features[k] - matchedRefFeaturesObj.features[k]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = k;
        }
    }

    let feedback;
    if (refUsesRight && userRightFrac < 0.3) {
        feedback = "Use your right hand for this sign.";
        speakFeedback(feedback);
    } else if (refUsesLeft && userLeftFrac < 0.3) {
        feedback = "This sign needs your left hand too.";
        speakFeedback(feedback);
    } else if (displayScore >= 70) {
        feedback = "Perfect alignment! Excellent movement matching.";
        speakFeedback("Excellent movement! Hold it steady.");
    } else if (maxDiff > 0.18 && maxDiffIdx >= 0) {
        feedback = FEATURE_FEEDBACK[maxDiffIdx];
        speakFeedback(feedback);
    } else if (displayScore >= 40) {
        feedback = "Good motion! Keep moving in synchronization.";
        speakFeedback("Good motion! Keep moving.");
    } else {
        feedback = "Replicate the sequence of movements shown in the reference sign.";
        speakFeedback("Watch the reference and match the movement.");
    }

    document.getElementById("coach-feedback-status").textContent = `Score: ${displayScore}%`;
    document.getElementById("coach-feedback-message").textContent = feedback;
    document.getElementById("coach-score-display").textContent = `${displayScore}%`;
}

function nearestRefFrame(userFrame) {
    let bestIdx = 0;
    let minDist = Infinity;
    for (let i = 0; i < refSequence.length; i++) {
        const d = getMaskedVectorDistance(userFrame, refSequence[i]);
        if (d < minDist) {
            minDist = d;
            bestIdx = i;
        }
    }
    return bestIdx;
}

// Subsequence DTW (open-ended start AND end): the user sequence may align to any
// contiguous window of the reference. Returns the normalized warp cost plus the
// coverage = fraction of the reference spanned by the optimal alignment path
// (recovered by backtracking), which the scorer uses to reject partial attempts.
function calculateDTW(seq1, seq2) {
    const n = seq1.length;
    const m = seq2.length;
    if (n === 0 || m === 0) return { cost: Infinity, coverage: 0 };

    const D = Array.from({ length: n }, () => new Float64Array(m));
    const P = Array.from({ length: n }, () => new Int8Array(m)); // 0=diag, 1=up, 2=left

    for (let j = 0; j < m; j++) {
        D[0][j] = getMaskedVectorDistance(seq1[0], seq2[j]); // open start: any column, no penalty
    }
    for (let i = 1; i < n; i++) {
        D[i][0] = getMaskedVectorDistance(seq1[i], seq2[0]) + D[i - 1][0];
        P[i][0] = 1;
        for (let j = 1; j < m; j++) {
            const cost = getMaskedVectorDistance(seq1[i], seq2[j]);
            const diag = D[i - 1][j - 1], up = D[i - 1][j], left = D[i][j - 1];
            let best = diag, p = 0;
            if (up < best) { best = up; p = 1; }
            if (left < best) { best = left; p = 2; }
            D[i][j] = cost + best;
            P[i][j] = p;
        }
    }

    // Open end: minimum over the final user row.
    let endJ = 0, minCost = Infinity;
    for (let j = 0; j < m; j++) {
        if (D[n - 1][j] < minCost) { minCost = D[n - 1][j]; endJ = j; }
    }

    // Backtrack to find the reference span the path covers.
    let i = n - 1, j = endJ, minJ = endJ, maxJ = endJ;
    while (i > 0) {
        const p = P[i][j];
        if (p === 0) { i--; j--; }
        else if (p === 1) { i--; }
        else { j--; }
        if (j < minJ) minJ = j;
        if (j > maxJ) maxJ = j;
    }

    return { cost: minCost / n, coverage: (maxJ - minJ + 1) / m };
}

// ---------------------------------------------------------------------------
// Feature extraction (26-D: handshape + orientation + spread + arm pose +
// hand location, plus visibility flags). See FEATURE_GROUPS for the layout.
// ---------------------------------------------------------------------------
const HAND_DEFAULT = new Array(9).fill(0.5); // 5 bend + 3 palm-normal + 1 spread

function extractFrameFeatures(results) {
    const visibility = {
        rightHand: !!results.rightHandLandmarks,
        leftHand: !!results.leftHandLandmarks,
        pose: !!results.poseLandmarks,
    };

    const features = [];
    features.push(...(results.rightHandLandmarks ? getHandFeatures(results.rightHandLandmarks) : HAND_DEFAULT));
    features.push(...(results.leftHandLandmarks ? getHandFeatures(results.leftHandLandmarks) : HAND_DEFAULT));

    const pose = results.poseLandmarks;
    if (pose) {
        features.push(getPoseAngle(pose[12], pose[14], pose[16]) / 180.0); // right elbow
        features.push(getPoseAngle(pose[11], pose[13], pose[15]) / 180.0); // left elbow
        features.push(getPoseAngle(pose[24], pose[12], pose[14]) / 180.0); // right shoulder elevation
        features.push(getPoseAngle(pose[23], pose[11], pose[13]) / 180.0); // left shoulder elevation
        const rw = wristLocation(pose, 16);
        features.push(rw[0], rw[1]); // right wrist x,y relative to shoulders
        const lw = wristLocation(pose, 15);
        features.push(lw[0], lw[1]); // left wrist x,y relative to shoulders
    } else {
        features.push(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
    }

    return { features, visibility };
}

// 9 features per hand: 5 finger bends, 3 palm-normal components, 1 finger spread.
function getHandFeatures(hand) {
    const n = palmNormal(hand);
    return [
        getFingerBentAngle(hand, 0, 1, 2, 4) / 180.0,
        getFingerBentAngle(hand, 5, 6, 7, 8) / 180.0,
        getFingerBentAngle(hand, 9, 10, 11, 12) / 180.0,
        getFingerBentAngle(hand, 13, 14, 15, 16) / 180.0,
        getFingerBentAngle(hand, 17, 18, 19, 20) / 180.0,
        n[0], n[1], n[2],
        fingerSpread(hand),
    ];
}

// Palm-facing direction as a unit normal (wrist->index x wrist->pinky), each
// component remapped from [-1,1] to [0,1]. Captures hand orientation, which the
// rotation-invariant bend angles miss.
function palmNormal(hand) {
    const w = hand[0], idx = hand[5], pky = hand[17];
    const a = { x: idx.x - w.x, y: idx.y - w.y, z: idx.z - w.z };
    const b = { x: pky.x - w.x, y: pky.y - w.y, z: pky.z - w.z };
    let cx = a.y * b.z - a.z * b.y;
    let cy = a.z * b.x - a.x * b.z;
    let cz = a.x * b.y - a.y * b.x;
    const mag = Math.sqrt(cx * cx + cy * cy + cz * cz) || 1;
    return [(cx / mag + 1) / 2, (cy / mag + 1) / 2, (cz / mag + 1) / 2];
}

// How splayed the fingers are: summed angle between adjacent proximal phalanges.
function fingerSpread(hand) {
    const dir = (mcp, pip) => ({ x: hand[pip].x - hand[mcp].x, y: hand[pip].y - hand[mcp].y, z: hand[pip].z - hand[mcp].z });
    const d = [dir(5, 6), dir(9, 10), dir(13, 14), dir(17, 18)];
    let sum = 0;
    for (let i = 0; i < 3; i++) sum += vecAngleDeg(d[i], d[i + 1]);
    return clamp01(sum / 120.0);
}

// Wrist position relative to the shoulder midpoint, scaled by shoulder width so
// it stays scale-invariant while restoring the "location" sign parameter.
function wristLocation(pose, wristIdx) {
    const ls = pose[11], rs = pose[12], wr = pose[wristIdx];
    const cx = (ls.x + rs.x) / 2, cy = (ls.y + rs.y) / 2;
    const shoulderW = Math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2) || 0.1;
    const dx = (wr.x - cx) / shoulderW;
    const dy = (wr.y - cy) / shoulderW;
    return [clamp01((dx + 2.5) / 5), clamp01((dy + 2.5) / 5)];
}

function vecAngleDeg(v1, v2) {
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const m1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const m2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    if (m1 === 0 || m2 === 0) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / (m1 * m2)))) * (180.0 / Math.PI);
}

function getPoseAngle(p1, p2, p3) {
    if (!p1 || !p2 || !p3) return 180.0;
    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y, z: p1.z - p2.z };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y, z: p3.z - p2.z };
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    if (mag1 === 0 || mag2 === 0) return 180.0;
    const clampedCos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
    return Math.acos(clampedCos) * (180.0 / Math.PI);
}

// Knuckle joint angle in degrees (180 = straight finger, small = curled).
function getFingerBentAngle(hand, p0, p1, p2, p3) {
    if (!hand[p0] || !hand[p1] || !hand[p2] || !hand[p3]) return 180;

    const v1 = { x: hand[p1].x - hand[p0].x, y: hand[p1].y - hand[p0].y, z: hand[p1].z - hand[p0].z };
    const v2 = { x: hand[p3].x - hand[p2].x, y: hand[p3].y - hand[p2].y, z: hand[p3].z - hand[p2].z };

    const dot = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    if (mag1 === 0 || mag2 === 0) return 180;

    const clampedCos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
    return (Math.acos(clampedCos) * 180) / Math.PI;
}
