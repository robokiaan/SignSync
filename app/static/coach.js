// SignSync ISL - Real-time AI Coach Feedback Engine
//
// A SINGLE MediaPipe Holistic model instance serves two jobs, switched via
// `holisticMode`:
//   'ref'  - priming: run the reference video through the model once to build
//            the reference feature sequence (then cached per sign).
//   'user' - coaching: run the live webcam and score the learner against the
//            reference's Movement-Hold phase model.
// This replaces the old design that ran a second Holistic model on the
// reference video every session (and accumulated frames across <video loop>).

let webcamStream = null;
let holisticModel = null;      // the one and only Holistic instance
let cameraHelper = null;

let isWebcamActive = false;

// ---------------------------------------------------------------------------
// Movement-Hold phase scoring
// ---------------------------------------------------------------------------
// A sign is modeled as an ordered sequence of HOLDS (informative target poses)
// linked by MOVES (transitions with a dominant direction). The reference is
// auto-segmented into this model at prime time from its motion profile. The
// learner is graded by how well they hit each hold pose, in order (+ a light
// check that the moves go the right way). Because holds are the target, holding
// a pose no longer collapses the score, and feedback is per-phase.
const HOLD_MATCH_THRESHOLD = 0.50; // masked distance at which a hold's quality hits 0
const HOLD_COMPLETE_Q = 0.55;      // quality above which a phase counts as "reached"
const REF_MOTION_HOLD_FRAC = 0.45; // ref frame is a hold if motion < this fraction of the sign's peak motion
const REF_MIN_HOLD_MOTION = 0.025; // absolute floor for the hold-motion threshold
const PHASE_MERGE_DIST = 0.10;     // merge consecutive ref holds whose targets are closer than this
// Idle-bookend trimming: the clip settles into a neutral resting pose (hands
// hanging down) before/after the sign. Leading/trailing holds whose highest hand
// is low in the frame (wrist-Y above REST_FLOOR) are dropped as dormant; active
// signing poses sit below the floor and are kept. We only trim the ends and stop
// at one hold, so every sign keeps at least its single most meaningful phase.
// Calibrated across ~20 sampled signs: genuine idle/rest poses land at wrist-Y
// 0.957-1.0 (usually exactly 1.0, hand fully hanging), while real end-of-sign
// holds that were getting wrongly swept up sit at 0.92-0.943 (e.g. "hello"'s
// closing pose at 0.928, "how are you"'s repeated-motion holds at ~0.93-0.94).
// 0.95 sits in the gap between those two populations. Anchored against
// "alright", whose intentional 3->1 collapse (bookends pinned at Y=1.0, real
// thumbs-up content at Y=0.902) must survive any reasonable threshold here.
const REST_FLOOR = 0.95;
const MOVE_WEIGHT = 0.20;          // how much movement-direction agreement modulates a phase's credit
// Feature indices that carry hand position/orientation (used for move direction).
const POSITION_DIMS = [5, 6, 7, 14, 15, 16, 22, 23, 24, 25];

// Per-attempt scoring state (reset in resetDTWSequences)
let activePhaseModel = null;   // { holds, moveDirs, holdTimes, requires }
let phaseBest = [];            // best hold quality achieved per phase this attempt
let phaseReached = [];         // has each phase been hit at least at HOLD_COMPLETE_Q?
let phaseUserPose = [];        // user feature vector captured when each phase was reached
let curPhase = 0;              // phase the learner is currently working toward
let leftFinalPose = false;     // has the learner moved off the final pose (guards attempt restart)?
let missingLimbFrames = 0;     // consecutive frames missing a required limb (debounces the prompt)
let lastDisplayScore = 0;      // last score shown (kept while prompting for limbs)
let prevUserSmoothed = null;   // last smoothed user frame, for feature EMA

// Sentence-practice session state (reset in endSentenceSession). The whole
// sentence is graded as ONE multi-phase model (sentenceCombinedModel - every
// word's holds concatenated in order), reusing the exact same phase state
// machine single-sign practice uses (activePhaseModel/phaseBest/phaseReached/
// curPhase, scoreActiveModel) rather than resetting to a fresh 1-3-phase
// model per word. That's deliberate: a lone word's score is coarse (few
// phases to average over, so it reads as near-binary), while the combined
// model spans every phase of every word, giving the same per-phase-quality
// averaging individual signs get, just with more phases feeding it - a
// smoother, more accurate percentage instead of one that resets each word.
let sentenceActive = false;         // is a sentence session in progress?
let sentenceGloss = [];             // ordered lowercase sign names for the active sentence
let sentenceCombinedModel = null;   // {holds, moveDirs, holdTimes, requires, wordPhaseRanges} spanning the whole sentence
let savedFocusStash = null;         // combined-model phaseBest/phaseReached/phaseUserPose/curPhase, saved while a single-word focus view (setSentenceViewMode) borrows the shared scoring state
let misclassifyStreak = 0;          // consecutive frames the classifier saw a DIFFERENT word than expected
let lastMisclassifiedWord = null;   // that word, for the "looks like X" feedback message
let sentenceViewMode = "all";       // sentence-mode view selector; "all" is the default and only mode today
let videoChainIdx = 0;              // index into sentenceGloss for the combined reference-video playback ("all" mode)

let coachDebug = true;         // show the live scoring breakdown panel

let canvasElement, canvasCtx;

// Reference sequence buffers + caches
let refSequence = [];
// Raw-landmark capture for the 3D avatar pipeline (offline extraction only -
// see scripts/extract_avatar_landmarks.py). Gated behind window.CAPTURE_RAW_LANDMARKS
// (default off) so this is a no-op, zero-behavior-change addition to the live
// coaching path. Populated alongside refSequence during 'ref' priming.
let rawRefSequence = [];
const refCache = {};           // signName -> reference feature sequence (in-memory)
const phaseCache = {};         // signName -> phase model
let holisticMode = "idle";     // 'idle' | 'ref' | 'user'
let refReadyPromise = Promise.resolve();
let activePrimeId = 0;         // guards against overlapping reference primings

// Offline-precomputed phase models (scripts/precompute_phases.py), keyed by
// lowercased sign name. When present for a sign, primeReference() skips live
// in-browser priming entirely instead of running the reference video through
// Holistic frame-by-frame. Missing/failed fetch just means every sign falls
// back to live priming, same as before this file existed.
const precomputedPhasesPromise = fetch("/phases.json")
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}));

const SAMPLE_FPS = 5;           // reference sampling rate (holds are matched per-frame, so the live rate can be higher)
const USER_THROTTLE_MS = 100;   // live webcam MediaPipe interval; lower = snappier score (send() is serialized, so no backup)

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
        if (window.CAPTURE_RAW_LANDMARKS) {
            rawRefSequence.push({
                pose: results.poseLandmarks || null,
                poseWorld: findWorldLandmarks(results),
                leftHand: results.leftHandLandmarks || null,
                rightHand: results.rightHandLandmarks || null,
            });
        }
        return;
    }
    if (holisticMode !== "user" || !isWebcamActive || !canvasElement || !canvasCtx) return;

    resizeCanvasIfNeeded();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    drawCustomSkeleton(results);

    if (sentenceActive) {
        // "all" = combined whole-sentence scoring; a specific focused word
        // reuses the plain single-sign scorer (activePhaseModel already points
        // at that word - see focusOnWord) without touching the combined model's
        // saved progress (savedFocusStash, restored by resumeCombinedModel).
        if (sentenceViewMode === "all") analyzeSentenceFeedback(results);
        else analyzeFeedback(results);
    } else if (activeSign) {
        analyzeFeedback(results);
    }
}

// ---------------------------------------------------------------------------
// Reference priming: run the reference video through the model exactly once.
// ---------------------------------------------------------------------------
async function primeReference(signName) {
    const myId = ++activePrimeId;

    if (refCache[signName]) {
        refSequence = refCache[signName];
        activatePhaseModel(signName);
        return;
    }

    const precomputed = await precomputedPhasesPromise;
    if (myId !== activePrimeId) return; // superseded while the fetch was in flight
    if (precomputed && precomputed[signName]) {
        phaseCache[signName] = precomputed[signName];
        activatePhaseModel(signName);
        return;
    }

    const video = document.getElementById("practice-ref-video");
    if (!video) { refSequence = []; rawRefSequence = []; return; }

    ensureHolisticModel();
    await waitForVideoReady(video);
    if (myId !== activePrimeId) return; // superseded by a newer sign selection

    document.getElementById("coach-feedback-status").textContent = "Analyzing reference...";
    refSequence = [];
    rawRefSequence = [];
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
        phaseCache[signName] = buildPhaseModel(refSequence);
        activatePhaseModel(signName);
        // Return the video to a clean looping state for the learner to watch.
        video.loop = true;
        try { video.currentTime = 0; } catch (e) { /* ignore */ }
        // The "Analyzing reference..." status only gets set on this slow path,
        // so only this path needs to clear it back out.
        const status = document.getElementById("coach-feedback-status");
        if (status && status.textContent === "Analyzing reference...") status.textContent = "Idle";
    }
}

// Load a sign's phase model as the active target and reset per-attempt state.
function activatePhaseModel(signName) {
    activePhaseModel = phaseCache[signName] || (refCache[signName] ? buildPhaseModel(refCache[signName]) : null);
    if (activePhaseModel && !phaseCache[signName]) phaseCache[signName] = activePhaseModel;
    resetPhaseProgress();
}

function resetPhaseProgress() {
    const P = activePhaseModel ? activePhaseModel.holds.length : 0;
    phaseBest = new Array(P).fill(0);
    phaseReached = new Array(P).fill(false);
    phaseUserPose = new Array(P).fill(null);
    curPhase = 0;
    leftFinalPose = false;
    missingLimbFrames = 0;
    lastDisplayScore = 0;
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
                if (now - lastProcessTime >= USER_THROTTLE_MS) {
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

// Per-feature weight in the distance. Joint angles (finger bends, elbow,
// shoulder) are inherently invariant to camera position/zoom, so they carry the
// score. Palm orientation and hand location still shift with camera angle even
// after normalization, so they contribute discrimination at a lower weight
// rather than inflating the cost when the learner is framed differently.
const FEATURE_WEIGHTS = [
    1, 1, 1, 1, 1,   // 0-4  right finger bends
    0.4, 0.4, 0.2,   // 5-7  right palm normal x,y,z
    0.5,             // 8    right finger spread
    1, 1, 1, 1, 1,   // 9-13 left finger bends
    0.4, 0.4, 0.2,   // 14-16 left palm normal x,y,z
    0.5,             // 17   left finger spread
    1, 1,            // 18-19 elbows
    0.8, 0.8,        // 20-21 shoulders
    0.6, 0.6,        // 22-23 right wrist location x,y (face-relative)
    0.6, 0.6,        // 24-25 left wrist location x,y (face-relative)
];

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
    rawRefSequence = [];
    activePhaseModel = null;
    prevUserSmoothed = null;
    resetPhaseProgress();
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
    let weightSum = 0;
    for (const g of FEATURE_GROUPS) {
        if (g.needRight && !(uVis.rightHand && rVis.rightHand)) continue;
        if (g.needLeft && !(uVis.leftHand && rVis.leftHand)) continue;
        if (g.needPose && !(uVis.pose && rVis.pose)) continue;
        for (let i = g.start; i < g.end; i++) {
            const d = uf[i] - rf[i];
            const w = FEATURE_WEIGHTS[i];
            sum += w * d * d;
            weightSum += w;
        }
    }

    if (weightSum === 0) return 1.0;
    return Math.sqrt(sum / weightSum);
}

// Light per-feature EMA on the live user stream to damp landmark jitter.
// Resets whenever the visibility profile changes (don't blend across a hand
// appearing/disappearing, or you'd average real values with 0.5 defaults).
function smoothUserFrame(frame) {
    const a = 0.7; // weight on the newest frame; higher = more responsive, less lag
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

// Which required-limb prompt (if any) applies to this frame, or null if the
// learner is showing everything the active sign/word needs.
function requiredLimbsMissing(user, need) {
    const handsShown = (user.visibility.rightHand ? 1 : 0) + (user.visibility.leftHand ? 1 : 0);
    const missingPose = need.pose && !user.visibility.pose;
    if (handsShown >= need.hands && !missingPose) return null;
    if (missingPose) return "Step back so your head and upper body are in view.";
    if (need.hands >= 2) return "Show both hands to the camera for this sign.";
    return "Show your signing hand to the camera.";
}

// Score the user's current frame against activePhaseModel: advance curPhase,
// latch best-per-phase quality, and compute the aggregate score. Shared by
// single-sign practice (analyzeFeedback) and sentence practice
// (analyzeSentenceFeedback) so the phase state machine isn't duplicated.
function scoreActiveModel(user) {
    const holds = activePhaseModel.holds;
    const P = holds.length;

    // How well the user's current pose matches each hold (mirror-aware). `um` is
    // the user frame in whichever orientation matched, for feedback + move dir.
    const matches = holds.map((h) => matchHold(user, h));
    const q = matches.map((m) => m.q);

    // Progress on the phase currently being worked toward.
    phaseBest[curPhase] = Math.max(phaseBest[curPhase], q[curPhase]);
    if (q[curPhase] >= HOLD_COMPLETE_Q) {
        phaseReached[curPhase] = true;
        phaseUserPose[curPhase] = matches[curPhase].um.features.slice();
    }

    // Advance when this phase is reached and the next hold now matches at least
    // as well (the learner has moved on to the next pose).
    if (curPhase < P - 1 && phaseReached[curPhase] && q[curPhase + 1] >= q[curPhase] && q[curPhase + 1] > 0.2) {
        curPhase++;
        phaseBest[curPhase] = Math.max(phaseBest[curPhase], q[curPhase]);
        if (q[curPhase] >= HOLD_COMPLETE_Q) {
            phaseReached[curPhase] = true;
            phaseUserPose[curPhase] = matches[curPhase].um.features.slice();
        }
    }

    // Score = mean over phases of (best hold quality x movement-direction credit).
    // Completed phases keep their best (latched), so holding the final pose can't
    // make the score collapse; unreached phases contribute 0.
    let sum = 0;
    for (let p = 0; p < P; p++) {
        let contrib;
        if (p === curPhase) contrib = Math.max(phaseBest[p], q[p]);
        else if (phaseReached[p] || p < curPhase) contrib = phaseBest[p];
        else contrib = 0;
        sum += contrib * moveCredit(p);
    }
    const displayScore = Math.round(clamp01(sum / P) * 99);

    // Diagnose against the current phase's target (post-advance), in whichever
    // orientation matched.
    const target = holds[curPhase];
    const { idx: worstIdx, diff: worstDiff } = jointDiagnostic(matches[curPhase].um, target);

    return { P, q, matches, displayScore, allPhasesReached: phaseReached.every(Boolean), worstIdx, worstDiff };
}

function analyzeFeedback(results) {
    const user = smoothUserFrame(extractFrameFeatures(results));

    if (!activePhaseModel || activePhaseModel.holds.length === 0) {
        document.getElementById("coach-feedback-status").textContent = "Analyzing reference...";
        return;
    }

    // Prompt (debounced) if the sign needs limbs the learner isn't showing, and
    // freeze the score meanwhile so it can't be gamed by hiding a required hand.
    const need = activePhaseModel.requires;
    const limbMsg = requiredLimbsMissing(user, need);
    if (limbMsg) {
        missingLimbFrames++;
        if (missingLimbFrames >= 3) {
            const P = activePhaseModel.holds.length;
            document.getElementById("coach-feedback-status").textContent = "Show required limbs";
            document.getElementById("coach-feedback-message").textContent = limbMsg;
            document.getElementById("coach-score-display").textContent = `${lastDisplayScore}%`;
            speakFeedback(limbMsg);
            updateCoachDebug({ P, curPhase, q: activePhaseModel.holds.map(() => 0), displayScore: lastDisplayScore, worstIdx: -1, worstDiff: 0, prompt: limbMsg });
            return;
        }
    } else {
        missingLimbFrames = 0;
    }

    const { P, q, matches, displayScore, allPhasesReached: done, worstIdx, worstDiff } = scoreActiveModel(user);
    lastDisplayScore = displayScore;

    // Attempt restart: only after the learner FINISHED, then moved off the final
    // pose, then returned to the start pose (avoids false restarts on signs whose
    // start and end look alike). Single-sign practice only - a sentence session
    // advances to the next WORD instead of restarting the same one.
    if (P > 1 && phaseReached[P - 1]) {
        if (q[P - 1] < 0.35) leftFinalPose = true;
        if (leftFinalPose && q[0] >= HOLD_COMPLETE_Q) {
            resetPhaseProgress();
            phaseBest[0] = q[0];
            phaseReached[0] = true;
            phaseUserPose[0] = matches[0].um.features.slice();
        }
    }

    // Feedback focuses on the current phase (diagnose against the matched orientation).
    let feedback;
    if (done && displayScore >= 75) {
        feedback = `All ${P} phase${P > 1 ? "s" : ""} matched — excellent!`;
        speakFeedback("Excellent! You matched the whole sign.");
    } else if (q[curPhase] >= HOLD_COMPLETE_Q) {
        feedback = P > 1 ? `Phase ${curPhase + 1}/${P} matched — move to the next pose.` : "Pose matched — hold it!";
        speakFeedback(P > 1 ? "Good. Now the next pose." : "Pose matched.");
    } else if (worstIdx >= 0 && worstDiff > 0.18) {
        feedback = `Phase ${curPhase + 1}/${P}: ${FEATURE_FEEDBACK[worstIdx]}`;
        speakFeedback(FEATURE_FEEDBACK[worstIdx]);
    } else {
        feedback = `Move into phase ${curPhase + 1} of ${P} as shown in the reference.`;
        speakFeedback("Match the next pose in the sign.");
    }

    document.getElementById("coach-feedback-status").textContent = `Score: ${displayScore}%`;
    document.getElementById("coach-feedback-message").textContent = feedback;
    document.getElementById("coach-score-display").textContent = `${displayScore}%`;

    updateCoachDebug({ P, curPhase, q, displayScore, worstIdx, worstDiff });
}

// Concatenates every word's precomputed holds into one ordered phase
// sequence, so the whole sentence scores through scoreActiveModel exactly
// like a single multi-phase sign. wordPhaseRanges[i] = [start, end) into the
// combined holds array for sentenceGloss[i], used to map a phase index back
// to "which word is this" (currentWordIndex) and to per-word done-ness
// (isWordDone) for the gloss-strip UI.
function buildCombinedPhaseModel(words) {
    const holds = [];
    const moveDirs = [];
    const holdTimes = [];
    let handsMax = 0;
    let poseNeeded = false;
    const wordPhaseRanges = [];

    for (const word of words) {
        const model = phaseCache[word];
        const start = holds.length;
        if (model && model.holds && model.holds.length) {
            holds.push(...model.holds);
            moveDirs.push(...model.moveDirs);
            holdTimes.push(...model.holdTimes);
            handsMax = Math.max(handsMax, model.requires.hands);
            poseNeeded = poseNeeded || model.requires.pose;
        }
        wordPhaseRanges.push([start, holds.length]);
    }

    return { holds, moveDirs, holdTimes, requires: { hands: handsMax, pose: poseNeeded }, wordPhaseRanges };
}

// Which sentenceGloss word the combined model's curPhase currently belongs
// to - derived from curPhase rather than tracked as separate state, so it's
// never at risk of drifting out of sync with the actual phase progress.
function currentWordIndex() {
    if (!sentenceCombinedModel) return 0;
    const ranges = sentenceCombinedModel.wordPhaseRanges;
    for (let i = 0; i < ranges.length; i++) {
        if (curPhase >= ranges[i][0] && curPhase < ranges[i][1]) return i;
    }
    return ranges.length - 1; // curPhase past the last range - sentence is done, report the last word
}

function isWordDone(i) {
    if (!sentenceCombinedModel) return false;
    const [start, end] = sentenceCombinedModel.wordPhaseRanges[i];
    for (let p = start; p < end; p++) {
        if (!phaseReached[p]) return false;
    }
    return end > start;
}

// ---------------------------------------------------------------------------
// Sentence practice: sign a full sentence in the correct ISL gloss order.
// Graded as one combined multi-phase model (buildCombinedPhaseModel) via the
// same phase state machine single-sign practice uses - order is enforced
// structurally (phases only advance in sequence), not just suggested. On top
// of that, every frame is ALSO classified against every OTHER word in the
// sentence (nearest-neighbor over the same matchHold/pose-distance function,
// scoped to this sentence's own words rather than the full 262-sign
// dictionary) so a learner performing the wrong word gets told what it
// looked like instead of a generic "keep trying". Word phase models come
// straight from the precomputed phases.json fetch (`precomputedPhasesPromise`)
// - unlike primeReference()'s per-sign lazy caching, we need every gloss word
// available at once, so they're copied into `phaseCache` up front here rather
// than one at a time - so no live priming happens between words.
// ---------------------------------------------------------------------------
async function beginSentenceSession(englishText, glossSignNames) {
    const precomputed = await precomputedPhasesPromise;

    sentenceGloss = glossSignNames.map((w) => w.toLowerCase());
    for (const word of sentenceGloss) {
        if (!phaseCache[word] && precomputed && precomputed[word]) {
            phaseCache[word] = precomputed[word];
        }
    }
    misclassifyStreak = 0;
    lastMisclassifiedWord = null;
    sentenceActive = true;

    sentenceCombinedModel = buildCombinedPhaseModel(sentenceGloss);
    savedFocusStash = null;
    activePhaseModel = sentenceCombinedModel;
    resetPhaseProgress();

    renderGlossStrip();
    applyViewMode();

    document.getElementById("coach-feedback-status").textContent = "Ready";
    document.getElementById("coach-feedback-message").textContent =
        `Sign "${sentenceGloss[0]}" first (word 1 of ${sentenceGloss.length}).`;
}

function endSentenceSession() {
    sentenceActive = false;
    sentenceGloss = [];
    sentenceCombinedModel = null;
    savedFocusStash = null;
    misclassifyStreak = 0;
    lastMisclassifiedWord = null;
    sentenceViewMode = "all"; // next sentence session always starts on the default view

    const strip = document.getElementById("sentence-gloss-strip");
    if (strip) {
        strip.style.display = "none";
        strip.innerHTML = "";
    }

    const video = document.getElementById("practice-ref-video");
    if (video) video.onended = null;

    const prefetch = document.getElementById("practice-ref-video-prefetch");
    if (prefetch) prefetch.removeAttribute("src");

    const gallery = document.getElementById("phase-frames");
    if (gallery) {
        gallery.style.display = "none";
        gallery.innerHTML = "";
    }
}

// Sentence-mode view selector - chips rendered into #sentence-gloss-strip (see
// renderGlossStrip): "all" (default) or a specific gloss word. Selecting a
// word switches the coach into plain single-sign scoring for THAT word alone
// (activePhaseModel repointed via focusOnWord, which - like single-sign
// practice - resets the shared phaseBest/phaseReached/curPhase state). Since
// that state is shared with the combined model's own progress, entering focus
// mode first stashes it (savedFocusStash) so free-practicing a word out of
// order can't corrupt the order-gated "all" progress; switching back to "all"
// (resumeCombinedModel) restores it, resuming exactly where it was.
function setSentenceViewMode(mode) {
    sentenceViewMode = mode;
    renderGlossStrip();
    if (!sentenceActive) return;

    if (mode === "all") {
        resumeCombinedModel();
    } else {
        if (!savedFocusStash) {
            savedFocusStash = { phaseBest, phaseReached, phaseUserPose, curPhase };
        }
        focusOnWord(mode); // repoint activePhaseModel at the selected word
    }
    applyViewMode();
}

// Drives the reference-video display + phase-frame gallery for whichever mode
// is active. Scoring state (activePhaseModel etc.) is set separately by
// resumeCombinedModel()/focusOnWord() before this is called.
function applyViewMode() {
    if (sentenceViewMode === "all") {
        playCombinedVideo();
        showSentencePhaseFrames();
    } else {
        loadWordVideo(sentenceViewMode);
        showWordPhaseFrames(sentenceViewMode);
    }
}

// Point the active phase model back at the whole-sentence combined model and
// restore its progress if a focus-word excursion stashed it (see
// setSentenceViewMode). Video display is handled separately by
// applyViewMode()/playCombinedVideo(), not here.
function resumeCombinedModel() {
    activePhaseModel = sentenceCombinedModel;
    if (savedFocusStash) {
        ({ phaseBest, phaseReached, phaseUserPose, curPhase } = savedFocusStash);
        savedFocusStash = null;
    }
}

// Point the active phase model at an arbitrary gloss word for free practice,
// independent of the combined model's order-gating (see setSentenceViewMode).
function focusOnWord(word) {
    activePhaseModel = phaseCache[word] || null;
    resetPhaseProgress();
}

// Single-word view mode: the reference video shows just that one word's clip,
// looping, like ordinary single-sign practice.
function loadWordVideo(word) {
    const video = document.getElementById("practice-ref-video");
    const placeholder = document.getElementById("video-placeholder");
    if (!video) return;
    video.pause();
    video.loop = true;
    video.onended = null; // clear any leftover chain handler from "all" mode
    // No crossOrigin: see the same note in app.js's startPractice - every sign
    // has a precomputed phase model, so nothing reads canvas frames off this
    // video anymore, and setting it would break playback on CORS-less hosts.
    video.onloadeddata = () => {
        if (placeholder) placeholder.style.display = "none";
        video.style.display = "block";
        video.play().catch(() => {});
    };
    video.src = `${VIDEO_BASE_URL}/${encodeURIComponent(word)}.mp4`;
    video.load();
}

// "All" view mode: chain every gloss word's reference clip into one continuous
// looping playback - dormant pose (each clip's own natural rest bookend) into
// sign 1, sign 2, ... back to a dormant pose, then wraps to sign 1 again -
// independent of the learner's scoring progress (activePhaseModel keeps
// tracking exactly as before via resumeCombinedModel/beginSentenceSession).
function playCombinedVideo() {
    const video = document.getElementById("practice-ref-video");
    if (!video || sentenceGloss.length === 0) return;
    videoChainIdx = 0;
    video.loop = false;
    video.onended = handleChainedVideoEnded;
    loadChainedWord(videoChainIdx);
}

// Loads word[idx] into the visible video AND kicks off prefetchNextChainedWord
// so the FOLLOWING word is already warmed in the browser's cache by the time
// it's needed - see the #practice-ref-video-prefetch comment in index.html.
function loadChainedWord(idx) {
    const video = document.getElementById("practice-ref-video");
    const placeholder = document.getElementById("video-placeholder");
    if (!video) return;
    const word = sentenceGloss[idx];
    video.pause();
    // No crossOrigin: see the same note in app.js's startPractice.
    video.onloadeddata = () => {
        if (placeholder) placeholder.style.display = "none";
        video.style.display = "block";
        video.play().catch(() => {});
        prefetchNextChainedWord(idx);
    };
    video.src = `${VIDEO_BASE_URL}/${encodeURIComponent(word)}.mp4`;
    video.load();
}

// Warms the browser's HTTP cache for whatever chain word comes after `idx`
// (wrapping back to word 0, same as handleChainedVideoEnded) by loading it
// into a hidden, never-played video element while the current clip is still
// playing. That gives the whole rest of the current clip's duration for the
// fetch to finish, so when handleChainedVideoEnded later sets the VISIBLE
// video's src to that same URL, the browser serves it from cache instead of
// starting a fresh network fetch - eliminating the black gap between clips.
function prefetchNextChainedWord(idx) {
    const pre = document.getElementById("practice-ref-video-prefetch");
    if (!pre || sentenceGloss.length === 0) return;
    const nextWord = sentenceGloss[(idx + 1) % sentenceGloss.length];
    pre.src = `${VIDEO_BASE_URL}/${encodeURIComponent(nextWord)}.mp4`;
    pre.load();
}

function handleChainedVideoEnded() {
    if (sentenceViewMode !== "all" || !sentenceActive || sentenceGloss.length === 0) return;
    videoChainIdx = (videoChainIdx + 1) % sentenceGloss.length;
    loadChainedWord(videoChainIdx);
}

// Every chip doubles as a view-mode selector (setSentenceViewMode): word
// chips still show sentence progress (green once done) but ALSO carry
// "view-active" (a ring, not a color override - see .gloss-chip.view-active)
// when that word is the focused view, so progress state stays visible
// alongside the selection. The "All" chip is plain except for that same ring.
// Unreached words are left unstyled (no "current" highlight) so they look
// like ordinary chips rather than implying one is more clickable than another.
function renderGlossStrip() {
    const strip = document.getElementById("sentence-gloss-strip");
    if (!strip) return;
    strip.style.display = "flex";
    const wordChips = sentenceGloss
        .map((word, i) => {
            const progressCls = isWordDone(i) ? "done" : "";
            const selectedCls = sentenceViewMode === word ? "view-active" : "";
            return `<span class="gloss-chip ${progressCls} ${selectedCls}" style="cursor:pointer;" onclick="setSentenceViewMode('${word}')" title="Focus this word: its own video, phase frames, and camera check">${i + 1}. ${word}</span>`;
        })
        .join("");
    const viewChip = `<span class="gloss-chip ${sentenceViewMode === "all" ? "view-active" : ""}" style="margin-left:auto; cursor:pointer;" onclick="setSentenceViewMode('all')" title="Combined video, phase frames, and camera check for the whole sentence">All</span>`;
    strip.innerHTML = wordChips + viewChip;
}

function analyzeSentenceFeedback(results) {
    // Sentence already complete (every phase of the combined model reached) -
    // leave the summary showing instead of re-scoring.
    if (phaseReached.length && phaseReached.every(Boolean)) return;

    const user = smoothUserFrame(extractFrameFeatures(results));

    if (!activePhaseModel || activePhaseModel.holds.length === 0) {
        document.getElementById("coach-feedback-status").textContent = "Loading word...";
        return;
    }

    const need = activePhaseModel.requires;
    const limbMsg = requiredLimbsMissing(user, need);
    if (limbMsg) {
        missingLimbFrames++;
        if (missingLimbFrames >= 3) {
            document.getElementById("coach-feedback-status").textContent = "Show required limbs";
            document.getElementById("coach-feedback-message").textContent = limbMsg;
            speakFeedback(limbMsg);
            return;
        }
    } else {
        missingLimbFrames = 0;
    }

    const result = scoreActiveModel(user);
    lastDisplayScore = result.displayScore;
    const curIdx = currentWordIndex();

    // Classification pass: does the learner look more like a DIFFERENT gloss
    // word than the one currently expected? Nearest-neighbor over each
    // candidate word's own precomputed holds, scoped to this sentence's words.
    let bestWord = sentenceGloss[curIdx];
    let bestQ = Math.max(...result.q, 0);
    for (let i = 0; i < sentenceGloss.length; i++) {
        if (i === curIdx) continue;
        const model = phaseCache[sentenceGloss[i]];
        if (!model || !model.holds.length) continue;
        const q = Math.max(...model.holds.map((h) => matchHold(user, h).q));
        if (q > bestQ + 0.15) {
            bestQ = q;
            bestWord = sentenceGloss[i];
        }
    }
    if (bestWord !== sentenceGloss[curIdx]) {
        misclassifyStreak++;
        lastMisclassifiedWord = bestWord;
    } else {
        misclassifyStreak = 0;
        lastMisclassifiedWord = null;
    }

    if (misclassifyStreak >= 3 && lastMisclassifiedWord) {
        const idx = sentenceGloss.indexOf(lastMisclassifiedWord);
        const where = idx < curIdx ? "you already signed that" : "that comes later in this sentence";
        const msg = `That looked like "${lastMisclassifiedWord}" — ${where}. We're on "${sentenceGloss[curIdx]}" now.`;
        document.getElementById("coach-feedback-status").textContent = "Out of order?";
        document.getElementById("coach-feedback-message").textContent = msg;
        document.getElementById("coach-score-display").textContent = `${result.displayScore}%`;
        speakFeedback(msg);
    } else {
        document.getElementById("coach-feedback-status").textContent = `Score: ${result.displayScore}%`;
        document.getElementById("coach-feedback-message").textContent = result.allPhasesReached
            ? "Sentence complete!"
            : `Sign "${sentenceGloss[curIdx]}" (word ${curIdx + 1}/${sentenceGloss.length}).`;
        document.getElementById("coach-score-display").textContent = `${result.displayScore}%`;
    }

    updateSentenceDebug(result, bestWord, bestQ, curIdx);
    renderGlossStrip(); // isWordDone() reads live phaseReached, so this can change any frame, not just on completion

    if (result.allPhasesReached) {
        finishSentenceSession(result.displayScore);
    }
}

// finalScore is the combined model's own displayScore at the moment every
// phase was reached - same aggregate-over-phases computation single-sign
// practice uses (scoreActiveModel), just fed more phases (every word's, not
// one), so it's already "the sentence score" with no separate averaging step.
function finishSentenceSession(finalScore) {
    document.getElementById("coach-feedback-status").textContent = "Sentence complete!";
    document.getElementById("coach-feedback-message").textContent =
        `All ${sentenceGloss.length} words matched — score ${finalScore}%.`;
    document.getElementById("coach-score-display").textContent = `${finalScore}%`;
    speakFeedback("Great job! You signed the whole sentence.");
}

function updateSentenceDebug(result, bestWord, bestQ, curIdx) {
    if (!coachDebug) return;
    const el = document.getElementById("coach-debug");
    if (!el) return;
    el.textContent =
        `word ${curIdx + 1}/${sentenceGloss.length} "${sentenceGloss[curIdx]}" | phase ${curPhase + 1}/${result.P} | ` +
        `score ${result.displayScore}% | classifier top match: ${bestWord} (${Math.round(bestQ * 100)}%)`;
}

// Match of a user frame to a hold target pose, considering BOTH orientations so
// scoring is robust to the mirrored selfie view (and to learners who mirror the
// sign / use their non-dominant hand). Returns the quality (0..1) and the user
// frame in whichever orientation matched better (for feedback + move direction).
function matchHold(user, hold) {
    const dDirect = getMaskedVectorDistance(user, hold);
    const mUser = mirrorFrame(user);
    const dMirror = getMaskedVectorDistance(mUser, hold);
    if (dMirror < dDirect) {
        return { q: clamp01(1 - dMirror / HOLD_MATCH_THRESHOLD), um: mUser };
    }
    return { q: clamp01(1 - dDirect / HOLD_MATCH_THRESHOLD), um: user };
}

function holdQuality(user, hold) {
    return matchHold(user, hold).q;
}

// Left-right mirror of a frame: swap the hand blocks, swap elbows/shoulders, and
// flip the x of every horizontal feature (palm-normal x, wrist-location x).
function mirrorFrame(frame) {
    const f = frame.features;
    const m = f.slice();
    for (let i = 0; i < 9; i++) { m[i] = f[9 + i]; m[9 + i] = f[i]; }     // right<->left hand blocks
    m[5] = 1 - m[5]; m[14] = 1 - m[14];                                     // palm-normal x (post-swap)
    m[18] = f[19]; m[19] = f[18];                                           // elbows
    m[20] = f[21]; m[21] = f[20];                                           // shoulders
    m[22] = 1 - f[24]; m[23] = f[25]; m[24] = 1 - f[22]; m[25] = f[23];     // wrist location (swap + flip x)
    return {
        features: m,
        visibility: { rightHand: frame.visibility.leftHand, leftHand: frame.visibility.rightHand, pose: frame.visibility.pose },
    };
}

// Credit (<=1) for having moved INTO phase p in the reference's direction.
function moveCredit(p) {
    if (p === 0 || !activePhaseModel) return 1;
    const dir = activePhaseModel.moveDirs[p];
    if (!dir) return 1;
    const a = phaseUserPose[p - 1];
    const b = phaseUserPose[p];
    if (!a || !b) return 1;
    const cos = cosineSim(positionDelta(a, b), dir);
    return (1 - MOVE_WEIGHT) + MOVE_WEIGHT * clamp01(cos);
}

function jointDiagnostic(user, target) {
    let maxDiff = 0, idx = -1;
    for (let k = 0; k < FEATURE_DIM; k++) {
        if (!featureVisible(k, user.visibility, target.visibility)) continue;
        const d = Math.abs(user.features[k] - target.features[k]);
        if (d > maxDiff) { maxDiff = d; idx = k; }
    }
    return { idx, diff: maxDiff };
}

function setCoachDebug(on) {
    coachDebug = on;
    const el = document.getElementById("coach-debug");
    if (el && !on) el.style.display = "none";
}

// [TEMP/DEBUG] Render the reference frame at each detected phase (hold) so the
// segmentation can be eyeballed. Seeks the ref video to each hold's timestamp
// and captures a thumbnail. Triggered by the "Show Phase Frames" button.
async function showPhaseFrames() {
    if (sentenceActive) {
        return sentenceViewMode === "all" ? showSentencePhaseFrames() : showWordPhaseFrames(sentenceViewMode);
    }

    const gallery = document.getElementById("phase-frames");
    if (!gallery) return;
    gallery.style.display = "flex";

    const signName = (typeof activeSign !== "undefined" && activeSign) ? activeSign.sign_name.toLowerCase() : null;
    const model = signName && phaseCache[signName];
    if (!model || !model.holdTimes || !model.holdTimes.length) {
        gallery.innerHTML = `<span style="color:var(--text-secondary)">No phase model yet — open a sign and let the reference finish analyzing.</span>`;
        return;
    }
    const video = document.getElementById("practice-ref-video");
    if (!video || !video.duration) {
        gallery.innerHTML = `<span style="color:var(--text-secondary)">Reference video not ready.</span>`;
        return;
    }

    const wasPaused = video.paused, wasLoop = video.loop, t0 = video.currentTime;
    video.pause();
    video.loop = false;
    gallery.innerHTML = `<span style="color:var(--text-secondary)">Capturing ${model.holdTimes.length} phase(s)…</span>`;

    const cnv = document.createElement("canvas");
    cnv.width = 160; cnv.height = 120;
    const ctx = cnv.getContext("2d");
    const figs = [];
    try {
        for (let i = 0; i < model.holdTimes.length; i++) {
            const t = Math.min(Math.max(model.holdTimes[i], 0), video.duration - 0.01);
            await seekVideo(video, t);
            ctx.drawImage(video, 0, 0, cnv.width, cnv.height);
            const need = model.requires;
            figs.push(`<div style="text-align:center;font-size:0.72rem;color:var(--text-secondary);">
                <img src="${cnv.toDataURL("image/jpeg", 0.7)}" style="width:160px;height:120px;object-fit:cover;border-radius:8px;border:1px solid var(--border-color);">
                <div style="margin-top:0.25rem;">Phase ${i + 1}/${model.holdTimes.length} · t=${t.toFixed(2)}s</div>
            </div>`);
            if (i === 0) figs.unshift(`<div style="align-self:center;font-size:0.72rem;color:var(--text-secondary);padding-right:0.5rem;">needs ${need.hands} hand(s)${need.pose ? " + body" : ""}:</div>`);
        }
        gallery.innerHTML = figs.join("");
    } catch (e) {
        // Tainted-canvas SecurityError: the video's host doesn't send CORS
        // headers, so pixels can't be read back for a thumbnail. Debug-only
        // feature - degrade to a message instead of an uncaught exception.
        gallery.innerHTML = `<span style="color:var(--text-secondary)">Can't capture thumbnails - reference video host doesn't allow cross-origin pixel reads.</span>`;
    }

    video.loop = wasLoop;
    try { video.currentTime = t0; } catch (e) { /* ignore */ }
    if (!wasPaused) video.play().catch(() => {});
}

// Loads `word`'s reference clip into an offscreen video (doesn't touch the
// visible #practice-ref-video, which the combined chain or focus mode owns
// independently) and captures a thumbnail at each of its phase model's hold
// timestamps. Shared by the combined ("all") and single-word galleries below.
async function captureWordPhaseThumbnails(word, ctx, cnv) {
    const model = phaseCache[word];
    if (!model || !model.holdTimes || !model.holdTimes.length) return null;

    const tempVideo = document.createElement("video");
    tempVideo.crossOrigin = "anonymous";
    tempVideo.muted = true;
    tempVideo.playsInline = true;
    tempVideo.src = `${VIDEO_BASE_URL}/${encodeURIComponent(word)}.mp4`;
    await new Promise((resolve) => {
        tempVideo.onloadeddata = resolve;
        tempVideo.onerror = resolve;
        tempVideo.load();
    });
    if (!tempVideo.duration) return null;

    const frames = [];
    for (let i = 0; i < model.holdTimes.length; i++) {
        const t = Math.min(Math.max(model.holdTimes[i], 0), tempVideo.duration - 0.01);
        await seekVideo(tempVideo, t);
        ctx.drawImage(tempVideo, 0, 0, cnv.width, cnv.height);
        frames.push(cnv.toDataURL("image/jpeg", 0.7));
    }
    return { model, frames };
}

// "All" view mode's combined phase-frame gallery: every gloss word's hold
// thumbnails, in sentence order.
async function showSentencePhaseFrames() {
    const gallery = document.getElementById("phase-frames");
    if (!gallery) return;
    gallery.style.display = "flex";

    if (!sentenceGloss.length) {
        gallery.innerHTML = `<span style="color:var(--text-secondary)">No sentence loaded.</span>`;
        return;
    }
    gallery.innerHTML = `<span style="color:var(--text-secondary)">Capturing phase frames for ${sentenceGloss.length} word(s)…</span>`;

    const cnv = document.createElement("canvas");
    cnv.width = 160; cnv.height = 120;
    const ctx = cnv.getContext("2d");
    const figs = [];

    for (let w = 0; w < sentenceGloss.length; w++) {
        const word = sentenceGloss[w];
        const result = await captureWordPhaseThumbnails(word, ctx, cnv);
        if (!result) continue;

        figs.push(`<div style="align-self:center;font-size:0.78rem;font-weight:600;color:var(--text-main);padding:0 0.35rem;">${w + 1}. ${word}</div>`);
        result.frames.forEach((dataUrl, i) => {
            figs.push(`<div style="text-align:center;font-size:0.72rem;color:var(--text-secondary);">
                <img src="${dataUrl}" style="width:160px;height:120px;object-fit:cover;border-radius:8px;border:1px solid var(--border-color);">
                <div style="margin-top:0.25rem;">Phase ${i + 1}/${result.frames.length}</div>
            </div>`);
        });
    }
    gallery.innerHTML = figs.join("") || `<span style="color:var(--text-secondary)">No phase models available yet.</span>`;
}

// Single-word view mode's phase-frame gallery: just the focused word's holds.
async function showWordPhaseFrames(word) {
    const gallery = document.getElementById("phase-frames");
    if (!gallery) return;
    gallery.style.display = "flex";
    gallery.innerHTML = `<span style="color:var(--text-secondary)">Capturing phase frames for "${word}"…</span>`;

    const cnv = document.createElement("canvas");
    cnv.width = 160; cnv.height = 120;
    const ctx = cnv.getContext("2d");
    const result = await captureWordPhaseThumbnails(word, ctx, cnv);
    if (!result) {
        gallery.innerHTML = `<span style="color:var(--text-secondary)">No phase model yet for "${word}".</span>`;
        return;
    }

    const need = result.model.requires;
    const figs = [`<div style="align-self:center;font-size:0.72rem;color:var(--text-secondary);padding-right:0.5rem;">needs ${need.hands} hand(s)${need.pose ? " + body" : ""}:</div>`];
    result.frames.forEach((dataUrl, i) => {
        figs.push(`<div style="text-align:center;font-size:0.72rem;color:var(--text-secondary);">
            <img src="${dataUrl}" style="width:160px;height:120px;object-fit:cover;border-radius:8px;border:1px solid var(--border-color);">
            <div style="margin-top:0.25rem;">Phase ${i + 1}/${result.frames.length}</div>
        </div>`);
    });
    gallery.innerHTML = figs.join("");
}

// Live scoring breakdown so calibration can be data-driven against a real camera.
function updateCoachDebug(d) {
    const el = document.getElementById("coach-debug");
    if (!el) return;
    if (!coachDebug) { el.style.display = "none"; return; }
    el.style.display = "block";
    const qs = d.q.map((x) => x.toFixed(2)).join(",");
    const bests = phaseBest.map((x) => x.toFixed(2)).join(",");
    const reached = phaseReached.map((x) => (x ? "✓" : "·")).join("");
    const worst = d.worstIdx >= 0 ? `${d.worstIdx} Δ${d.worstDiff.toFixed(2)}` : "none";
    el.textContent =
        `phases ${d.P} | on ${d.curPhase + 1}/${d.P} | q[${qs}] | best[${bests}] | reached ${reached} | ` +
        `score ${d.displayScore}% | worst joint: ${worst}`;
}

// ---------------------------------------------------------------------------
// Reference phase model: auto-segment a reference sequence into an ordered list
// of HOLD target poses (low-motion runs) joined by MOVES (with a direction),
// using the Movement-Hold structure of the sign.
// ---------------------------------------------------------------------------
function buildPhaseModel(seq) {
    const n = seq.length;
    const dt = 1 / SAMPLE_FPS; // seconds per reference frame (seeking step)
    const pack = (holds, moveDirs, times) => ({ holds, moveDirs, holdTimes: times, requires: requiredLimbs(seq, holds) });
    if (n === 0) return pack([], [], []);
    if (n <= 2) return pack([avgFrames(seq)], [null], [((n - 1) / 2) * dt]);

    // A hand tracked through most of the clip is "in play" for this sign.
    // frameDistance masks out a hand wherever either frame can't see it, so a
    // brief tracking dropout (common near the face) reads as near-zero motion
    // even while the real gesture is mid-swing - a false hold. Gaps where an
    // in-play hand drops out in either frame are never eligible to seed or
    // extend a hold; they're always treated as a MOVE.
    const handInPlay = {
        right: seq.filter((f) => f.visibility.rightHand).length / n >= 0.5,
        left: seq.filter((f) => f.visibility.leftHand).length / n >= 0.5,
    };
    const gapEligible = (a, b) =>
        !(handInPlay.right && (!a.visibility.rightHand || !b.visibility.rightHand)) &&
        !(handInPlay.left && (!a.visibility.leftHand || !b.visibility.leftHand));

    // Motion between consecutive frames (peak is measured only over eligible
    // gaps, so a dropout-inflated spike can't distort the hold threshold).
    const motion = [];
    const eligible = [];
    for (let i = 1; i < n; i++) {
        motion.push(frameDistance(seq[i], seq[i - 1]));
        eligible.push(gapEligible(seq[i - 1], seq[i]));
    }
    const eligibleMotions = motion.filter((_, i) => eligible[i]);
    const maxM = eligibleMotions.length ? Math.max(...eligibleMotions) : Math.max(...motion);
    const thr = Math.max(REF_MIN_HOLD_MOTION, maxM * REF_MOTION_HOLD_FRAC);

    // A HOLD is a run of consecutive eligible LOW-motion gaps (a sustained,
    // fully-tracked pose); everything else is a MOVE we don't score.
    // Each hold run over gaps [start..g-1] connects frames [start..g].
    let segs = [];
    let g = 0;
    while (g < motion.length) {
        if (eligible[g] && motion[g] < thr) {
            const start = g;
            while (g < motion.length && eligible[g] && motion[g] < thr) g++;
            segs.push({ pose: avgFrames(seq.slice(start, g + 1)), start, end: g });
        } else {
            g++;
        }
    }
    // Best available proxy for "the sign's content" when segmentation can't
    // find (or trimming leaves) any non-resting pose: the single frame where
    // the in-play hand(s) are most raised, rather than a rest-adjacent frame.
    const raisedFrameFallback = () => {
        let bestIdx = 0, bestY = Infinity;
        for (let i = 0; i < n; i++) {
            const f = seq[i];
            const y = Math.min(
                handInPlay.right && f.visibility.rightHand ? f.features[23] : 1,
                handInPlay.left && f.visibility.leftHand ? f.features[25] : 1
            );
            if (y < bestY) { bestY = y; bestIdx = i; }
        }
        const winStart = Math.max(0, bestIdx - 1), winEnd = Math.min(n - 1, bestIdx + 1);
        return { pose: avgFrames(seq.slice(winStart, winEnd + 1)), start: winStart, end: winEnd };
    };

    // Nothing segmented cleanly (brief/fast sign, or dropouts masked every
    // genuine pause).
    if (segs.length === 0) segs = [raisedFrameFallback()];

    // Merge consecutive holds whose target poses are nearly identical.
    const merged = [segs[0]];
    for (let k = 1; k < segs.length; k++) {
        const prev = merged[merged.length - 1];
        if (frameDistance(segs[k].pose, prev.pose) < PHASE_MERGE_DIST) {
            merged[merged.length - 1] = { pose: avgFrames([prev.pose, segs[k].pose]), start: prev.start, end: segs[k].end };
        } else {
            merged.push(segs[k]);
        }
    }

    let holds = merged.map((m) => m.pose);
    let times = merged.map((m) => ((m.start + m.end) / 2) * dt);

    // Drop idle/dormant bookend holds — the neutral resting pose the signer
    // settles into before and after the sign (hands down). Only leading/
    // trailing holds are trimmed, so a real hold in the middle survives.
    const dormant = (h) => Math.min(h.features[23], h.features[25]) > REST_FLOOR; // hands hanging down
    if (holds.length > 1) {
        while (holds.length > 1 && dormant(holds[0])) { holds.shift(); times.shift(); }
        while (holds.length > 1 && dormant(holds[holds.length - 1])) { holds.pop(); times.pop(); }
    }
    // The clip's only genuine low-motion run was itself a resting pose (e.g. a
    // sign whose real gesture is continuous hand-shape motion with no static
    // pause, so the only "hold" segmentation can find is before/after it) -
    // as uninformative as finding no hold at all, so fall back the same way.
    if (holds.every(dormant)) {
        const fb = raisedFrameFallback();
        holds = [fb.pose];
        times = [((fb.start + fb.end) / 2) * dt];
    }

    // Dominant movement direction into each hold (over position/orientation dims).
    const moveDirs = holds.map((h, p) => {
        if (p === 0) return null;
        const dir = positionDelta(holds[p - 1].features, holds[p].features);
        const mag = Math.sqrt(dir.reduce((s, v) => s + v * v, 0));
        if (mag < 0.08) return null; // negligible move -> no direction constraint
        return dir.map((v) => v / mag);
    });

    return pack(holds, moveDirs, times);
}

// What the sign needs the learner to show. A hand counts as REQUIRED only if it
// is actually used — visible in most frames AND either moving or raised into the
// signing space — so a merely-resting hand in the reference doesn't force the
// learner to show a second hand (the visibility != usage pitfall).
function requiredLimbs(seq, holds) {
    // A hand is USED if it's visible in most frames AND its WRIST either moves
    // through space or is held up in the signing area. Wrist location is used
    // (not finger angles) because a resting hand's low-confidence finger
    // landmarks jitter and would falsely read as "active".
    const handUsed = (visKey, xIdx, yIdx) => {
        const vis = seq.filter((f) => f.visibility[visKey]);
        if (!seq.length || vis.length / seq.length < 0.5) return false;
        let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity, sumY = 0;
        for (const f of vis) {
            mnx = Math.min(mnx, f.features[xIdx]); mxx = Math.max(mxx, f.features[xIdx]);
            mny = Math.min(mny, f.features[yIdx]); mxy = Math.max(mxy, f.features[yIdx]);
            sumY += f.features[yIdx];
        }
        const locRange = Math.max(mxx - mnx, mxy - mny);
        const meanY = sumY / vis.length;
        return locRange >= 0.12 || meanY < 0.85; // moves in space OR raised (not resting at the bottom)
    };
    const hands = (handUsed("rightHand", 22, 23) ? 1 : 0) + (handUsed("leftHand", 24, 25) ? 1 : 0);

    let pose = 0;
    for (const h of holds) if (h.visibility.pose) pose++;
    return { hands, pose: holds.length > 0 && pose * 2 >= holds.length };
}

// Average a set of frames into one representative pose (majority-vote visibility).
function avgFrames(frames) {
    const f = new Array(FEATURE_DIM).fill(0);
    let r = 0, l = 0, p = 0;
    for (const fr of frames) {
        for (let i = 0; i < FEATURE_DIM; i++) f[i] += fr.features[i];
        if (fr.visibility.rightHand) r++;
        if (fr.visibility.leftHand) l++;
        if (fr.visibility.pose) p++;
    }
    const c = frames.length || 1;
    for (let i = 0; i < FEATURE_DIM; i++) f[i] /= c;
    return { features: f, visibility: { rightHand: r * 2 >= c, leftHand: l * 2 >= c, pose: p * 2 >= c } };
}

// Weighted masked distance between two frames WITHOUT the face-only anti-cheat
// (used for reference-side segmentation, where "no hands" just means no change).
function frameDistance(a, b) {
    let s = 0, w = 0;
    for (const g of FEATURE_GROUPS) {
        if (g.needRight && !(a.visibility.rightHand && b.visibility.rightHand)) continue;
        if (g.needLeft && !(a.visibility.leftHand && b.visibility.leftHand)) continue;
        if (g.needPose && !(a.visibility.pose && b.visibility.pose)) continue;
        for (let i = g.start; i < g.end; i++) {
            const d = a.features[i] - b.features[i];
            const wt = FEATURE_WEIGHTS[i];
            s += wt * d * d;
            w += wt;
        }
    }
    return w ? Math.sqrt(s / w) : 0;
}

function positionDelta(aFeat, bFeat) {
    return POSITION_DIMS.map((i) => bFeat[i] - aFeat[i]);
}

function cosineSim(u, v) {
    let dot = 0, mu = 0, mv = 0;
    for (let i = 0; i < u.length; i++) { dot += u[i] * v[i]; mu += u[i] * u[i]; mv += v[i] * v[i]; }
    if (mu === 0 || mv === 0) return 0;
    return dot / Math.sqrt(mu * mv);
}

// Metric 3D pose landmarks ("world landmarks" - hip-centered, not 0-1 image-
// normalized) for the avatar retargeting pipeline. The @mediapipe/holistic
// CDN build this app loads does NOT expose these under the documented
// `results.poseWorldLandmarks` name (verified empirically - that property is
// absent), but the data IS present under an internal, minified property key
// that isn't a stable public API (and could rename on any future update,
// since the CDN URL is unpinned). So this detects it by SHAPE instead of by
// name: a same-length array of 33 landmarks, structurally like poseLandmarks
// (x/y/z/visibility), but a DIFFERENT array (world landmarks are centered
// near 0 and can be negative; poseLandmarks are 0-1 normalized image coords).
function findWorldLandmarks(results) {
    const pose = results.poseLandmarks;
    if (!pose) return null;
    for (const key of Object.keys(results)) {
        const val = results[key];
        if (val === pose || !Array.isArray(val) || val.length !== pose.length) continue;
        const p0 = val[0];
        if (p0 && typeof p0.x === "number" && typeof p0.y === "number" && typeof p0.z === "number" && typeof p0.visibility === "number") {
            return val;
        }
    }
    return null;
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
        features.push(rw[0], rw[1]); // right wrist x,y relative to the face (nose)
        const lw = wristLocation(pose, 15);
        features.push(lw[0], lw[1]); // left wrist x,y relative to the face (nose)
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

// Wrist position relative to the FACE (nose), scaled by head width. In sign
// language, location is defined relative to the face/head — hands at the lips,
// forehead, cheek, chin — so anchoring here (instead of the shoulders) captures
// the linguistic "location" parameter and stays scale/position invariant.
function wristLocation(pose, wristIdx) {
    const nose = pose[0], wr = pose[wristIdx];
    // Anchor to the face (nose); scale by shoulder width — larger and more stable
    // than head width, so vertical hand positions don't saturate at the clamp.
    let scale = Math.sqrt((pose[11].x - pose[12].x) ** 2 + (pose[11].y - pose[12].y) ** 2);
    if (scale < 0.02) scale = 0.15;
    const dx = (wr.x - nose.x) / scale;
    const dy = (wr.y - nose.y) / scale;
    return [clamp01((dx + 2) / 4), clamp01((dy + 2) / 4)];
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
