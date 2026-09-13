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
const HOLD_MATCH_THRESHOLD = 0.60; // masked distance at which a hold's quality hits 0
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
// How much movement-direction agreement modulates a phase's credit. Currently 0
// - the check is disabled rather than deleted.
//
// It compares the straight line between two banked poses against the same line
// in the reference, but the measurement is dominated by noise: a learner
// performing correctly (hold quality ~0.7) reproduces the reference direction at
// only ~0.44 agreement, and the result was clamped so that moving BACKWARDS
// scored the same as moving sideways. It was therefore taking up to 20% off a
// pose on the strength of a reading that barely distinguishes right from wrong,
// which cost real points on correct performances.
//
// Re-enable once the measurement is worth trusting: average the frames a pose is
// held for (agreement 0.44 -> 0.71), ignore movements below 0.20 where the
// travel is too small to measure, and use the full -1..+1 range so backwards is
// worse than sideways rather than equal to it.
const MOVE_WEIGHT = 0;
// Feature indices that carry hand position/orientation (used for move direction).
const POSITION_DIMS = [5, 6, 7, 14, 15, 16, 22, 23, 24, 25];
// Minimum time the learner must dwell in a phase before the state machine will
// advance them to the next one - gives a beat to physically switch poses
// (whether that's the next hold within one sign, or the next word in a
// sentence - curPhase advancement is the same mechanism for both since the
// sentence combined model, so this one gate covers both uniformly). Without
// it, a noisy frame during a fast transition could satisfy the advance
// condition and skip a phase the learner never actually held.
const PHASE_TRANSITION_DELAY_MS = 300;

// A checkpoint counts if it clears HOLD_COMPLETE_Q and lands within this margin
// of the best-scoring checkpoint - it does not have to win outright. Two
// checkpoints in the same sign can be genuinely identical, and an exact tie used
// to resolve to whichever came first, leaving the later one permanently
// unreachable. Swept against the labelled data: correct-acceptance plateaus at
// 0.05, while larger margins only add false advances.
const MATCH_TOLERANCE = 0.05;

// Wrist height (pose-derived, so it survives hand-tracking loss) below which a
// hand counts as raised and therefore in use. Matches requiredLimbs' own test.
const HAND_ACTIVE_Y = 0.85;

// Elbow angle (0..1 = 0..180 degrees) below which the arm is bent enough that
// the hand is being used, whatever the wrist height says.
//
// Wrist height alone is not enough, because it comes from the POSE landmarks,
// and MediaPipe drags those toward the hips whenever the hands are worked in
// front of the torso - then wristLocation's (bodyDown + 2) / 4 clamps at
// exactly 1.0, which is indistinguishable from an arm hanging at the side. Half
// of every tracked hand in the reference set (531 of 1056) was being written
// off as resting because of it, and a checkpoint with both hands written off
// grades no handshape at all: "doctor" scored 0.84 against an arbitrary pose
// and 0.835 against "she", so a sentence like "she doctor" completed itself.
//
// The elbow is the corroborating signal, and it separates the two populations
// cleanly: hands actually being signed with sit at 0.41-0.66, arms genuinely
// hanging at 0.78-0.85. Measured over every reference checkpoint, adding this
// takes "doctor"'s mean confusion against the rest of the dictionary from 0.84
// to 0.297 and the poses clearing its bank threshold from 596/617 to 91/617,
// while a learner whose idle hand simply falls differently still scores 1.000
// (grading every tracked hand instead - no wrist or elbow test - costs that
// learner 0.786 mean and 0.355 worst case, which is the regression this whole
// gate exists to prevent).
const ELBOW_ACTIVE = 0.70;

// How long the learner must have no hands on camera before a finished attempt
// closes and the next one can start. Hand tracking flickers for a frame or two
// during fast motion, so this has to be long enough that a dropout mid-sign is
// never mistaken for a deliberate "I'm done".
const HANDS_AWAY_RESET_MS = 500;

// Time penalty: every transition between consecutive checkpoints gets the same
// flat TRANSITION_BUDGET_MS, and each completed budget multiplies the WHOLE
// attempt score by TIME_PENALTY_STEP, compounding, down to TIME_PENALTY_FLOOR.
// So a 5s transition is free, 5-10s costs x0.9, 10-15s x0.81, and so on for the
// rest of the attempt.
//
// Flat rather than a multiple of the reference clip's length, which is what it
// used to be: a sentence has no single clip to scale against - the reference
// video cycles one word at a time - so the whole sentence ended up budgeted
// against whichever word happened to be loaded. One number covers signs and
// sentences alike, and it is the same number the learner experiences either way.
const TRANSITION_BUDGET_MS = 5000;
const TIME_PENALTY_STEP = 0.9;
const TIME_PENALTY_FLOOR = 0.5;

// Per-attempt scoring state (reset in resetDTWSequences)
let activePhaseModel = null;   // { holds, moveDirs, holdTimes, requires }
let phaseBest = [];            // best hold quality achieved per phase this attempt
let phaseReached = [];         // has each phase been hit at least at HOLD_COMPLETE_Q?
let phaseUserPose = [];        // user feature vector captured when each phase was reached
let curPhase = 0;              // phase the learner is currently working toward
let phaseEnteredAt = 0;        // performance.now() when curPhase last changed - see PHASE_TRANSITION_DELAY_MS
let attemptComplete = false;   // final checkpoint banked; waiting for hands to leave before re-arming
let handsAwaySince = 0;        // when the learner's hands first went off camera (0 = they're visible)
let attemptArmed = true;       // ready to begin a new attempt on the next first-checkpoint hit
let timeCrossings = 0;         // banked time-budget overruns this attempt
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

let canvasElement, canvasCtx;

// Reference sequence buffers + caches
let refSequence = [];
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
// no-cache: without it, a browser that ever cached this response keeps
// serving it forever regardless of what's actually on disk now - the same
// bug class as index.html/app.js/data-*.json earlier this session, just
// missed for this file. Also matters for build_phases_from_labels.py, which
// relies on this fetch seeing the CURRENT file (it writes phases.json empty
// before priming so live priming actually runs) rather than a stale cache.
const precomputedPhasesPromise = fetch("phases.json", { cache: "no-cache" })
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}));

const SAMPLE_FPS = 25;          // reference sampling rate; matches the 25fps source so a hand-labelled
                                // timestamp lands on a real frame instead of being rounded onto a 0.2s
                                // grid and then averaged across a 0.4s window (which blurred neighbouring
                                // checkpoints into each other - some came out identical and unreachable)
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
    // The time-penalty budget is a multiple of the reference clip's own length,
    // so a 5s sign isn't held to a 1.6s sign's pace. Read it off the loaded
    // reference video; if it isn't known the penalty simply never fires.
    if (activePhaseModel) {
        activePhaseModel.requires = gradedLimbs(activePhaseModel);
    }
    resetPhaseProgress();
}

function resetPhaseProgress() {
    const P = activePhaseModel ? activePhaseModel.holds.length : 0;
    phaseBest = new Array(P).fill(0);
    phaseReached = new Array(P).fill(false);
    phaseUserPose = new Array(P).fill(null);
    curPhase = 0;
    phaseEnteredAt = performance.now();
    attemptComplete = false;
    handsAwaySince = 0;
    attemptArmed = true;
    timeCrossings = 0;
    missingLimbFrames = 0;
    lastDisplayScore = 0;
}

// Manual reset for single-sign practice ("Reset" button): clears the
// per-attempt phase state and the frame-smoothing buffer ("camera history",
// same pairing resetDTWSequences/the sentence-redo reset use), and forces the
// displayed score to 0% immediately rather than waiting for the next frame.
function resetSignAttempt() {
    prevUserSmoothed = null;
    resetPhaseProgress();
    document.getElementById("coach-score-display").textContent = "0%";
    document.getElementById("coach-feedback-status").textContent = "Reset";
    document.getElementById("coach-feedback-message").textContent = "Score cleared — try the sign again.";
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

    const step = 1 / SAMPLE_FPS; // one source frame; the live webcam rate is independent of this
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
    if (isWebcamActive) {
        const btn = document.getElementById("btn-toggle-camera");
        stopWebcamStream();
        btn.textContent = "Start Camera";
        btn.style.background = "";
        resetCoachState();
        return;
    }
    await startWebcamStream();
}

// Split out of toggleWebcam so app.js can auto-start the camera when
// entering the practice arena, not just on the button click. Idempotent -
// safe to call whenever a session begins regardless of whether the camera
// is already running from a previous sign/sentence in the same visit.
async function startWebcamStream() {
    if (isWebcamActive) return;
    const btn = document.getElementById("btn-toggle-camera");

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
        if (holisticMode === "user") holisticMode = "idle";
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
    // Wrist LOCATION comes from the pose landmarks (15/16), not the hand model,
    // so it stays usable when hand tracking drops - gating it on hand visibility
    // threw away position data we still had.
    { start: 22, end: 24, needPose: true },
    { start: 24, end: 26, needPose: true },
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

const FEATURE_WEIGHT_TOTAL = FEATURE_WEIGHTS.reduce((a, b) => a + b, 0);

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

function resetDTWSequences() {
    refSequence = [];
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

    // Ungradeable features count as MAXIMUM error and the divisor is always the
    // full weight, so every hold is judged on the same terms. Skipping them (and
    // dividing by only what was left) scored a hold on fewer features as an
    // average over fewer chances to be wrong, so sparse holds looked better than
    // properly-formed ones and won matches they should have lost.
    let sum = 0;
    for (const g of FEATURE_GROUPS) {
        // Skip a hand's SHAPE entirely when the reference isn't using that hand.
        // A hand hanging at the side takes whatever shape it happens to fall
        // into, in the reference and in the learner alike, so comparing the two
        // scores noise - and it capped one-handed signs around 75% for anyone
        // who kept their idle hand on camera. Wrist position is still compared
        // (it is pose-derived and a resting hand really is down at the side).
        if (g.needRight || g.needLeft) {
            const side = g.needRight ? "right" : "left";
            if (!refHandRequired(refFrame, side)) continue;
        }

        let gradeable = true;
        if (g.needRight && !(uVis.rightHand && rVis.rightHand)) gradeable = false;
        if (g.needLeft && !(uVis.leftHand && rVis.leftHand)) gradeable = false;
        if (g.needPose && !(uVis.pose && rVis.pose)) gradeable = false;

        // Anything still here is a hand the reference IS using, so a learner not
        // showing it is a real miss and scores as maximum error.

        for (let i = g.start; i < g.end; i++) {
            const w = FEATURE_WEIGHTS[i];
            if (gradeable) {
                const d = uf[i] - rf[i];
                sum += w * d * d;
            } else {
                sum += w; // maximum per-feature error
            }
        }
    }

    return Math.sqrt(sum / FEATURE_WEIGHT_TOTAL);
}

// Is this reference hold's `side` hand something the learner must be showing?
// Only if it is doing something: RAISED in this pose, and actually captured so
// there is a handshape to compare against.
//
// Resting counts even when the tracker caught it. Most references keep the idle
// hand in frame hanging at the signer's side, and a learner performing a
// one-handed sign will often have theirs out of shot - demanding it there
// scored 11 features as maximum error on a correct performance, which held 89%
// of signs at zero.
//
// Height comes from the pose landmarks, so it is known whether or not the hand
// model kept its lock.
// What the sign actually needs on camera, derived from its checkpoints rather
// than read from the stored `requires` field.
//
// The stored value comes from requiredLimbs at build time, which counts a hand
// as used only if the tracker saw it in at least half the reference frames. On
// a clip where tracking struggled that returns fewer hands than the checkpoints
// are graded on - "doctor" stores 0 while both of its hands are compared - and
// then the prompt never fires and the learner just watches the score sit near
// zero with no explanation. 62 of 261 signs disagreed this way.
//
// Deriving it from refHandRequired ties the prompt to exactly what is graded.
// Floored at one hand: a handful of signs grade no handshape at all (see
// check_phase_health.py), and "no hands needed" would let the score be earned
// with nothing on camera.
function gradedLimbs(model) {
    let hands = 0;
    for (const hold of model.holds || []) {
        const n = (refHandRequired(hold, "right") ? 1 : 0) + (refHandRequired(hold, "left") ? 1 : 0);
        if (n > hands) hands = n;
    }
    const pose = model.requires ? model.requires.pose : true;
    return { hands: Math.max(1, hands), pose };
}

function refHandRequired(refFrame, side) {
    const tracked = side === "right" ? refFrame.visibility.rightHand : refFrame.visibility.leftHand;
    if (!tracked) return false;   // reference has no handshape here - nothing to compare against
    if (!refFrame.visibility.pose) return true; // nothing to judge by; assume it matters
    const wristY = side === "right" ? refFrame.features[23] : refFrame.features[25];
    if (wristY < HAND_ACTIVE_Y) return true;
    // Wrist height says "down", but pose-derived wrist height is unreliable for
    // hands held in front of the body - see ELBOW_ACTIVE. A bent elbow means the
    // arm is doing something, so the handshape is real and must be graded.
    const elbow = side === "right" ? refFrame.features[18] : refFrame.features[19];
    return elbow < ELBOW_ACTIVE;
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

    let bestIdx = 0;
    for (let i = 1; i < P; i++) if (q[i] > q[bestIdx]) bestIdx = i;

    // Phases run strictly in order: we are only ever looking for curPhase, so a
    // pose resembling an EARLIER hold mid-sign is ignored rather than treated as
    // progress. What still has to be ruled out is banking curPhase while the
    // learner is really still standing in the previous pose, which happens when
    // two consecutive holds look alike - hence comparing against the best hold
    // rather than accepting any pose over the threshold.
    //
    // The comparison allows MATCH_TOLERANCE of slack instead of demanding an
    // outright win: two holds in one sign can be genuinely identical, and an
    // exact tie resolved to whichever came first, leaving the later one
    // permanently unreachable.
    if (q[curPhase] > phaseBest[curPhase]) phaseBest[curPhase] = q[curPhase];
    if (q[curPhase] >= HOLD_COMPLETE_Q && q[curPhase] >= q[bestIdx] - MATCH_TOLERANCE) {
        if (!phaseReached[curPhase]) {
            // Close off the transition that led here, for the time penalty.
            recordTransitionTiming(curPhase);
        }
        phaseReached[curPhase] = true;
        phaseUserPose[curPhase] = matches[curPhase].um.features.slice();
    }

    // Advance once this phase is reached and they've dwelt here for at least
    // PHASE_TRANSITION_DELAY_MS - a beat to physically switch poses. Advancing
    // just moves which phase is being watched/diagnosed next; it does NOT
    // mark that next phase reached; the check above has to independently see
    // it on a later, fresh frame before it counts as done.
    if (
        curPhase < P - 1 &&
        phaseReached[curPhase] &&
        performance.now() - phaseEnteredAt >= PHASE_TRANSITION_DELAY_MS
    ) {
        curPhase++;
        phaseEnteredAt = performance.now();
        if (q[curPhase] > phaseBest[curPhase]) phaseBest[curPhase] = q[curPhase];
    }

    // Taking too long multiplies the whole attempt, compounding per overrun
    // budget. Evaluated live so the number reflects the delay as it happens
    // rather than arriving as a surprise at the end.
    const timePenalty = currentTimePenalty();
    const displayScore = Math.min(100, Math.round(aggregatePhaseScore(P) * timePenalty * 100));

    // Diagnose against the current phase's target (post-advance), in whichever
    // orientation matched.
    const target = holds[curPhase];
    const { idx: worstIdx, diff: worstDiff } = jointDiagnostic(matches[curPhase].um, target);

    return { P, q, matches, displayScore, allPhasesReached: phaseReached.every(Boolean), worstIdx, worstDiff, bestIdx, timePenalty };
}

// How the banked per-checkpoint qualities add up to one score.
//
// Only checkpoints the learner has actually BANKED count. The one being worked
// on contributes nothing: it used to contribute its live quality, so once the
// state machine advanced you were credited for a pose you had not hit - and
// because consecutive poses in a sign resemble each other (the two in "he"
// match each other at 0.62), simply standing in the previous pose scored most
// of the next one. Half of "he" performed read as 81%. Banked qualities latch,
// so the number only ever climbs within an attempt.
//
// A plain sign averages over its checkpoints. A SENTENCE averages per word
// first, then over words, so every word is worth the same 1/N regardless of how
// many checkpoints it happens to have - otherwise a 5-checkpoint word counted
// five times a 1-checkpoint one, and finishing the short words moved the number
// almost not at all. Words with no phase model contribute nothing and are left
// out of the divisor rather than counted as zero, which would make the sentence
// unscorable through no fault of the learner.
function aggregatePhaseScore(P) {
    const phaseValue = (p) => (phaseReached[p] ? phaseBest[p] : 0) * moveCredit(p);
    const ranges = activePhaseModel && activePhaseModel.wordPhaseRanges;

    if (!ranges || ranges.length === 0) {
        let sum = 0;
        for (let p = 0; p < P; p++) sum += phaseValue(p);
        return clamp01(sum / P);
    }

    let sum = 0;
    let scorableWords = 0;
    for (const [start, end] of ranges) {
        if (end <= start) continue;
        let wordSum = 0;
        for (let p = start; p < end; p++) wordSum += phaseValue(p);
        sum += wordSum / (end - start);
        scorableWords++;
    }
    return scorableWords ? clamp01(sum / scorableWords) : 0;
}

// --- Time penalty -----------------------------------------------------------
// Every transition between consecutive checkpoints gets the same flat
// TRANSITION_BUDGET_MS. Each completed budget is one "crossing", crossings
// accumulate across the attempt, and the whole score is multiplied by
// TIME_PENALTY_STEP per crossing down to TIME_PENALTY_FLOOR. Time before the
// FIRST checkpoint is free - lining yourself up is not signing - so a
// single-checkpoint sign carries no penalty at all.
//
// Identical for a sentence, where the gap between two words is just another
// transition: the combined model's checkpoints run continuously across word
// boundaries, so nothing here has to know where one word ends.
function crossingsFor(elapsedMs) {
    if (!isFinite(elapsedMs) || elapsedMs <= 0) return 0;
    return Math.floor(elapsedMs / TRANSITION_BUDGET_MS);
}

// Banked crossings, plus the transition currently in progress so the score ticks
// down while the learner is still stalling rather than only afterwards.
function currentTimePenalty() {
    let crossings = timeCrossings;
    if (curPhase > 0 && !phaseReached[curPhase] && phaseEnteredAt) {
        crossings += crossingsFor(performance.now() - phaseEnteredAt);
    }
    return Math.max(TIME_PENALTY_FLOOR, Math.pow(TIME_PENALTY_STEP, crossings));
}

function recordTransitionTiming(phaseIdx) {
    if (phaseIdx === 0 || !phaseEnteredAt) return; // nothing precedes the first checkpoint
    timeCrossings += crossingsFor(performance.now() - phaseEnteredAt);
}

// Taking both hands off camera ends the attempt, finished or not - it's the
// learner saying "I'm done with this go". Called before the missing-limb prompt
// in both scorers, because that prompt returns early and would otherwise stop
// an abandoned attempt from ever clearing.
//
// HANDS_AWAY_RESET_MS is what separates "put my hands down" from "the tracker
// blinked": hand tracking drops for a frame or two during fast motion, and a
// blink must never wipe an attempt in progress.
//
// Shared by single-sign and sentence practice. In a sentence the combined model
// spans every word, so clearing it puts the learner back on word 1 - which is
// also the ONLY way a sentence restarts now.
//
// Returns true if it cleared the attempt and the caller should stop here.
function clearAttemptIfHandsAway(user) {
    const handsVisible = user.visibility.rightHand || user.visibility.leftHand;
    if (handsVisible) {
        handsAwaySince = 0;
        return false;
    }
    if (!handsAwaySince) handsAwaySince = performance.now();

    // Only fires when there is progress to clear, so simply standing away from
    // the camera doesn't re-reset every HANDS_AWAY_RESET_MS.
    const somethingToClear = attemptComplete || phaseReached.some(Boolean);
    if (!somethingToClear || performance.now() - handsAwaySince < HANDS_AWAY_RESET_MS) return false;

    const wasMidAttempt = !attemptComplete && phaseReached.some(Boolean);
    attemptComplete = false;
    attemptArmed = true;      // next first-checkpoint hit starts a fresh attempt
    handsAwaySince = 0;
    resetPhaseProgress();
    prevUserSmoothed = null;  // don't blend across the gap

    const restartTarget = sentenceActive && sentenceViewMode === "all" && sentenceGloss.length
        ? `"${sentenceGloss[0]}"`
        : "the sign";
    document.getElementById("coach-score-display").textContent = "0%";
    document.getElementById("coach-feedback-status").textContent = "Ready";
    document.getElementById("coach-feedback-message").textContent = wasMidAttempt
        ? `Attempt cleared. Start again from ${restartTarget} whenever you're ready.`
        : `Start from ${restartTarget} whenever you're ready.`;
    if (sentenceActive && sentenceViewMode === "all") renderGlossStrip();
    return true;
}

function analyzeFeedback(results) {
    const user = smoothUserFrame(extractFrameFeatures(results));

    if (!activePhaseModel || activePhaseModel.holds.length === 0) {
        document.getElementById("coach-feedback-status").textContent = "Analyzing reference...";
        return;
    }

    if (clearAttemptIfHandsAway(user)) return;

    // Prompt (debounced) if the sign needs limbs the learner isn't showing, and
    // freeze the score meanwhile so it can't be gamed by hiding a required hand.
    const need = activePhaseModel.requires;
    const limbMsg = attemptComplete ? null : requiredLimbsMissing(user, need);
    if (limbMsg) {
        missingLimbFrames++;
        if (missingLimbFrames >= 3) {
            document.getElementById("coach-feedback-status").textContent = "Show required limbs";
            document.getElementById("coach-feedback-message").textContent = limbMsg;
            document.getElementById("coach-score-display").textContent = `${lastDisplayScore}%`;
            return;
        }
    } else {
        missingLimbFrames = 0;
    }

    const { P, q, displayScore, allPhasesReached: done, worstIdx, worstDiff } = scoreActiveModel(user);
    lastDisplayScore = displayScore;

    // Finishing locks the score on screen; taking the hands away (handled above)
    // is what clears it and arms the next attempt.
    if (done && !attemptComplete) {
        attemptComplete = true;
        attemptArmed = false;
    }

    // Feedback focuses on the current phase (diagnose against the matched orientation).
    let feedback;
    if (done) {
        // Score is locked now; hands off camera is what starts the next attempt.
        const praise = displayScore >= 75
            ? "Excellent — you matched the whole sign! "
            : (displayScore < 60 ? "Try again more slowly, following the reference closely. " : "");
        feedback = `${praise}Lower your hands to go again.`;
    } else if (q[curPhase] >= HOLD_COMPLETE_Q) {
        // Deliberately says nothing about which checkpoint this is. The model
        // splits a sign into held poses to score it, but the learner performs
        // one continuous movement and never experiences those divisions -
        // consecutive checkpoints are small variations of each other (the median
        // one already scores 0.82 from simply standing in the one before), and
        // some sit ON the path between their neighbours, so "pose 3 of 5" named
        // a boundary that isn't there and appeared to skip.
        const onLastPose = curPhase === P - 1;
        if (onLastPose) feedback = "That's it — hold it steady.";
        else feedback = "Good — keep going.";
    } else if (worstIdx >= 0 && worstDiff > 0.18) {
        feedback = FEATURE_FEEDBACK[worstIdx];
    } else {
        feedback = "Follow the movement in the reference video.";
    }

    document.getElementById("coach-feedback-status").textContent = `Score: ${displayScore}%`;
    document.getElementById("coach-feedback-message").textContent = feedback;
    document.getElementById("coach-score-display").textContent = `${displayScore}%`;
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
    const wordRequires = [];

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
        // Each word keeps its OWN limb requirement, derived the same way single
        // -sign practice derives it. The max across the sentence (still computed
        // above, for anything wanting a whole-sentence answer) is the wrong
        // thing to prompt on: one two-handed word in the sentence made the coach
        // demand two hands for every one-handed word in it too.
        wordRequires.push(model && model.holds ? gradedLimbs(model) : { hands: 1, pose: true });
    }

    return {
        holds, moveDirs, holdTimes,
        requires: { hands: handsMax, pose: poseNeeded },
        wordPhaseRanges, wordRequires,
    };
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

// The limb requirement of the word the learner is on right now. Falls back to
// the whole-model requirement for anything without per-word data.
function currentWordRequires() {
    const perWord = activePhaseModel && activePhaseModel.wordRequires;
    if (!perWord || !perWord.length) return activePhaseModel.requires;
    return perWord[currentWordIndex()] || activePhaseModel.requires;
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
            savedFocusStash = { phaseBest, phaseReached, phaseUserPose, curPhase, phaseEnteredAt };
        }
        focusOnWord(mode); // repoint activePhaseModel at the selected word
    }
    applyViewMode();
}

// Drives the reference-video display for whichever mode is active. Scoring
// state (activePhaseModel etc.) is set separately by resumeCombinedModel()/
// focusOnWord() before this is called.
function applyViewMode() {
    if (sentenceViewMode === "all") playCombinedVideo();
    else loadWordVideo(sentenceViewMode);
}

// Point the active phase model back at the whole-sentence combined model and
// restore its progress if a focus-word excursion stashed it (see
// setSentenceViewMode). Video display is handled separately by
// applyViewMode()/playCombinedVideo(), not here.
function resumeCombinedModel() {
    activePhaseModel = sentenceCombinedModel;
    if (savedFocusStash) {
        ({ phaseBest, phaseReached, phaseUserPose, curPhase, phaseEnteredAt } = savedFocusStash);
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
    video.src = videoUrl(word);
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
    video.src = videoUrl(word);
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
    pre.src = videoUrl(nextWord);
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
            return `<span class="gloss-chip ${progressCls} ${selectedCls}" style="cursor:pointer;" onclick="setSentenceViewMode('${word}')" title="Focus this word: its own video and camera check">${i + 1}. ${word}</span>`;
        })
        .join("");
    const viewChip = `<span class="gloss-chip ${sentenceViewMode === "all" ? "view-active" : ""}" style="margin-left:auto; cursor:pointer;" onclick="setSentenceViewMode('all')" title="Combined video and camera check for the whole sentence">All</span>`;
    strip.innerHTML = wordChips + viewChip;
}

function analyzeSentenceFeedback(results) {
    const user = smoothUserFrame(extractFrameFeatures(results));

    if (!activePhaseModel || activePhaseModel.holds.length === 0) {
        document.getElementById("coach-feedback-status").textContent = "Loading word...";
        return;
    }

    // Lowering both hands is the one and only thing that restarts a sentence,
    // finished or not, and it restarts it from word 1. The old rule - wipe the
    // board as soon as a completed sentence saw a pose resembling word 1's
    // first checkpoint - fired without warning and was the only path by which a
    // sentence score could fall to 0.
    if (clearAttemptIfHandsAway(user)) return;

    // A finished sentence holds its score on screen until the hands come down.
    if (attemptComplete) return;

    // Per-word limb requirement, not the sentence-wide maximum: a one-handed
    // word is satisfied by one hand even when a later word in the same sentence
    // needs two.
    const need = currentWordRequires();
    const limbMsg = requiredLimbsMissing(user, need);
    if (limbMsg) {
        missingLimbFrames++;
        if (missingLimbFrames >= 3) {
            document.getElementById("coach-feedback-status").textContent = "Show required limbs";
            document.getElementById("coach-feedback-message").textContent = limbMsg;
            return;
        }
    } else {
        missingLimbFrames = 0;
    }

    const result = scoreActiveModel(user);
    lastDisplayScore = result.displayScore;
    const curIdx = currentWordIndex();

    // Freeze the finished score in place; clearAttemptIfHandsAway above is what
    // releases it.
    if (result.allPhasesReached) attemptComplete = true;

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
    } else {
        document.getElementById("coach-feedback-status").textContent = `Score: ${result.displayScore}%`;
        document.getElementById("coach-feedback-message").textContent = result.allPhasesReached
            ? "Sentence complete!"
            : `Sign "${sentenceGloss[curIdx]}" (word ${curIdx + 1}/${sentenceGloss.length}).`;
        document.getElementById("coach-score-display").textContent = `${result.displayScore}%`;
    }

    renderGlossStrip(); // isWordDone() reads live phaseReached, so this can change any frame, not just on completion

    if (result.allPhasesReached) {
        finishSentenceSession(result.displayScore);
        // No reset here - lowering both hands is what starts the next attempt.
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
//
// Also torso-tilt invariant: raw image dx/dy would shift if the signer (or
// camera) is tilted/leaning even though the hand's position relative to their
// own head/torso hasn't changed. Projected onto a torso-aligned frame instead
// - "down" is the nose-to-mid-shoulder direction (always available from a
// typical chest-up webcam framing; hip landmarks aren't, so they're not used),
// "right" is perpendicular to it. Which of the two perpendicular directions is
// picked as "right" doesn't matter for correctness, only consistency - a
// left/right sign ambiguity here is exactly what matchHold's mirror-aware
// matching (mirrorFrame) already exists to absorb.
function wristLocation(pose, wristIdx) {
    const nose = pose[0], wr = pose[wristIdx];
    const ls = pose[11], rs = pose[12];
    const midShoulderX = (ls.x + rs.x) / 2, midShoulderY = (ls.y + rs.y) / 2;

    let scale = Math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2);
    if (scale < 0.02) scale = 0.15;

    let downX = midShoulderX - nose.x, downY = midShoulderY - nose.y;
    const downLen = Math.sqrt(downX * downX + downY * downY) || 1;
    downX /= downLen; downY /= downLen;
    const rightX = -downY, rightY = downX; // 90 deg rotation of "down"

    const dxImg = wr.x - nose.x, dyImg = wr.y - nose.y;
    const bodyRight = (dxImg * rightX + dyImg * rightY) / scale;
    const bodyDown = (dxImg * downX + dyImg * downY) / scale;

    return [clamp01((bodyRight + 2) / 4), clamp01((bodyDown + 2) / 4)];
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
