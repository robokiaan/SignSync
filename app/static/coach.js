// SignSync ISL - Real-time AI Coach Feedback Engine
let webcamStream = null;
let holisticModel = null;
let cameraHelper = null;
let tfModel = null;

let isWebcamActive = false;
let maxObservedScore = 0;
let lastFeedbackTime = 0;

// Canvas details
let canvasElement, canvasCtx;

function resetCoachState() {
    maxObservedScore = 0;
    document.getElementById("coach-score-display").textContent = "--%";
    document.getElementById("coach-feedback-status").textContent = "Idle";
    document.getElementById("coach-feedback-message").textContent = "Activate your webcam to begin real-time gesture analysis.";
    
    // Prepare canvas elements
    canvasElement = document.getElementById("webcam-canvas");
    canvasCtx = canvasElement.getContext("2d");
}

// Start Webcam and MediaPipe
async function toggleWebcam() {
    const btn = document.getElementById("btn-toggle-camera");
    
    if (isWebcamActive) {
        stopWebcamStream();
        btn.textContent = "Start Camera";
        btn.style.background = "";
        isWebcamActive = false;
        resetCoachState();
        return;
    }

    document.getElementById("coach-feedback-status").textContent = "Starting camera...";
    btn.textContent = "Stop Camera";
    btn.style.background = "var(--danger)";
    
    try {
        const videoElement = document.getElementById("webcam-raw");
        
        // Request camera stream
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" }
        });
        videoElement.srcObject = webcamStream;
        
        // Initialize MediaPipe Holistic if not done yet
        if (!holisticModel) {
            document.getElementById("coach-feedback-status").textContent = "Loading AI models...";
            holisticModel = new Holistic({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
            });
            
            holisticModel.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            holisticModel.onResults(processLandmarks);
        }

        // Initialize TF.js if needed (fail-safe for demo)
        await loadTensorFlowModel();

        // Start Camera helper
        isWebcamActive = true;
        document.getElementById("coach-feedback-status").textContent = "Coach Active";
        
        let lastProcessTime = 0;
        cameraHelper = new Camera(videoElement, {
            onFrame: async () => {
                if (isWebcamActive && holisticModel) {
                    const now = performance.now();
                    if (now - lastProcessTime >= 200) { // Limit MediaPipe to once every 200ms (5 FPS)
                        lastProcessTime = now;
                        await holisticModel.send({ image: videoElement });
                    }
                }
            },
            width: 640,
            height: 480
        });
        
        cameraHelper.start();
        
    } catch (err) {
        console.error("Camera startup failed:", err);
        document.getElementById("coach-feedback-status").textContent = "Camera Error";
        document.getElementById("coach-feedback-message").textContent = "Could not access your webcam. Check browser permissions.";
        btn.textContent = "Start Camera";
        btn.style.background = "";
        isWebcamActive = false;
    }
}

// Stop webcam
function stopWebcamStream() {
    isWebcamActive = false;
    
    if (cameraHelper) {
        cameraHelper.stop();
        cameraHelper = null;
    }
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    const videoElement = document.getElementById("webcam-raw");
    if (videoElement) videoElement.srcObject = null;

    // Clear canvas
    if (canvasCtx && canvasElement) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
}

// Load TF.js model
async function loadTensorFlowModel() {
    try {
        if (!tfModel) {
            // Attempt to load converted LSTM model
            tfModel = await tf.loadLayersModel('/static/model/model.json');
            console.log("TF.js Sign Language classifier loaded successfully.");
        }
    } catch (err) {
        // Log, but proceed with geometric similarity which runs client-side dynamically anyway
        console.warn("TF.js model shard not found. Falling back to real-time geometric similarity analyzer.");
    }
}

// MediaPipe landmark processor callback
function processLandmarks(results) {
    if (!isWebcamActive || !canvasElement || !canvasCtx) return;

    // Ensure canvas matches viewport
    canvasElement.width = canvasElement.clientWidth;
    canvasElement.height = canvasElement.clientHeight;

    // Clear the overlay canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw skeleton landmark connections on webcam canvas
    drawCustomSkeleton(results);

    // AI Gesture Coach Logic
    if (activeSign) {
        analyzeFeedback(results);
    }
}

// Draw stylized glowing neon skeleton links
function drawCustomSkeleton(results) {
    const ctx = canvasCtx;
    const w = canvasElement.width;
    const h = canvasElement.height;

    // Helper to project landmark coordinates
    const project = (lm) => {
        // Mirrored coordinate
        return {
            x: lm.x * w,
            y: lm.y * h
        };
    };

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
        ctx.shadowBlur = 0; // reset
    };

    const drawJoint = (pt, color, radius = 4) => {
        if (!pt) return;
        const p = project(pt);
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
    };

    // Draw shoulders/elbows/wrists
    const pose = results.poseLandmarks;
    if (pose) {
        const leftSh = pose[11];
        const rightSh = pose[12];
        const leftEl = pose[13];
        const rightEl = pose[14];
        const leftWr = pose[15];
        const rightWr = pose[16];

        drawLine(leftSh, rightSh, "#4f46e5", 3);
        drawLine(leftSh, leftEl, "#6366f1", 3);
        drawLine(leftEl, leftWr, "#06b6d4", 3);
        drawLine(rightSh, rightEl, "#6366f1", 3);
        drawLine(rightEl, rightWr, "#06b6d4", 3);

        [leftSh, rightSh, leftEl, rightEl, leftWr, rightWr].forEach(p => drawJoint(p, "#14b8a6", 5));
    }

    // Draw Left Hand (21 points)
    if (results.leftHandLandmarks) {
        drawHandBones(results.leftHandLandmarks, drawLine, drawJoint);
    }

    // Draw Right Hand (21 points)
    if (results.rightHandLandmarks) {
        drawHandBones(results.rightHandLandmarks, drawLine, drawJoint);
    }
}

function drawHandBones(landmarks, drawLine, drawJoint) {
    const wrist = landmarks[0];
    const colorLine = "#10b981";
    const colorJoint = "#67e8f9";

    // 5 fingers connection
    for (let f = 0; f < 5; f++) {
        const start = 1 + (f * 4);
        drawLine(wrist, landmarks[start], colorLine, 2);
        for (let j = 0; j < 3; j++) {
            drawLine(landmarks[start + j], landmarks[start + j + 1], colorLine, 2);
        }
    }
    landmarks.forEach(p => drawJoint(p, colorJoint, 3));
}

// // Continuous Sequence Buffers
let isRefVideoProcessing = false;
let refHolisticModel = null;
let refSequence = [];
let userSequence = [];

// Dynamic Time Warping (DTW) Gesture Analyzer
let lastSpokenTime = 0;
let lastSpokenText = "";

const FEEDBACK_MESSAGES = [
    "Adjust your right thumb shape.",
    "Straighten or curl your right index finger.",
    "Adjust your right middle finger.",
    "Adjust your right ring finger.",
    "Adjust your right pinky finger.",
    "Adjust your left thumb shape.",
    "Straighten or curl your left index finger.",
    "Adjust your left middle finger.",
    "Adjust your left ring finger.",
    "Adjust your left pinky finger.",
    "Check your right arm or elbow bend.",
    "Check your left arm or elbow bend."
];

function speakFeedback(text) {
    const now = performance.now();
    if (now - lastSpokenTime >= 3500 && text !== lastSpokenText) {
        lastSpokenTime = now;
        lastSpokenText = text;
        window.speechSynthesis.cancel(); // cancel ongoing speech to prevent overlap
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        window.speechSynthesis.speak(utterance);
    }
}

function resetDTWSequences() {
    refSequence = [];
    userSequence = [];
}

function getMaskedVectorDistance(userFrame, refFrame) {
    // If no hands are visible in the user frame, force maximum distance (1.0)
    if (!userFrame.visibility.rightHand && !userFrame.visibility.leftHand) {
        return 1.0;
    }
    
    let sum = 0;
    let count = 0;
    
    // Compare Right Hand (indices 0 to 4)
    if (userFrame.visibility.rightHand && refFrame.visibility.rightHand) {
        for (let i = 0; i < 5; i++) {
            sum += Math.pow(userFrame.features[i] - refFrame.features[i], 2);
            count++;
        }
    }
    
    // Compare Left Hand (indices 5 to 9)
    if (userFrame.visibility.leftHand && refFrame.visibility.leftHand) {
        for (let i = 5; i < 10; i++) {
            sum += Math.pow(userFrame.features[i] - refFrame.features[i], 2);
            count++;
        }
    }
    
    // Compare Pose / Elbows (indices 10 to 11)
    if (userFrame.visibility.pose && refFrame.visibility.pose) {
        for (let i = 10; i < 12; i++) {
            sum += Math.pow(userFrame.features[i] - refFrame.features[i], 2);
            count++;
        }
    }
    
    // If no joints are mutually visible, return 1.0
    if (count === 0) return 1.0;
    
    return Math.sqrt(sum / count); // normalized distance
}

function analyzeFeedback(results) {
    // Record user features for DTW matching
    const featuresObj = extractFrameFeatures(results);
    userSequence.push(featuresObj);

    if (refSequence.length < 5 || userSequence.length < 5) {
        document.getElementById("coach-feedback-status").textContent = "Recording movement...";
        document.getElementById("coach-feedback-message").textContent = "Keep replicating the sign movements as shown in the reference video.";
        return;
    }
    
    const dtwCost = calculateDTWDistance(userSequence, refSequence);
    
    // Max cost threshold for matching (Subsequence DTW has tighter cost bounds)
    const maxThreshold = 0.32; 
    let score = 0;
    if (dtwCost < maxThreshold) {
        score = Math.round(100 * (1 - (dtwCost / maxThreshold)));
        score = Math.min(Math.max(score, 20), 98); // Bound score realistically
    } else {
        score = 20; // Default lower bound
    }
    
    if (score > maxObservedScore) {
        maxObservedScore = score;
    }
    
    // JOINT-LEVEL DIAGNOSTIC FEEDBACK
    const userFeaturesObj = userSequence[userSequence.length - 1];
    let bestMatchIdx = 0;
    let minDistance = Infinity;
    
    // Find closest matched frame in reference sequence
    for (let i = 0; i < refSequence.length; i++) {
        const dist = getMaskedVectorDistance(userFeaturesObj, refSequence[i]);
        if (dist < minDistance) {
            minDistance = dist;
            bestMatchIdx = i;
        }
    }
    
    const matchedRefFeaturesObj = refSequence[bestMatchIdx];
    let maxDiff = 0;
    let maxDiffIdx = -1;
    
    // Evaluate largest joint deviation, ignoring unobserved limbs
    for (let k = 0; k < userFeaturesObj.features.length; k++) {
        // Map feature index to visibility checks
        let isVisible = false;
        if (k >= 0 && k < 5) {
            isVisible = userFeaturesObj.visibility.rightHand && matchedRefFeaturesObj.visibility.rightHand;
        } else if (k >= 5 && k < 10) {
            isVisible = userFeaturesObj.visibility.leftHand && matchedRefFeaturesObj.visibility.leftHand;
        } else if (k >= 10 && k < 12) {
            isVisible = userFeaturesObj.visibility.pose && matchedRefFeaturesObj.visibility.pose;
        }
        
        if (!isVisible) continue; // Skip if either is missing this joint group!
        
        const diff = Math.abs(userFeaturesObj.features[k] - matchedRefFeaturesObj.features[k]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = k;
        }
    }
    
    let feedback = "";
    if (score >= 75) {
        feedback = "Perfect alignment! Excellent movement matching.";
        speakFeedback("Excellent movement! Hold it steady.");
    } else if (maxDiff > 0.18 && maxDiffIdx >= 0 && maxDiffIdx < FEEDBACK_MESSAGES.length) {
        // High joint error, offer specific correction tips
        feedback = FEEDBACK_MESSAGES[maxDiffIdx];
        speakFeedback(feedback);
    } else if (score >= 50) {
        feedback = "Good motion! Keep moving in synchronization.";
        speakFeedback("Good motion! Keep moving.");
    } else {
        feedback = "Replicate the sequence of movements shown in the reference sign.";
        speakFeedback("Watch the reference and match the movement.");
    }
    
    document.getElementById("coach-feedback-status").textContent = `Score: ${maxObservedScore}%`;
    document.getElementById("coach-feedback-message").textContent = feedback;
    document.getElementById("coach-score-display").textContent = `${maxObservedScore}%`;
}

// Compute Knuckle/Joint distance
function getDistance3D(p1, p2) {
    if (!p1 || !p2) return 1.0;
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

// Dynamic Time Warping (DTW) subsequence calculation engine in JS
function calculateDTWDistance(seq1, seq2) {
    const n = seq1.length;
    const m = seq2.length;
    if (n === 0 || m === 0) return Infinity;
    
    const dtw = Array(n).fill(null).map(() => Array(m).fill(Infinity));
    
    // 1. Open-Ended Start: Initialize first row (user's first frame matches any ref frame)
    for (let j = 0; j < m; j++) {
        dtw[0][j] = getMaskedVectorDistance(seq1[0], seq2[j]);
    }
    
    // 2. Propagate cost matrix
    for (let i = 1; i < n; i++) {
        for (let j = 0; j < m; j++) {
            const cost = getMaskedVectorDistance(seq1[i], seq2[j]);
            
            if (j === 0) {
                dtw[i][0] = cost + dtw[i-1][0];
            } else {
                dtw[i][j] = cost + Math.min(
                    dtw[i-1][j],     // User moves, Reference waits
                    dtw[i][j-1],     // User waits, Reference moves
                    dtw[i-1][j-1]    // Match (Both move)
                );
            }
        }
    }
    
    // 3. Open-Ended End: Find minimum cost in the final row
    let minCost = Infinity;
    for (let j = 0; j < m; j++) {
        if (dtw[n-1][j] < minCost) {
            minCost = dtw[n-1][j];
        }
    }
    
    // Normalize by user sequence length
    return minCost / n;
}

// Extract 12 key scale-invariant features per frame along with visibility flags
function extractFrameFeatures(results) {
    const features = [];
    const visibility = {
        rightHand: !!results.rightHandLandmarks,
        leftHand: !!results.leftHandLandmarks,
        pose: !!results.poseLandmarks
    };
    
    // 1. Right Hand joint extension angles (5 angles)
    if (results.rightHandLandmarks) {
        features.push(getFingerBentAngle(results.rightHandLandmarks, 0, 1, 2, 4) / 180.0);
        features.push(getFingerBentAngle(results.rightHandLandmarks, 5, 6, 7, 8) / 180.0);
        features.push(getFingerBentAngle(results.rightHandLandmarks, 9, 10, 11, 12) / 180.0);
        features.push(getFingerBentAngle(results.rightHandLandmarks, 13, 14, 15, 16) / 180.0);
        features.push(getFingerBentAngle(results.rightHandLandmarks, 17, 18, 19, 20) / 180.0);
    } else {
        features.push(0.5, 0.5, 0.5, 0.5, 0.5);
    }
    
    // 2. Left Hand joint extension angles (5 angles)
    if (results.leftHandLandmarks) {
        features.push(getFingerBentAngle(results.leftHandLandmarks, 0, 1, 2, 4) / 180.0);
        features.push(getFingerBentAngle(results.leftHandLandmarks, 5, 6, 7, 8) / 180.0);
        features.push(getFingerBentAngle(results.leftHandLandmarks, 9, 10, 11, 12) / 180.0);
        features.push(getFingerBentAngle(results.leftHandLandmarks, 13, 14, 15, 16) / 180.0);
        features.push(getFingerBentAngle(results.leftHandLandmarks, 17, 18, 19, 20) / 180.0);
    } else {
        features.push(0.5, 0.5, 0.5, 0.5, 0.5);
    }
    
    // 3. Pose joints extension (Right & Left Elbows - 2 angles)
    const pose = results.poseLandmarks;
    if (pose) {
        features.push(getPoseAngle(pose[12], pose[14], pose[16]) / 180.0);
        features.push(getPoseAngle(pose[11], pose[13], pose[15]) / 180.0);
    } else {
        features.push(0.5, 0.5);
    }
    
    return { features, visibility };
}

function getPoseAngle(p1, p2, p3) {
    if (!p1 || !p2 || !p3) return 180.0;
    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y, z: p1.z - p2.z };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y, z: p3.z - p2.z };
    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    if (mag1 === 0 || mag2 === 0) return 180.0;
    const cosAngle = dot / (mag1 * mag2);
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
    return Math.acos(clampedCos) * (180.0 / Math.PI);
}

// Reference Video play/pause tracking hooks
async function startRefVideoTracking() {
    const video = document.getElementById("practice-ref-video");
    if (!video) return;
    refSequence = [];
    
    if (!refHolisticModel) {
        refHolisticModel = new Holistic({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });
        refHolisticModel.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        refHolisticModel.onResults(processRefLandmarks);
    }
    
    isRefVideoProcessing = true;
    processRefVideoFrame();
}

function stopRefVideoTracking() {
    isRefVideoProcessing = false;
}

async function processRefVideoFrame() {
    const video = document.getElementById("practice-ref-video");
    if (!isRefVideoProcessing || video.paused || video.ended) return;
    
    try {
        await refHolisticModel.send({ image: video });
    } catch (err) {
        console.error("Ref video frame processing error:", err);
    }
    
    if (isRefVideoProcessing && !video.paused && !video.ended) {
        setTimeout(processRefVideoFrame, 200); // 5 FPS to match user input rate
    }
}

function processRefLandmarks(results) {
    if (!isRefVideoProcessing) return;
    const features = extractFrameFeatures(results);
    refSequence.push(features);
}

// Compute knuckle joint angle (returns degrees, 180 is fully straight)
function getFingerBentAngle(hand, p0, p1, p2, p3) {
    if (!hand[p0] || !hand[p1] || !hand[p2] || !hand[p3]) return 180;
    
    // Vector 1: Knuckle base to joint
    const v1 = {
        x: hand[p1].x - hand[p0].x,
        y: hand[p1].y - hand[p0].y,
        z: hand[p1].z - hand[p0].z
    };
    
    // Vector 2: Joint to tip
    const v2 = {
        x: hand[p3].x - hand[p2].x,
        y: hand[p3].y - hand[p2].y,
        z: hand[p3].z - hand[p2].z
    };
    
    // Dot product
    const dot = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    const mag1 = Math.sqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
    const mag2 = Math.sqrt(v2.x*v2.x + v2.y*v2.y + v2.z*v2.z);
    
    if (mag1 === 0 || mag2 === 0) return 180;
    
    const cosAngle = dot / (mag1 * mag2);
    // Clamp values
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
    return (Math.acos(clampedCos) * 180) / Math.PI;
}

// Check if hand is flat (all fingers straight)
function checkHandFlat(hand) {
    // Indexes: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    const indexAngle = getFingerBentAngle(hand, 5, 6, 7, 8);
    const middleAngle = getFingerBentAngle(hand, 9, 10, 11, 12);
    const ringAngle = getFingerBentAngle(hand, 13, 14, 15, 16);
    
    return (indexAngle > 120 && middleAngle > 120 && ringAngle > 120);
}

// Check if hand is in a fist shape (fingers curled)
function checkHandFist(hand) {
    const indexAngle = getFingerBentAngle(hand, 5, 6, 7, 8);
    const middleAngle = getFingerBentAngle(hand, 9, 10, 11, 12);
    const ringAngle = getFingerBentAngle(hand, 13, 14, 15, 16);
    
    return (indexAngle < 100 && middleAngle < 100 && ringAngle < 100);
}
