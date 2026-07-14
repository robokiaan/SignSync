// State management
let activePage = "dashboard";
let activeSign = null;

// Where the reference .mp4 files are served from. Default: same origin (/videos).
// To host the ~1.3 GB video set off the app server (e.g. Cloudflare R2 / S3),
// set this to that bucket's base URL — it MUST send CORS headers, because the
// phase model draws each frame to a canvas (cross-origin video would taint it).
const VIDEO_BASE_URL = "/videos";

// Initialize App -> straight to the dashboard (no authentication).
document.addEventListener("DOMContentLoaded", () => {
    switchPage("dashboard");
});

// Navigation Router
function switchPage(pageId) {
    const pages = ["dashboard", "dictionary", "practice", "sentences"];
    if (!pages.includes(pageId)) pageId = "dashboard";

    pages.forEach((p) => {
        document.getElementById(`page-${p}`).classList.remove("active");
    });

    ["nav-dashboard", "nav-dictionary", "nav-sentences"].forEach((btnId) => {
        const btn = document.getElementById(btnId);
        if (btn) btn.classList.remove("active");
    });

    document.getElementById(`page-${pageId}`).classList.add("active");
    activePage = pageId;

    document.getElementById("main-header").style.display = "flex";
    if (pageId === "dashboard") document.getElementById("nav-dashboard").classList.add("active");
    if (pageId === "dictionary") document.getElementById("nav-dictionary").classList.add("active");
    if (pageId === "sentences") document.getElementById("nav-sentences").classList.add("active");

    // Load data specific to the page
    if (pageId === "dashboard") {
        loadLessons();
    } else if (pageId === "dictionary") {
        loadDictionary();
    } else if (pageId === "sentences") {
        loadSentences();
    }

    // Terminate webcam/reference when leaving the practice arena
    if (pageId !== "practice") {
        stopPracticeArena();
    }
}

// Show Alerts
function showAlert(message, type = "success") {
    const alert = document.getElementById("global-alert");
    alert.textContent = message;
    alert.className = `alert-popup ${type === "success" ? "success-bg" : "error-bg"}`;
    alert.style.display = "block";
    setTimeout(() => {
        alert.style.display = "none";
    }, 4000);
}

// Load Lessons in Dashboard
async function loadLessons() {
    try {
        const response = await fetch("/api/lessons");
        if (!response.ok) return;
        const lessons = await response.json();

        // Sort lessons: Beginner -> Intermediate -> Advanced
        const difficultyOrder = { beginner: 1, intermediate: 2, advanced: 3 };
        lessons.sort((a, b) => {
            const diffA = difficultyOrder[a.difficulty_level.toLowerCase()] || 99;
            const diffB = difficultyOrder[b.difficulty_level.toLowerCase()] || 99;
            return diffA - diffB;
        });

        const container = document.getElementById("dashboard-lessons-container");
        container.innerHTML = "";

        lessons.forEach((lesson) => {
            const lessonCard = document.createElement("div");
            lessonCard.className = "lesson-card glass";

            let itemsListHtml = '<div style="display:flex; flex-wrap:wrap; gap: 0.5rem; margin-top: 0.75rem;">';
            lesson.items.forEach((item) => {
                itemsListHtml += `
                    <button class="btn btn-secondary" onclick="startPractice('${item.sign.sign_name}', '${lesson.title}')" style="width:auto; padding:0.4rem 0.8rem; font-size:0.85rem; text-transform:capitalize;">
                        🎬 ${item.sign.sign_name}
                    </button>
                `;
            });
            itemsListHtml += "</div>";

            lessonCard.innerHTML = `
                <div class="lesson-details">
                    <div class="tag-list">
                        <span class="tag tag-beginner">${lesson.difficulty_level}</span>
                        <span class="tag tag-category">${lesson.category}</span>
                    </div>
                    <h3 style="margin-top: 0.5rem;">${lesson.title}</h3>
                    <p>${lesson.description || "No description provided."}</p>
                    ${itemsListHtml}
                </div>
            `;
            container.appendChild(lessonCard);
        });
    } catch (err) {
        console.error("Error loading lessons:", err);
    }
}

// Load Dictionary Grid
async function loadDictionary() {
    const category = document.getElementById("dict-category-select").value;
    const url = category ? `/api/dictionary?category=${encodeURIComponent(category)}` : "/api/dictionary";

    try {
        const response = await fetch(url);
        if (!response.ok) return;
        const signs = await response.json();
        const container = document.getElementById("dictionary-container");
        container.innerHTML = "";

        if (signs.length === 0) {
            container.innerHTML = `<p style="grid-column: 1/-1; text-align:center; color:var(--text-secondary); margin-top:2rem;">No signs found in this category.</p>`;
            return;
        }

        signs.forEach((sign) => {
            const dictCard = document.createElement("div");
            dictCard.className = "dict-card glass";
            dictCard.onclick = () => startPractice(sign.sign_name, "Dictionary Search");
            dictCard.innerHTML = `
                <h3>${sign.sign_name}</h3>
                <span class="tag tag-category" style="margin-bottom:0.5rem; display:inline-block;">${sign.category}</span>
                <p>${sign.description || "Practice standard signs mapped directly from the INCLUDE dataset."}</p>
            `;
            container.appendChild(dictCard);
        });
    } catch (err) {
        console.error("Error loading dictionary:", err);
    }
}

// Load Sentences List
async function loadSentences() {
    try {
        const response = await fetch("/api/sentences");
        if (!response.ok) return;
        const sentences = await response.json();

        const difficultyOrder = { beginner: 1, intermediate: 2, advanced: 3 };
        sentences.sort((a, b) => {
            const diffA = difficultyOrder[a.difficulty_level.toLowerCase()] || 99;
            const diffB = difficultyOrder[b.difficulty_level.toLowerCase()] || 99;
            return diffA - diffB;
        });

        const container = document.getElementById("sentences-list-container");
        container.innerHTML = "";

        if (sentences.length === 0) {
            container.innerHTML = `<p style="color:var(--text-secondary);">No sentences available yet.</p>`;
            return;
        }

        sentences.forEach((sentence) => {
            const card = document.createElement("div");
            card.className = "lesson-card glass";
            const glossPreview = sentence.items.map((item) => item.sign.sign_name).join(" · ");
            card.innerHTML = `
                <div class="lesson-details" style="width:100%;">
                    <div class="tag-list">
                        <span class="tag tag-beginner">${sentence.difficulty_level}</span>
                        <span class="tag tag-category">${sentence.items.length} sign${sentence.items.length === 1 ? "" : "s"}</span>
                    </div>
                    <h3 style="margin-top: 0.5rem;">${sentence.english_text}</h3>
                    <p style="text-transform:capitalize;">${glossPreview}</p>
                    <button class="btn btn-primary" style="width:auto; margin-top:0.75rem; padding:0.5rem 1.25rem;" onclick="startSentencePractice(${sentence.id})">
                        ▶ Practice
                    </button>
                </div>
            `;
            container.appendChild(card);
        });
    } catch (err) {
        console.error("Error loading sentences:", err);
    }
}

// Enter the practice arena in sentence mode. Shared by the curated-list flow
// (startSentencePractice) and the type-your-own/auto-generate flow
// (practiceCustomSentence) - both already know the gloss word list, so
// neither needs anything beyond this DOM/session setup.
function enterSentenceArena(englishText, glossWords) {
    activeSign = null;
    document.getElementById("btn-arena-back").onclick = () => switchPage("sentences");

    document.getElementById("practice-lesson-title").textContent = "SENTENCE PRACTICE";
    document.getElementById("practice-sign-name").textContent = englishText;
    document.getElementById("practice-sign-desc").textContent = `Sign ${glossWords.length} word${glossWords.length === 1 ? "" : "s"} in order.`;

    switchPage("practice");

    resetDTWSequences();
    resetCoachState();

    beginSentenceSession(englishText, glossWords);
}

// Start a sentence practice session from the curated list. All words' phase
// models are already precomputed (phases.json, fetched up front by coach.js),
// so this needs no live per-word priming.
async function startSentencePractice(sentenceId) {
    try {
        const response = await fetch(`/api/sentences/${sentenceId}`);
        if (!response.ok) {
            showAlert("Error loading sentence.", "error");
            return;
        }
        const sentence = await response.json();
        const glossWords = sentence.items.map((item) => item.sign.sign_name.toLowerCase());
        enterSentenceArena(sentence.english_text, glossWords);
    } catch (err) {
        console.error("Error starting sentence practice:", err);
        showAlert("Could not start sentence practice.", "error");
    }
}

// "Auto-generate" button: fills the input with a suggested sentence. Only
// fills the text - the gloss is always (re)computed at Practice-time from
// whatever's actually in the box, so editing the suggestion can't go stale.
async function generateCustomSentence() {
    try {
        const response = await fetch("/api/sentences/generate");
        if (!response.ok) {
            showAlert("Could not generate a sentence.", "error");
            return;
        }
        const data = await response.json();
        document.getElementById("custom-sentence-input").value = data.english;
    } catch (err) {
        console.error("Error generating sentence:", err);
        showAlert("Could not generate a sentence.", "error");
    }
}

// "Practice" button: match whatever's typed against the dictionary's own
// vocabulary (in typing order) and start a sentence session with the words
// found. Not a translator - words with no dictionary match are skipped and
// called out, not guessed at.
async function practiceCustomSentence() {
    const input = document.getElementById("custom-sentence-input");
    const text = (input.value || "").trim();
    if (!text) {
        showAlert("Type a sentence first.", "error");
        return;
    }
    try {
        const response = await fetch("/api/sentences/parse", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        if (!response.ok) {
            showAlert("Could not parse that sentence.", "error");
            return;
        }
        const { gloss, unmatched } = await response.json();
        if (gloss.length === 0) {
            showAlert("No dictionary words found — try simpler words, like the ones in the Dictionary tab.", "error");
            return;
        }
        if (unmatched.length > 0) {
            showAlert(`Skipped (not in the dictionary): ${unmatched.join(", ")}`, "error");
        }
        enterSentenceArena(text, gloss);
    } catch (err) {
        console.error("Error parsing custom sentence:", err);
        showAlert("Could not start practice for that sentence.", "error");
    }
}

// Start Practice Session
async function startPractice(signName, lessonTitle) {
    try {
        const response = await fetch(`/api/dictionary/${encodeURIComponent(signName)}`);
        if (!response.ok) {
            showAlert("Error loading sign.", "error");
            return;
        }
        activeSign = await response.json();
        endSentenceSession();
        document.getElementById("btn-arena-back").onclick = () => switchPage("dashboard");

        document.getElementById("practice-lesson-title").textContent = lessonTitle;
        document.getElementById("practice-sign-name").textContent = activeSign.sign_name;
        document.getElementById("practice-sign-desc").textContent = activeSign.description || "N/A";

        switchPage("practice");

        // Reset coach + DTW buffers before priming the new reference.
        resetDTWSequences();
        resetCoachState();

        const signKey = activeSign.sign_name.toLowerCase();

        const video = document.getElementById("practice-ref-video");
        const placeholder = document.getElementById("video-placeholder");
        if (video && placeholder) {
            video.pause();
            video.style.display = "none";
            placeholder.style.display = "block";
            document.getElementById("video-placeholder-text").innerHTML = `Loading reference video for <strong>${activeSign.sign_name}</strong>...`;

            video.onerror = () => {
                fetch("/api/log-error", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        sign: activeSign.sign_name,
                        code: video.error ? video.error.code : null,
                        message: video.error ? video.error.message : "unknown",
                        src: video.src,
                    }),
                }).catch((err) => console.error(err));

                placeholder.style.display = "block";
                document.getElementById("video-placeholder-text").innerHTML = `
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📹</div>
                    Reference video not found at:<br>
                    <code style="color:#f43f5e; background:rgba(0,0,0,0.3); padding:0.2rem 0.5rem; border-radius:4px; font-size:0.8rem; display:inline-block; margin-top:0.5rem;">app/static/videos/${activeSign.sign_name.toLowerCase()}.mp4</code>
                `;
                video.style.display = "none";
            };

            video.onloadeddata = () => {
                placeholder.style.display = "none";
                video.style.display = "block";

                // Prime the DTW reference (runs the video through the single
                // MediaPipe model once, then caches it). After priming, loop the
                // video muted for the learner to follow along.
                refReadyPromise = primeReference(signKey).then(() => {
                    const toggle = document.getElementById("toggle-disable-video");
                    handleVideoToggle(toggle ? toggle.checked : false);
                    video.playbackRate = parseFloat(document.getElementById("video-speed-select").value || "1");
                    video.loop = true;
                    video.play().catch(() => {});
                });
            };

            // crossOrigin is required so the phase model can read frames off the
            // canvas when videos are served from another origin (with CORS).
            video.crossOrigin = "anonymous";
            video.src = `${VIDEO_BASE_URL}/${encodeURIComponent(signKey)}.mp4`;
            video.load();
        }
    } catch (err) {
        console.error("Error starting practice:", err);
        showAlert("Could not start practice session.", "error");
    }
}

function stopPracticeArena() {
    stopWebcamStream();

    const video = document.getElementById("practice-ref-video");
    if (video) {
        video.pause();
        video.onloadeddata = null;
        video.onerror = null;
        video.src = "";
        video.style.display = "none";
    }
    resetDTWSequences();
    endSentenceSession();

    const toggle = document.getElementById("toggle-disable-video");
    if (toggle) toggle.checked = false;
    handleVideoToggle(false);

    const placeholder = document.getElementById("video-placeholder");
    if (placeholder) {
        placeholder.style.display = "block";
        document.getElementById("video-placeholder-text").innerHTML = `Please select a sign to start practicing.`;
    }
}

function toggleVideoPlay() {
    const video = document.getElementById("practice-ref-video");
    if (!video) return;
    if (video.paused) video.play();
    else video.pause();
}

function changeVideoSpeed(speed) {
    const video = document.getElementById("practice-ref-video");
    if (!video) return;
    video.playbackRate = parseFloat(speed);
}

// Finish a practice session (no server persistence). Sentence sessions return
// to the Sentences list; single-sign practice returns to the dashboard.
function finishSession() {
    showAlert("Practice session completed. Great work!");
    switchPage(sentenceActive ? "sentences" : "dashboard");
}

function handleVideoToggle(isShowSign) {
    const target = document.getElementById("practice-ref-video");
    if (!target) return;
    if (isShowSign) {
        target.style.opacity = "1";
        target.style.pointerEvents = "auto";

        // The video keeps looping in the background while hidden, so by the
        // time "Show Sign" is checked it can be mid-motion. Rewind to the
        // dormant/rest frame at the start of the clip and resume from there
        // so revealing it always looks like a clean start, not a jump-cut.
        target.pause();
        const resume = () => {
            target.removeEventListener("seeked", resume);
            target.play().catch(() => {});
        };
        target.addEventListener("seeked", resume);
        target.currentTime = 0;
    } else {
        target.style.opacity = "0";
        target.style.pointerEvents = "none";
    }
}
