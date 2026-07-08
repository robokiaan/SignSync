// State management
let currentUser = null;
let token = localStorage.getItem("token") || null;
let activePage = "login";
let activeSign = null;

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
    token = "mock-token";
    verifySession();
});

// Navigation Router
function switchPage(pageId) {
    // Hide all pages
    const pages = ["login", "register", "dashboard", "dictionary", "practice"];
    pages.forEach(p => {
        document.getElementById(`page-${p}`).classList.remove("active");
    });

    // Remove active nav styling
    const navButtons = ["nav-dashboard", "nav-dictionary"];
    navButtons.forEach(btnId => {
        const btn = document.getElementById(btnId);
        if (btn) btn.classList.remove("active");
    });

    // Handle authentication boundaries
    if (!token && pageId !== "login" && pageId !== "register") {
        pageId = "login";
    }

    // Activating page
    document.getElementById(`page-${pageId}`).classList.add("active");
    activePage = pageId;

    // Show/hide navigation bar
    const header = document.getElementById("main-header");
    if (pageId === "login" || pageId === "register") {
        header.style.display = "none";
    } else {
        header.style.display = "flex";
        // Apply active nav class
        if (pageId === "dashboard") document.getElementById("nav-dashboard").classList.add("active");
        if (pageId === "dictionary") document.getElementById("nav-dictionary").classList.add("active");
    }

    // Load data specific to the page
    if (pageId === "dashboard") {
        loadProfile();
        loadLessons();
    } else if (pageId === "dictionary") {
        loadDictionary();
    }

    // Terminate webcam if leaving practice page
    if (pageId !== "practice") {
        stopPracticeArena();
    }
}

function switchAuthPage(pageId) {
    switchPage(pageId);
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

// Verify Session
async function verifySession() {
    try {
        const response = await fetch("/api/profile", {
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (response.ok) {
            currentUser = await response.json();
            updateHeaderProfile(currentUser);
            switchPage("dashboard");
        } else if (response.status === 401 || response.status === 403 || response.status === 404) {
            // Explicit auth error: token is invalid or expired
            logout();
        } else {
            // Server error: keep token but try to show dashboard anyway
            switchPage("dashboard");
        }
    } catch (err) {
        console.error("Session verification failed:", err);
        // Connection error: do not log out, let them browse the dashboard in offline mode
        showAlert("Unable to connect to server. Working in offline mode.", "error");
        switchPage("dashboard");
    }
}

// User Profile Fetching
async function loadProfile() {
    try {
        const response = await fetch("/api/profile", {
            headers: { "Authorization": `Bearer ${token}` }
        });
        
        if (response.ok) {
            currentUser = await response.json();
            updateHeaderProfile(currentUser);
            
            // Populate Dashboard Stats
            document.getElementById("dash-username").textContent = currentUser.username;
            document.getElementById("dash-streak").textContent = currentUser.current_streak;
            document.getElementById("dash-xp").textContent = currentUser.xp;
            document.getElementById("dash-level").textContent = currentUser.current_level;
        }
    } catch (err) {
        console.error("Error loading profile:", err);
    }
}

function updateHeaderProfile(user) {
    document.getElementById("nav-avatar").src = user.avatar_url;
    document.getElementById("nav-username").textContent = user.username;
    document.getElementById("nav-level-xp").textContent = `Lvl ${user.current_level} • ${user.xp} XP`;
    document.getElementById("nav-streak-count").textContent = user.current_streak;
}

// Auth Actions
async function handleLogin(e) {
    e.preventDefault();
    const username_or_email = document.getElementById("login-username").value;
    const password = document.getElementById("login-password").value;

    try {
        const response = await fetch("/api/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username_or_email, password })
        });

        const data = await response.json();
        
        if (response.ok) {
            token = data.access_token;
            localStorage.setItem("token", token);
            showAlert("Logged in successfully!");
            verifySession();
        } else {
            showAlert(data.detail || "Login failed. Check your credentials.", "error");
        }
    } catch (err) {
        showAlert("Connection error. Is backend server running?", "error");
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById("register-username").value;
    const email = document.getElementById("register-email").value;
    const password = document.getElementById("register-password").value;

    try {
        const response = await fetch("/api/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, email, password })
        });

        const data = await response.json();
        
        if (response.ok) {
            token = data.access_token;
            localStorage.setItem("token", token);
            showAlert("Registration complete! Welcome aboard.");
            verifySession();
        } else {
            showAlert(data.detail || "Registration failed.", "error");
        }
    } catch (err) {
        showAlert("Connection error.", "error");
    }
}

function logout() {
    token = null;
    currentUser = null;
    localStorage.removeItem("token");
    switchPage("login");
}

// Load Lessons in Dashboard
async function loadLessons() {
    try {
        const response = await fetch("/api/lessons");
        if (response.ok) {
            const lessons = await response.json();
            
            // Sort lessons: Beginner -> Intermediate -> Advanced
            const difficultyOrder = { "beginner": 1, "intermediate": 2, "advanced": 3 };
            lessons.sort((a, b) => {
                const diffA = difficultyOrder[a.difficulty_level.toLowerCase()] || 99;
                const diffB = difficultyOrder[b.difficulty_level.toLowerCase()] || 99;
                return diffA - diffB;
            });
            
            const container = document.getElementById("dashboard-lessons-container");
            container.innerHTML = "";
            
            lessons.forEach(lesson => {
                const lessonCard = document.createElement("div");
                lessonCard.className = "lesson-card glass";
                
                // Build items list html
                let itemsListHtml = '<div style="display:flex; flex-wrap:wrap; gap: 0.5rem; margin-top: 0.75rem;">';
                lesson.items.forEach(item => {
                    itemsListHtml += `
                        <button class="btn btn-secondary" onclick="startPractice('${item.sign.sign_name}', '${lesson.title}')" style="width:auto; padding:0.4rem 0.8rem; font-size:0.85rem; text-transform:capitalize;">
                            🎬 ${item.sign.sign_name}
                        </button>
                    `;
                });
                itemsListHtml += '</div>';

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
        }
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
        if (response.ok) {
            const signs = await response.json();
            const container = document.getElementById("dictionary-container");
            container.innerHTML = "";
            
            if (signs.length === 0) {
                container.innerHTML = `<p style="grid-column: 1/-1; text-align:center; color:var(--text-secondary); margin-top:2rem;">No signs found in this category.</p>`;
                return;
            }
            
            signs.forEach(sign => {
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
        }
    } catch (err) {
        console.error("Error loading dictionary:", err);
    }
}

// Start Practice Session
async function startPractice(signName, lessonTitle) {
    try {
        const response = await fetch(`/api/dictionary/${encodeURIComponent(signName)}`);
        if (!response.ok) {
            showAlert("Error loading sign animation.", "error");
            return;
        }

        activeSign = await response.json();
        
        // Update practice UI texts
        document.getElementById("practice-lesson-title").textContent = lessonTitle;
        document.getElementById("practice-sign-name").textContent = activeSign.sign_name;
        document.getElementById("practice-sign-desc").textContent = activeSign.description || "N/A";
        
        // Navigate
        switchPage("practice");

        // Load Reference Video Player
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
                        src: video.src
                    })
                }).catch(err => console.error(err));
                
                placeholder.style.display = "block";
                document.getElementById("video-placeholder-text").innerHTML = `
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📹</div>
                    Please place the raw video file at:<br>
                    <code style="color:#f43f5e; background:rgba(0,0,0,0.3); padding:0.2rem 0.5rem; border-radius:4px; font-size:0.8rem; display:inline-block; margin-top:0.5rem;">app/static/videos/${activeSign.sign_name.toLowerCase()}.mp4</code><br>
                    (or run <code>python scripts/fetch_include_videos.py</code>).
                `;
                video.style.display = "none";
            };
            
            video.onloadeddata = () => {
                placeholder.style.display = "none";
                video.style.display = "block";
                
                const toggle = document.getElementById("toggle-disable-video");
                const isShowSign = toggle ? toggle.checked : false;
                handleVideoToggle(isShowSign);
                
                video.playbackRate = parseFloat(document.getElementById("video-speed-select").value || "1");
                video.play();
            };
            
            video.onplay = () => {
                startRefVideoTracking();
            };
            video.onpause = () => {
                stopRefVideoTracking();
            };
            video.onended = () => {
                stopRefVideoTracking();
            };
            
            video.src = `/videos/${activeSign.sign_name.toLowerCase()}.mp4`;
            video.load();
        }
        
        // Reset buffers and weights for DTW
        resetDTWSequences();
        
        // Reset coach feedback state
        resetCoachState();
        
    } catch (err) {
        console.error("Error starting practice:", err);
        showAlert("Could not start practice session.", "error");
    }
}

function stopPracticeArena() {
    // Terminate webcam session via coach.js
    stopWebcamStream();
    
    // Stop and reset video player
    const video = document.getElementById("practice-ref-video");
    if (video) {
        video.pause();
        video.src = "";
        video.style.display = "none";
    }
    // Stop ref video tracking and reset buffers
    stopRefVideoTracking();
    resetDTWSequences();
    
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
    if (video.paused) {
        video.play();
    } else {
        video.pause();
    }
}

function changeVideoSpeed(speed) {
    const video = document.getElementById("practice-ref-video");
    if (!video) return;
    video.playbackRate = parseFloat(speed);
}

// Submit practice session progress
async function submitPracticeScore() {
    if (!activeSign) return;
    
    // Get highest current similarity score from display
    const scoreText = document.getElementById("coach-score-display").textContent;
    let score = parseInt(scoreText.replace("%", ""));
    
    // Fallback if user didn't start webcam or scored 0
    if (isNaN(score)) {
        score = 85; // Give them a decent mock passing score for the demo if requested
    }
    
    try {
        const response = await fetch("/api/practice", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                sign_id: activeSign.id,
                score: score
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showAlert(`Practice session completed successfully!`);
            // Switch back to dashboard
            switchPage("dashboard");
        } else {
            showAlert(data.detail || "Failed to record practice.", "error");
        }
    } catch (err) {
        console.error("Error recording progress:", err);
        showAlert("Network error recording progress.", "error");
    }
}

function handleVideoToggle(isShowSign) {
    const video = document.getElementById("practice-ref-video");
    if (!video) return;
    if (isShowSign) {
        video.style.opacity = "1";
        video.style.pointerEvents = "auto";
    } else {
        video.style.opacity = "0";
        video.style.pointerEvents = "none";
    }
}
