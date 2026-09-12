import re

with open('templates/students.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the live camera section: remove video/getUserMedia, use MJPEG img tag instead
old_cam_section = re.search(
    r'<!-- RIGHT: Live Camera Preview -->.*?(?=</div>\s*</div>\s*</div>\s*</div>\s*<div class="modal-footer)',
    content, re.DOTALL
)

if old_cam_section:
    print("Found camera section. Replacing...")
    
    new_cam_section = '''<!-- RIGHT: Live Camera Preview (MJPEG from OpenCV server) -->
                    <div class="col-md-7">
                        <div class="d-flex flex-column align-items-center h-100" style="background: rgba(0,0,0,0.4); border-radius: 12px; padding: 14px; border: 2px solid rgba(0,174,255,0.3);">
                            <div style="position: relative; width: 100%; max-width: 440px;">
                                <!-- MJPEG stream from OpenCV backend — shows real face detection box -->
                                <img id="reg-mjpeg" src="" alt="Live Camera"
                                     style="width:100%; border-radius:10px; border: 2px solid #00aeff; display:none; background:#000;"
                                     onerror="this.style.display='none'; document.getElementById('reg-cam-placeholder').style.display='flex';">
                                <!-- Placeholder when camera is off -->
                                <div id="reg-cam-placeholder" style="width:100%; min-height:300px; border-radius:10px; background: linear-gradient(135deg,#1a1a2e,#16213e); display:flex; flex-direction:column; align-items:center; justify-content:center; border: 2px dashed rgba(0,174,255,0.4);">
                                    <i class="fa-solid fa-camera fa-4x text-info mb-3" style="opacity:0.5;"></i>
                                    <p class="text-muted mb-1 fw-bold">Live Face Camera</p>
                                    <p class="text-muted small">Click Start Camera to see OpenCV live preview</p>
                                </div>
                                <!-- Progress badge -->
                                <div id="reg-progress-badge" style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.75); color:#00ff88; font-size:1em; font-weight:bold; padding:4px 13px; border-radius:20px; display:none;">
                                    Captured: <span id="reg-captured-count">0</span> / 50
                                </div>
                            </div>
                            <div class="mt-3 text-center w-100">
                                <div id="reg-cam-status" class="small mb-2">
                                    <i class="fa-solid fa-circle text-danger me-1"></i><span id="reg-cam-status-text">Camera offline</span>
                                </div>
                                <button type="button" onclick="startRegistrationCamera()" class="btn btn-sm btn-outline-info me-2">
                                    <i class="fa-solid fa-play me-1"></i>Start Camera
                                </button>
                                <button type="button" onclick="stopRegistrationCamera()" class="btn btn-sm btn-outline-secondary">
                                    <i class="fa-solid fa-stop me-1"></i>Stop Camera
                                </button>
                            </div>
                        </div>
                    </div>'''
    
    content = content[:old_cam_section.start()] + new_cam_section + content[old_cam_section.end():]
    print("Camera section replaced successfully")
else:
    print("Camera section not found via regex. Trying simple replace...")
    # Simple targeted replace of the img/video element
    old_vid = '<video id="reg-video" autoplay muted playsinline style="width:100%; border-radius:10px; background:#000; display:none; border: 2px solid #00aeff;"></video>'
    new_vid = '<img id="reg-mjpeg" src="" alt="Live Camera" style="width:100%; border-radius:10px; border: 2px solid #00aeff; display:none; background:#000;">'
    if old_vid in content:
        content = content.replace(old_vid, new_vid)
        print("Video element replaced with img MJPEG element")
    else:
        print("Video element also not found!")

# Now also update the JS: startRegistrationCamera should load MJPEG src, stopRegistrationCamera clears it
old_start_fn = '''function startRegistrationCamera() {
    const video = document.getElementById('reg-video');
    const placeholder = document.getElementById('reg-cam-placeholder');
    const statusText = document.getElementById('reg-cam-status-text');
    const statusIcon = document.querySelector('#reg-cam-status i');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support webcam access.');
        return;
    }

    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
        .then(stream => {
            _regStream = stream;
            video.srcObject = stream;
            video.style.display = 'block';
            placeholder.style.display = 'none';
            statusText.innerText = 'Camera live — look into the camera!';
            statusIcon.className = 'fa-solid fa-circle text-success me-1';
        })
        .catch(err => {
            console.error('Camera error:', err);
            statusText.innerText = 'Camera error: ' + err.message;
            statusIcon.className = 'fa-solid fa-circle text-danger me-1';
        });
}

function stopRegistrationCamera() {
    if (_regStream) {
        _regStream.getTracks().forEach(t => t.stop());
        _regStream = null;
    }
    const video = document.getElementById('reg-video');
    const placeholder = document.getElementById('reg-cam-placeholder');
    const statusText = document.getElementById('reg-cam-status-text');
    const statusIcon = document.querySelector('#reg-cam-status i');
    const badge = document.getElementById('reg-progress-badge');

    if (video) { video.srcObject = null; video.style.display = 'none'; }
    if (placeholder) placeholder.style.display = 'flex';
    if (statusText) statusText.innerText = 'Camera offline';
    if (statusIcon) statusIcon.className = 'fa-solid fa-circle text-danger me-1';
    if (badge) badge.style.display = 'none';
}'''

new_start_fn = '''function startRegistrationCamera() {
    const img = document.getElementById('reg-mjpeg');
    const placeholder = document.getElementById('reg-cam-placeholder');
    const statusText = document.getElementById('reg-cam-status-text');
    const statusIcon = document.querySelector('#reg-cam-status i');

    if (!img) return;
    // Load MJPEG stream from OpenCV Flask backend
    img.src = '/api/students/live_feed?' + Date.now();
    img.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
    statusText.innerText = 'Camera live — OpenCV face detection active!';
    statusIcon.className = 'fa-solid fa-circle text-success me-1';
}

function stopRegistrationCamera() {
    const img = document.getElementById('reg-mjpeg');
    const placeholder = document.getElementById('reg-cam-placeholder');
    const statusText = document.getElementById('reg-cam-status-text');
    const statusIcon = document.querySelector('#reg-cam-status i');
    const badge = document.getElementById('reg-progress-badge');

    if (img) { img.src = ''; img.style.display = 'none'; }
    if (placeholder) placeholder.style.display = 'flex';
    if (statusText) statusText.innerText = 'Camera offline';
    if (statusIcon) statusIcon.className = 'fa-solid fa-circle text-danger me-1';
    if (badge) badge.style.display = 'none';
}'''

if old_start_fn in content:
    content = content.replace(old_start_fn, new_start_fn)
    print("JS camera functions updated to MJPEG approach")
else:
    print("JS camera functions block not found — may need manual check")

with open('templates/students.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("DONE: students.html updated")
