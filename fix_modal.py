import re

with open('templates/students.html', 'r', encoding='utf-8') as f:
    content = f.read()

new_modal = """<!-- Add Student Modal -->
<div class="modal fade" id="addStudentModal" tabindex="-1">
    <div class="modal-dialog modal-dialog-centered modal-xl">
        <div class="modal-content bg-dark text-white border-secondary" style="background-color: #141722 !important;">
            <div class="modal-header border-secondary">
                <h5 class="modal-title fw-bold text-info"><i class="fa-solid fa-user-plus me-2"></i>Register New Student</h5>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" onclick="stopRegistrationCamera()"></button>
            </div>
            <div class="modal-body">
                <div class="row g-4">
                    <!-- LEFT: Form -->
                    <div class="col-md-5">
                        <form id="add-student-form">
                            <div class="mb-3">
                                <label class="form-label-custom">Student ID / USN *</label>
                                <input type="text" id="reg-id" class="form-control form-control-custom" placeholder="e.g. 1AM24MC099" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label-custom">Full Name *</label>
                                <input type="text" id="reg-name" class="form-control form-control-custom" placeholder="e.g. Rahul Sharma" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label-custom">Class / Subject *</label>
                                <input type="text" id="reg-subject" list="subject-options" class="form-control form-control-custom" placeholder="e.g. MCA Section A, Python, Java" required>
                                <datalist id="subject-options">
                                    {% for subj in subjects %}
                                    <option value="{{ subj }}">
                                    {% endfor %}
                                    <option value="MCA Section A">
                                    <option value="Python">
                                    <option value="Java">
                                </datalist>
                            </div>
                            <div class="mb-3">
                                <label class="form-label-custom">Parent Email Address</label>
                                <input type="email" id="reg-email" class="form-control form-control-custom" placeholder="parent@gmail.com">
                            </div>
                            <div class="mb-3">
                                <label class="form-label-custom">Parent Phone Number</label>
                                <input type="text" id="reg-phone" class="form-control form-control-custom" placeholder="+91 9876543210">
                            </div>
                            <div id="capture-status-msg" class="small text-info mb-2 fw-bold" style="display:none;"></div>
                        </form>
                    </div>
                    <!-- RIGHT: Live Camera Preview -->
                    <div class="col-md-7">
                        <div class="d-flex flex-column align-items-center h-100" style="background: rgba(0,0,0,0.4); border-radius: 12px; padding: 14px; border: 2px solid rgba(0,174,255,0.3);">
                            <div style="position: relative; width: 100%; max-width: 440px;">
                                <video id="reg-video" autoplay muted playsinline style="width:100%; border-radius:10px; background:#000; display:none; border: 2px solid #00aeff;"></video>
                                <div id="reg-cam-placeholder" style="width:100%; min-height:300px; border-radius:10px; background: linear-gradient(135deg,#1a1a2e,#16213e); display:flex; flex-direction:column; align-items:center; justify-content:center; border: 2px dashed rgba(0,174,255,0.4);">
                                    <i class="fa-solid fa-camera fa-4x text-info mb-3" style="opacity:0.5;"></i>
                                    <p class="text-muted mb-1 fw-bold">Live Face Camera</p>
                                    <p class="text-muted small">Click Start Camera to see yourself live</p>
                                </div>
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
                    </div>
                </div>
            </div>
            <div class="modal-footer border-secondary d-flex justify-content-between">
                <button type="button" id="btn-capture-photos" onclick="captureStudentPhotos()" class="btn btn-warning btn-sm fw-bold">
                    <i class="fa-solid fa-camera me-1"></i>Capture 50 Face Samples &amp; Train AI
                </button>
                <div>
                    <button type="button" class="btn btn-outline-secondary btn-sm" data-bs-dismiss="modal" onclick="stopRegistrationCamera()">Cancel</button>
                    <button type="button" onclick="submitAddStudent()" class="btn btn-neon-blue btn-sm"><i class="fa-solid fa-check me-1"></i>Save &amp; Register</button>
                </div>
            </div>
        </div>
    </div>
</div>

"""

content = re.sub(
    r'<!-- Add Student Modal -->.*?(?=<!-- Edit Student Modal -->)',
    new_modal,
    content,
    flags=re.DOTALL
)

with open('templates/students.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done - modal replaced successfully')
