import re

with open('templates/students.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add search input field
search_html = '''        <div class="d-flex gap-2">
            <div class="input-group">
                <span class="input-group-text bg-dark border-secondary text-info"><i class="fa-solid fa-search"></i></span>
                <input type="text" id="search-student" class="form-control form-control-custom bg-dark border-secondary text-white" placeholder="Search by ID or Name..." onkeyup="searchStudents()">
            </div>
            <button onclick="retrainModel()" class="btn btn-outline-warning text-nowrap"><i class="fa-solid fa-brain me-1"></i>Retrain AI Model</button>
            <button class="btn btn-neon-blue text-nowrap" data-bs-toggle="modal" data-bs-target="#addStudentModal"><i class="fa-solid fa-user-plus me-1"></i>Register New Student</button>
        </div>'''

old_header = '''        <div class="d-flex gap-2">
            <button onclick="retrainModel()" class="btn btn-outline-warning"><i class="fa-solid fa-brain me-1"></i>Retrain AI Model</button>
            <button class="btn btn-neon-blue" data-bs-toggle="modal" data-bs-target="#addStudentModal"><i class="fa-solid fa-user-plus me-1"></i>Register New Student</button>
        </div>'''

if old_header in content:
    content = content.replace(old_header, search_html)
else:
    print("WARNING: Old header for search bar not found.")

# 2. Add searchStudents JS function and update loadStudents
old_js = '''function loadStudents() {
    fetch('/api/students')
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('students-table-body');
            tbody.innerHTML = '';

            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No students registered yet.</td></tr>';
                return;
            }'''

new_js = '''function searchStudents() {
    loadStudents();
}

function loadStudents() {
    const q = document.getElementById('search-student') ? document.getElementById('search-student').value.trim() : '';
    const url = q ? `/api/students/search?q=${encodeURIComponent(q)}` : '/api/students';
    
    fetch(url)
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('students-table-body');
            tbody.innerHTML = '';

            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No students found.</td></tr>';
                return;
            }'''

if old_js in content:
    content = content.replace(old_js, new_js)
else:
    print("WARNING: Old loadStudents function not found.")

with open('templates/students.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("DONE: Search UI and API integration added to students.html")
