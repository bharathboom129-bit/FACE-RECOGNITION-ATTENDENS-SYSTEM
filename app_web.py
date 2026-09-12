import os
import threading
import datetime
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, send_file, redirect, url_for

from pdf_generator import generate_attendance_pdf
from email_notifier import send_defaulter_alert_email, send_hod_summary_email, get_email_config

app = Flask(__name__)

MASTER_CSV = os.path.join("Attendance", "Master_Attendance.csv")
STUDENTS_CSV = os.path.join("StudentDetails", "StudentDetails.csv")
SUBJECTS_CSV = os.path.join("Subjects", "Subjects.csv")


def _get_master_df():
    if os.path.isfile(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            if df.empty:
                return pd.DataFrame()
            if 'DATE' in df.columns and 'TIME' in df.columns:
                df['DATETIME'] = pd.to_datetime(
                    df['DATE'].astype(str) + ' ' + df['TIME'].astype(str),
                    format='%d-%m-%Y %H:%M:%S', errors='coerce'
                )
                df = df.sort_values('DATETIME')

                session_keys = []
                last_seen = {}
                sess_cnt = {}
                for _, row in df.iterrows():
                    subj = str(row.get('CLASS_SUBJECT', ''))
                    dt_val = row['DATETIME']
                    dt_str = dt_val.strftime('%Y-%m-%d') if pd.notnull(dt_val) else str(row.get('DATE', ''))
                    key = (subj, dt_str)
                    if key not in last_seen or pd.isnull(dt_val) or (dt_val - last_seen[key]).total_seconds() > 600:
                        sess_cnt[key] = sess_cnt.get(key, 0) + 1
                        if pd.notnull(dt_val):
                            last_seen[key] = dt_val
                    session_keys.append(f"{subj}|{dt_str}|S{sess_cnt[key]}")
                df['SESSION_KEY'] = session_keys
                df = df.drop_duplicates(subset=['ID', 'SESSION_KEY'])
            return df
        except Exception as e:
            print("Web app master df error:", e)
    return pd.DataFrame()


def _get_students_df():
    if os.path.isfile(STUDENTS_CSV):
        try:
            return pd.read_csv(STUDENTS_CSV)
        except Exception as e:
            print("Web app students df error:", e)
    return pd.DataFrame()


HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Classroom Academic Web Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', Roboto, sans-serif; }
        .navbar-custom { background-color: #1e1e1e; border-bottom: 2px solid #00aeff; }
        .card-custom { background-color: #1e1e1e; border: 1px solid #2d2d2d; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
        .metric-card { border-left: 4px solid #00aeff; transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-3px); }
        .table-dark-custom { background-color: #1e1e1e; color: #e0e0e0; }
        .table-dark-custom th { background-color: #252525; border-bottom: 2px solid #00aeff; color: #00aeff; }
        .table-dark-custom td { border-bottom: 1px solid #2a2a2a; }
        .badge-regular { background-color: #1b4332; color: #74c69d; font-size: 0.85rem; padding: 6px 12px; border-radius: 20px; }
        .badge-defaulter { background-color: #4a1515; color: #ff8585; font-size: 0.85rem; padding: 6px 12px; border-radius: 20px; }
        .btn-custom { background-color: #00aeff; color: #000; font-weight: 600; border: none; border-radius: 8px; }
        .btn-custom:hover { background-color: #0088cc; color: #fff; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark navbar-custom py-3 mb-4">
        <div class="container">
            <a class="navbar-brand font-weight-bold" href="/"><i class="fa-solid fa-graduation-cap me-2 text-info"></i>Smart Classroom Academic Portal</a>
            <div class="d-flex">
                <a href="/download_pdf" class="btn btn-outline-info btn-sm me-2"><i class="fa-solid fa-file-pdf me-1"></i>Download PDF Report</a>
                <a href="/api/send_email_warnings" class="btn btn-outline-warning btn-sm"><i class="fa-solid fa-paper-plane me-1"></i>Alert Defaulters Email</a>
            </div>
        </div>
    </nav>

    <div class="container mb-5">
        {% block content %}{% endblock %}
    </div>

    <footer class="text-center text-muted py-3 border-top border-secondary mt-auto">
        <small>&copy; Smart Classroom Biometric Attendance & Analytics Platform — Academic Web Portal</small>
    </footer>
</body>
</html>
"""

INDEX_TEMPLATE = HTML_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="row mb-4">
    <div class="col-md-3 mb-3">
        <div class="card card-custom metric-card p-3" style="border-left-color: #00aeff;">
            <div class="text-muted small">TOTAL STUDENTS</div>
            <h2 class="fw-bold my-1 text-info">{{ total_students }}</h2>
            <div class="small text-secondary"><i class="fa-solid fa-users me-1"></i>Registered Capacity</div>
        </div>
    </div>
    <div class="col-md-3 mb-3">
        <div class="card card-custom metric-card p-3" style="border-left-color: #2ec4b6;">
            <div class="text-muted small">TOTAL SESSIONS</div>
            <h2 class="fw-bold my-1 text-success">{{ total_sessions }}</h2>
            <div class="small text-secondary"><i class="fa-solid fa-calendar-check me-1"></i>Tracked Class Slots</div>
        </div>
    </div>
    <div class="col-md-3 mb-3">
        <div class="card card-custom metric-card p-3" style="border-left-color: #ff9f1c;">
            <div class="text-muted small">AVG ATTENDANCE RATE</div>
            <h2 class="fw-bold my-1 text-warning">{{ "%.1f"|format(avg_att) }}%</h2>
            <div class="small text-secondary"><i class="fa-solid fa-chart-line me-1"></i>Overall Classroom %</div>
        </div>
    </div>
    <div class="col-md-3 mb-3">
        <div class="card card-custom metric-card p-3" style="border-left-color: #e71d36;">
            <div class="text-muted small">DEFAULTERS (<75%)</div>
            <h2 class="fw-bold my-1 text-danger">{{ defaulter_count }}</h2>
            <div class="small text-secondary"><i class="fa-solid fa-triangle-exclamation me-1"></i>Action Required</div>
        </div>
    </div>
</div>

<!-- Search Bar -->
<div class="card card-custom p-4 mb-4">
    <h5 class="fw-bold text-info mb-3"><i class="fa-solid fa-magnifying-glass me-2"></i>Student Quick Portal Lookup</h5>
    <form action="/student_search" method="GET" class="row g-2">
        <div class="col-md-9">
            <input type="text" name="query" class="form-control bg-dark text-white border-secondary" placeholder="Enter Student Name or USN / ID (e.g. 1AM24MC015 or Bharath)" required>
        </div>
        <div class="col-md-3">
            <button type="submit" class="btn btn-custom w-100"><i class="fa-solid fa-search me-1"></i>Search Student</button>
        </div>
    </form>
</div>

<!-- Student Attendance Table -->
<div class="card card-custom p-4">
    <div class="d-flex justify-content-between align-items-center mb-3">
        <h5 class="fw-bold text-white mb-0"><i class="fa-solid fa-table-list me-2 text-info"></i>Student Attendance Performance Tracker</h5>
        <span class="badge bg-secondary">Updated Real-Time</span>
    </div>
    <div class="table-responsive">
        <table class="table table-dark-custom align-middle">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Student ID (USN)</th>
                    <th>Student Name</th>
                    <th>Class / Section</th>
                    <th>Sessions Attended</th>
                    <th>Attendance %</th>
                    <th>Status</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                {% for st in student_list %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td class="fw-bold text-info">{{ st.id }}</td>
                    <td>{{ st.name }}</td>
                    <td>{{ st.class }}</td>
                    <td>{{ st.present }} / {{ total_sessions }}</td>
                    <td><strong class="{{ 'text-success' if st.pct >= 75 else 'text-danger' }}">{{ "%.1f"|format(st.pct) }}%</strong></td>
                    <td>
                        {% if st.pct >= 75 %}
                        <span class="badge-regular"><i class="fa-solid fa-circle-check me-1"></i>Regular</span>
                        {% else %}
                        <span class="badge-defaulter"><i class="fa-solid fa-circle-exclamation me-1"></i>DEFAULTER (<75%)</span>
                        {% endif %}
                    </td>
                    <td>
                        <a href="/student/{{ st.id }}" class="btn btn-sm btn-outline-info"><i class="fa-solid fa-eye me-1"></i>View Profile</a>
                    </td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="8" class="text-center text-muted py-4">No student attendance records found.</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
""")


STUDENT_TEMPLATE = HTML_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="mb-3">
    <a href="/" class="btn btn-outline-secondary btn-sm"><i class="fa-solid fa-arrow-left me-1"></i>Back to Dashboard</a>
</div>

<div class="card card-custom p-4 mb-4">
    <div class="row align-items-center">
        <div class="col-md-8">
            <h3 class="fw-bold text-info mb-1"><i class="fa-solid fa-id-card me-2"></i>{{ student_name }}</h3>
            <p class="text-muted mb-0">USN / Student ID: <strong>{{ student_id }}</strong> | Class: <strong>{{ student_class }}</strong></p>
        </div>
        <div class="col-md-4 text-md-end mt-3 mt-md-0">
            {% if att_pct >= 75 %}
            <span class="badge-regular fs-6"><i class="fa-solid fa-circle-check me-1"></i>Regular Attendance ({{ "%.1f"|format(att_pct) }}%)</span>
            {% else %}
            <span class="badge-defaulter fs-6"><i class="fa-solid fa-circle-exclamation me-1"></i>DEFAULTER ({{ "%.1f"|format(att_pct) }}%)</span>
            {% endif %}
        </div>
    </div>
</div>

<div class="card card-custom p-4">
    <h5 class="fw-bold text-white mb-3"><i class="fa-solid fa-list-check me-2 text-info"></i>Attended Session History</h5>
    <div class="table-responsive">
        <table class="table table-dark-custom align-middle">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Subject</th>
                    <th>Date</th>
                    <th>Time</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for log in session_logs %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td class="fw-bold text-info">{{ log.CLASS_SUBJECT }}</td>
                    <td>{{ log.DATE }}</td>
                    <td>{{ log.TIME }}</td>
                    <td><span class="badge bg-success">Present</span></td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="5" class="text-center text-muted py-4">No session records logged for this student.</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
""")


@app.route("/")
def index():
    master_df = _get_master_df()
    students_df = _get_students_df()

    total_registered = len(students_df)
    total_sessions = master_df['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0

    student_list = []
    defaulter_cnt = 0
    total_pct_sum = 0.0

    if not students_df.empty:
        for _, r in students_df.iterrows():
            st_id = str(r['ID'])
            st_name = str(r['NAME'])
            st_cls = str(r.get('CLASS_SUBJECT', ''))

            p_cnt = master_df[master_df['ID'].astype(str) == st_id]['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
            pct = (p_cnt / total_sessions * 100.0) if total_sessions > 0 else 0.0
            total_pct_sum += pct

            if pct < 75.0:
                defaulter_cnt += 1

            student_list.append({
                'id': st_id,
                'name': st_name,
                'class': st_cls,
                'present': p_cnt,
                'pct': pct
            })

    avg_att = (total_pct_sum / total_registered) if total_registered > 0 else 0.0

    return render_template_string(
        INDEX_TEMPLATE,
        total_students=total_registered,
        total_sessions=total_sessions,
        avg_att=avg_att,
        defaulter_count=defaulter_cnt,
        student_list=student_list
    )


@app.route("/student/<student_id>")
def student_profile(student_id):
    master_df = _get_master_df()
    students_df = _get_students_df()

    st_row = students_df[students_df['ID'].astype(str) == str(student_id)] if not students_df.empty else None
    st_name = st_row.iloc[0]['NAME'] if (st_row is not None and not st_row.empty) else student_id
    st_cls = st_row.iloc[0].get('CLASS_SUBJECT', '') if (st_row is not None and not st_row.empty) else ''

    total_sessions = master_df['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
    p_cnt = master_df[master_df['ID'].astype(str) == str(student_id)]['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
    att_pct = (p_cnt / total_sessions * 100.0) if total_sessions > 0 else 0.0

    session_logs = []
    if not master_df.empty:
        st_logs = master_df[master_df['ID'].astype(str) == str(student_id)]
        session_logs = st_logs.to_dict('records')

    return render_template_string(
        STUDENT_TEMPLATE,
        student_id=student_id,
        student_name=st_name,
        student_class=st_cls,
        att_pct=att_pct,
        session_logs=session_logs
    )


@app.route("/student_search")
def student_search():
    query = request.args.get("query", "").strip().lower()
    students_df = _get_students_df()
    if not students_df.empty:
        matches = students_df[
            students_df['ID'].astype(str).str.lower().str.contains(query) |
            students_df['NAME'].astype(str).str.lower().str.contains(query)
        ]
        if not matches.empty:
            target_id = str(matches.iloc[0]['ID'])
            return redirect(url_for('student_profile', student_id=target_id))

    return redirect(url_for('index'))


@app.route("/download_pdf")
def download_pdf():
    subj = request.args.get("subject", None)
    filepath = generate_attendance_pdf(subject_name=subj)
    return send_file(filepath, as_attachment=True)


@app.route("/api/send_email_warnings")
def api_send_email_warnings():
    master_df = _get_master_df()
    students_df = _get_students_df()
    total_sessions = master_df['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0

    sent_count = 0
    errors = []

    if not students_df.empty and total_sessions > 0:
        for _, srow in students_df.iterrows():
            st_id = str(srow['ID'])
            st_name = str(srow['NAME'])
            p_email = srow.get('PARENT_EMAIL', '')
            p_cnt = master_df[master_df['ID'].astype(str) == st_id]['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
            att_pct = (p_cnt / total_sessions * 100.0)

            if att_pct < 75.0:
                success, msg = send_defaulter_alert_email(st_id, st_name, p_email, att_pct, total_sessions, p_cnt)
                if success:
                    sent_count += 1
                else:
                    errors.append(msg)

    if sent_count > 0:
        return f"<h3>✅ Sent {sent_count} email warnings to defaulters!</h3><a href='/'>Return to Portal</a>"
    elif errors:
        return f"<h3>⚠️ Email Alert Notice</h3><p>{errors[0]}</p><a href='/'>Return to Portal</a>"
    else:
        return "<h3>ℹ️ No defaulters (<75%) found or no sessions tracked yet!</h3><a href='/'>Return to Portal</a>"


def run_flask_in_background(host="127.0.0.1", port=5000):
    def start_app():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=start_app, daemon=True)
    t.start()
    print(f"🌐 Flask Web Portal running in background at http://{host}:{port}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
