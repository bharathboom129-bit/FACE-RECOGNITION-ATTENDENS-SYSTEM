import os
import sys
import cv2
import csv
import glob
import time
import datetime
import pandas as pd
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, send_file, redirect, url_for

from pdf_generator import generate_attendance_pdf
from email_notifier import send_defaulter_alert_email, send_hod_summary_email, get_email_config, update_email_config

app = Flask(__name__)

# Paths
STUDENT_DETAILS_DIR = "StudentDetails"
STUDENT_DETAILS_CSV = os.path.join(STUDENT_DETAILS_DIR, "StudentDetails.csv")
ATTENDANCE_DIR = "Attendance"
MASTER_ATTENDANCE_CSV = os.path.join(ATTENDANCE_DIR, "Master_Attendance.csv")
TRAINING_IMG_DIR = "TrainingImage"
TRAINING_LABEL_DIR = "TrainingImageLabel"
TRAINER_YML = os.path.join(TRAINING_LABEL_DIR, "Trainner.yml")
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
SUBJECTS_DIR = "Subjects"
SUBJECTS_CSV = os.path.join(SUBJECTS_DIR, "Subjects.csv")
REPORTS_DIR = "Reports"

for d in [STUDENT_DETAILS_DIR, ATTENDANCE_DIR, TRAINING_IMG_DIR, TRAINING_LABEL_DIR, SUBJECTS_DIR, REPORTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# Global Camera State
camera = None
is_tracking = False
active_subject = "Python"

# Initialize Haar Cascade
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_CASCADE_FILE)


def get_active_subjects():
    if os.path.isfile(SUBJECTS_CSV):
        try:
            df = pd.read_csv(SUBJECTS_CSV)
            return df['SUBJECT_NAME'].dropna().unique().tolist()
        except:
            pass
    return ["Python", "Java", "web appliction"]


def get_students_df():
    if os.path.isfile(STUDENT_DETAILS_CSV):
        try:
            df = pd.read_csv(STUDENT_DETAILS_CSV)
            if not df.empty:
                for col in ['PARENT_EMAIL', 'PARENT_PHONE']:
                    if col not in df.columns:
                        df[col] = ''
                df = df.fillna('')
                return df
        except:
            pass
    return pd.DataFrame(columns=['SERIAL NO.', 'ID', 'NAME', 'CLASS_SUBJECT', 'REGISTRATION_DATE', 'PARENT_EMAIL', 'PARENT_PHONE'])



def get_master_df():
    if os.path.isfile(MASTER_ATTENDANCE_CSV):
        try:
            df = pd.read_csv(MASTER_ATTENDANCE_CSV)
            if df.empty:
                return pd.DataFrame()

            # Ensure all records with non-null CLASS_SUBJECT are preserved
            if 'CLASS_SUBJECT' in df.columns:
                df = df[df['CLASS_SUBJECT'].notnull() & (df['CLASS_SUBJECT'] != '')]

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
            print("Get master df web error:", e)
    return pd.DataFrame()


def train_model():
    """Train LBPH Face Recognizer model."""
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        image_paths = [os.path.join(TRAINING_IMG_DIR, f) for f in os.listdir(TRAINING_IMG_DIR) if f.endswith(('.jpg', '.png'))]
        if not image_paths:
            return False, "No face samples found."

        faces, ids = [], []
        for ipath in image_paths:
            try:
                pil_img = Image.open(ipath).convert('L')
                img_np = np.array(pil_img, 'uint8')
                fname = os.path.split(ipath)[-1]
                serial_id = int(fname.split('.')[1])
                faces.append(img_np)
                ids.append(serial_id)
            except Exception as ex:
                continue

        if faces and ids:
            recognizer.train(faces, np.array(ids))
            recognizer.save(TRAINER_YML)
            return True, f"Trained successfully on {len(faces)} face samples!"
    except Exception as e:
        return False, str(e)
    return False, "Training failed."


def get_student_map():
    """Map serial_number -> (ID, NAME)."""
    s_df = get_students_df()
    mapping = {}
    if not s_df.empty:
        for _, row in s_df.iterrows():
            try:
                s_no = int(row['SERIAL NO.'])
                mapping[s_no] = (str(row['ID']), str(row['NAME']))
            except:
                pass
    return mapping


def gen_attendance_video_stream(subject_name):
    """Video streaming generator for live attendance camera feed."""
    global camera, is_tracking
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    if not os.path.isfile(TRAINER_YML):
        # Model not trained yet
        while True:
            success, frame = camera.read()
            if not success:
                break
            cv2.putText(frame, "Model Not Trained! Register Student First.", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_YML)
    student_map = get_student_map()

    session_start_time = datetime.datetime.now()
    date_str = session_start_time.strftime("%d-%m-%Y")
    time_str = session_start_time.strftime("%H-%M-%S")
    subj_slug = subject_name.replace(" ", "_").replace("/", "_")
    session_csv = os.path.join(ATTENDANCE_DIR, f"Attendance_{subj_slug}_{date_str}_{time_str}.csv")

    logged_ids = set()

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            serial_id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 65 and serial_id in student_map:
                st_id, st_name = student_map[serial_id]
                display_txt = f"{st_name} ({st_id})"
                cv2.putText(frame, display_txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Log attendance
                now = datetime.datetime.now()
                curr_date = now.strftime("%d-%m-%Y")
                curr_time = now.strftime("%H:%M:%S")

                if st_id not in logged_ids:
                    logged_ids.add(st_id)
                    rec = [st_id, st_name, subject_name, curr_date, curr_time, "Present"]

                    # Append to Master
                    write_header = not os.path.isfile(MASTER_ATTENDANCE_CSV) or os.stat(MASTER_ATTENDANCE_CSV).st_size == 0
                    with open(MASTER_ATTENDANCE_CSV, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(['ID', 'NAME', 'CLASS_SUBJECT', 'DATE', 'TIME', 'STATUS'])
                        writer.writerow(rec)

                    # Append to Session CSV
                    write_sess_header = not os.path.isfile(session_csv) or os.stat(session_csv).st_size == 0
                    with open(session_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if write_sess_header:
                            writer.writerow(['ID', 'NAME', 'CLASS_SUBJECT', 'DATE', 'TIME', 'STATUS'])
                        writer.writerow(rec)

            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Overlay status info
        cv2.putText(frame, f"Subject: {subject_name} | Marked: {len(logged_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ================================= WEB ROUTES =================================

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/attendance")
def attendance():
    subjects = get_active_subjects()
    return render_template("attendance.html", subjects=subjects)

@app.route("/students")
def students():
    subjects = get_active_subjects()
    return render_template("students.html", subjects=subjects)

@app.route("/subjects")
def subjects():
    return render_template("subjects.html")

@app.route("/reports")
def reports():
    subjects = get_active_subjects()
    return render_template("reports.html", subjects=subjects)

@app.route("/settings")
def settings():
    cfg = get_email_config()
    return render_template("settings.html", config=cfg)

# ================================= API ENDPOINTS =================================

@app.route("/video_feed")
def video_feed():
    subj = request.args.get("subject", "Python")
    return Response(gen_attendance_video_stream(subj), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stop_video")
def stop_video():
    global camera
    if camera is not None:
        try:
            camera.release()
        except:
            pass
        camera = None
    return jsonify({"success": True, "message": "Camera hardware stopped & released successfully!"})


@app.route("/api/analytics")
def api_analytics():
    students_df = get_students_df()
    master_df = get_master_df()

    total_students = len(students_df)
    total_sessions = master_df['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0

    student_stats = []
    defaulters = []
    total_pct_sum = 0.0

    if not students_df.empty:
        for _, r in students_df.iterrows():
            st_id = str(r['ID'])
            st_name = str(r['NAME'])
            st_cls = str(r.get('CLASS_SUBJECT', ''))

            p_cnt = master_df[master_df['ID'].astype(str) == st_id]['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
            pct = (p_cnt / total_sessions * 100.0) if total_sessions > 0 else 0.0
            total_pct_sum += pct

            item = {
                'id': st_id,
                'name': st_name,
                'class': st_cls,
                'present': p_cnt,
                'total': total_sessions,
                'pct': round(pct, 1),
                'status': 'REGULAR' if pct >= 75.0 else 'DEFAULTER'
            }
            student_stats.append(item)
            if pct < 75.0:
                defaulters.append(item)

    avg_att = (total_pct_sum / total_students) if total_students > 0 else 0.0

    # Subject performance breakdown
    subject_rates = []
    if not master_df.empty and 'CLASS_SUBJECT' in master_df.columns and total_students > 0:
        for s_name, grp in master_df.groupby('CLASS_SUBJECT'):
            sess_cnt = grp['SESSION_KEY'].nunique()
            unique_students = grp['ID'].nunique()
            attended = grp.groupby('SESSION_KEY')['ID'].nunique().sum()
            denom = total_students * sess_cnt if sess_cnt > 0 else 1
            rate = min(100.0, attended / denom * 100.0)
            subject_rates.append({
                'subject': s_name,
                'sessions': sess_cnt,
                'unique_students': unique_students,
                'capacity': total_students,
                'rate': round(rate, 1)
            })

    return jsonify({
        'total_students': total_students,
        'total_sessions': total_sessions,
        'avg_attendance': round(avg_att, 1),
        'defaulter_count': len(defaulters),
        'students': student_stats,
        'defaulters': defaulters,
        'subject_rates': subject_rates
    })


@app.route("/api/analytics/subject_sessions", methods=["GET"])
def api_analytics_subject_sessions():
    """Return all session CSV files for a given subject (click drill-down)."""
    subject = request.args.get("subject", "").strip()
    if not subject:
        return jsonify({"sessions": [], "subject": subject})

    total_students = len(get_students_df())
    subj_slug = subject.replace(' ', '_').replace('/', '_').lower()

    sessions = []
    if os.path.isdir(ATTENDANCE_DIR):
        all_files = sorted([
            f for f in os.listdir(ATTENDANCE_DIR)
            if f.startswith('Attendance_') and f.endswith('.csv') and f != 'Master_Attendance.csv'
        ])
        matched = []
        for fname in all_files:
            core = fname[11:-4]          # strip "Attendance_" and ".csv"
            parts = core.split('_')
            if len(parts) >= 3:
                file_subj_slug = '_'.join(parts[:-2]).lower()
                if file_subj_slug == subj_slug:
                    matched.append(fname)

        total_cnt = len(matched)
        for i, fname in enumerate(matched):
            fpath = os.path.join(ATTENDANCE_DIR, fname)
            # Parse Date & Time from filename e.g. Attendance_Java_12-09-2026_18-08-53.csv
            core = fname[11:-4]
            parts = core.split('_')
            dt_str = "Unknown"
            if len(parts) >= 2:
                d_part = parts[-2]
                t_part = parts[-1].replace('-', ':')
                dt_str = f"{d_part} {t_part}"

            row_cnt = 0
            try:
                fdf = pd.read_csv(fpath)
                row_cnt = len(fdf)
            except Exception:
                pass

            pct_val = (row_cnt / total_students * 100.0) if total_students > 0 else 0.0
            sessions.append({
                "idx": i + 1,
                "slot": f"Session {i+1} of {total_cnt}",
                "datetime": dt_str,
                "present": row_cnt,
                "total": total_students,
                "pct": round(pct_val, 1),
                "filename": fname,
                "filepath": fpath.replace('\\', '/')
            })

    return jsonify({"sessions": sessions, "subject": subject})


@app.route("/api/analytics/weekly", methods=["GET"])
def api_analytics_weekly():
    df = get_master_df()
    if df.empty or 'DATE' not in df.columns:
        return jsonify({'weeks': [], 'counts': []})

    df = df.dropna(subset=['DATE'])
    # Need datetime parsing for date logic
    df['DATETIME_OBJ'] = pd.to_datetime(df['DATE'].astype(str), format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['DATETIME_OBJ'])
    df['WEEK'] = df['DATETIME_OBJ'].dt.isocalendar().week.astype(int)
    df['YEAR'] = df['DATETIME_OBJ'].dt.year
    df['WEEK_LABEL'] = df['DATETIME_OBJ'].dt.strftime('W%U\n%d %b')

    weekly_counts = df.groupby(['YEAR', 'WEEK', 'WEEK_LABEL'])['ID'].count().reset_index()
    weekly_counts = weekly_counts.sort_values(['YEAR', 'WEEK']).tail(8)

    return jsonify({
        'weeks': weekly_counts['WEEK_LABEL'].tolist(),
        'counts': weekly_counts['ID'].tolist()
    })


@app.route("/api/analytics/monthly", methods=["GET"])
def api_analytics_monthly():
    df = get_master_df()
    if df.empty or 'DATE' not in df.columns:
        return jsonify({'labels': [], 'counts': []})

    df = df.dropna(subset=['DATE'])
    df['DATETIME_OBJ'] = pd.to_datetime(df['DATE'].astype(str), format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['DATETIME_OBJ'])
    df['MONTH_NUM'] = df['DATETIME_OBJ'].dt.to_period('M')
    
    monthly = df.groupby('MONTH_NUM')['ID'].count().reset_index()
    monthly = monthly.sort_values('MONTH_NUM')
    
    labels_short = [pd.Period(p, freq='M').strftime('%b %Y') for p in monthly['MONTH_NUM']]
    counts = monthly['ID'].tolist()
    
    return jsonify({
        'labels': labels_short,
        'counts': counts
    })


@app.route("/api/analytics/classwise", methods=["GET"])
def api_analytics_classwise():
    df = get_master_df()
    if df.empty:
        return jsonify({'hours': {}, 'dow': {}})
        
    hour_labels = []
    hour_counts = []
    dow_labels = []
    dow_counts = []

    # Period breakdown by hour
    if 'TIME' in df.columns:
        period_counts = {}
        for t in df['TIME'].dropna():
            try:
                hour = int(str(t).split(':')[0])
                label = f"{hour:02d}:00"
                period_counts[label] = period_counts.get(label, 0) + 1
            except:
                pass
        
        periods = sorted(period_counts.keys())
        pcounts = [period_counts[p] for p in periods]
        hour_labels = periods
        hour_counts = pcounts

    # Day-of-week breakdown
    if 'DATE' in df.columns:
        df2 = df.dropna(subset=['DATE']).copy()
        df2['DATETIME_OBJ'] = pd.to_datetime(df2['DATE'].astype(str), format='%d-%m-%Y', errors='coerce')
        df2 = df2.dropna(subset=['DATETIME_OBJ'])
        df2['DOW'] = df2['DATETIME_OBJ'].dt.day_name()
        
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts_ser = df2['DOW'].value_counts().reindex(dow_order, fill_value=0)
        
        dow_labels = dow_counts_ser.index.tolist()
        dow_counts = dow_counts_ser.values.tolist()

    return jsonify({
        'hours': {'labels': hour_labels, 'counts': hour_counts},
        'dow': {'labels': dow_labels, 'counts': dow_counts}
    })


@app.route("/api/students", methods=["GET"])
def api_get_students():
    df = get_students_df()
    return jsonify(df.to_dict('records') if not df.empty else [])


@app.route("/api/students/add", methods=["POST"])
def api_add_student():
    data = request.json or {}
    st_id = data.get("id", "").strip()
    st_name = data.get("name", "").strip()
    st_subj = data.get("subject", "").strip()
    p_email = data.get("parent_email", "").strip()
    p_phone = data.get("parent_phone", "").strip()

    if not st_id or not st_name:
        return jsonify({"success": False, "message": "ID and Name are required!"}), 400

    df = get_students_df()
    if not df.empty and st_id in df['ID'].astype(str).values:
        return jsonify({"success": False, "message": f"Student ID {st_id} already exists!"}), 400

    next_serial = (df['SERIAL NO.'].astype(int).max() + 1) if not df.empty else 1
    reg_date = datetime.datetime.now().strftime("%d-%m-%Y")

    new_row = {
        'SERIAL NO.': next_serial,
        'ID': st_id,
        'NAME': st_name,
        'CLASS_SUBJECT': st_subj,
        'REGISTRATION_DATE': reg_date,
        'PARENT_EMAIL': p_email,
        'PARENT_PHONE': p_phone
    }

    write_header = not os.path.isfile(STUDENT_DETAILS_CSV) or os.stat(STUDENT_DETAILS_CSV).st_size == 0
    with open(STUDENT_DETAILS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SERIAL NO.', 'ID', 'NAME', 'CLASS_SUBJECT', 'REGISTRATION_DATE', 'PARENT_EMAIL', 'PARENT_PHONE'])
        if write_header:
            writer.writeheader()
        writer.writerow(new_row)

    return jsonify({"success": True, "message": f"Student {st_name} registered successfully!"})


@app.route("/api/students/delete", methods=["POST"])
def api_delete_student():
    data = request.json or {}
    st_id = str(data.get("id", "")).strip()

    df = get_students_df()
    if df.empty or st_id not in df['ID'].astype(str).values:
        return jsonify({"success": False, "message": "Student ID not found!"}), 404

    # Remove student
    df = df[df['ID'].astype(str) != st_id]
    df.to_csv(STUDENT_DETAILS_CSV, index=False)

    # Clean face images
    for f in os.listdir(TRAINING_IMG_DIR):
        if f.startswith(f"{st_id}."):
            try:
                os.remove(os.path.join(TRAINING_IMG_DIR, f))
            except:
                pass

    # Retrain model
    train_model()
    return jsonify({"success": True, "message": f"Student {st_id} deleted and dataset cleaned!"})


@app.route("/api/students/edit", methods=["POST"])
def api_edit_student():
    data = request.json or {}
    old_id = str(data.get("old_id", "")).strip()
    st_id = str(data.get("id", "")).strip()
    st_name = data.get("name", "").strip()
    st_subj = data.get("subject", "").strip()
    p_email = data.get("parent_email", "").strip()
    p_phone = data.get("parent_phone", "").strip()

    if not old_id or not st_id or not st_name:
        return jsonify({"success": False, "message": "Original ID, New ID, and Name are required!"}), 400

    df = get_students_df()
    if df.empty or old_id not in df['ID'].astype(str).values:
        return jsonify({"success": False, "message": f"Student with ID {old_id} not found!"}), 404

    mask = df['ID'].astype(str) == old_id
    df.loc[mask, 'ID'] = st_id
    df.loc[mask, 'NAME'] = st_name
    df.loc[mask, 'CLASS_SUBJECT'] = st_subj
    df.loc[mask, 'PARENT_EMAIL'] = p_email
    df.loc[mask, 'PARENT_PHONE'] = p_phone

    df.to_csv(STUDENT_DETAILS_CSV, index=False)
    return jsonify({"success": True, "message": f"Student {st_name} updated successfully!"})


@app.route("/api/students/live_feed")
def api_students_live_feed():
    """MJPEG live camera stream with face detection overlay for web registration preview."""
    def generate_frames():
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return
        detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
        if detector.empty():
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                cv2.putText(frame, "Align your face with the camera", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Face Detected - Ready", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ret2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret2:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        cam.release()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/students/capture_photos", methods=["POST"])
def api_capture_student_photos():
    data = request.json or {}
    st_id = str(data.get("id", "")).strip()
    st_name = data.get("name", "").strip()
    st_subj = data.get("subject", "").strip() or "General"
    p_email = data.get("parent_email", "").strip()
    p_phone = data.get("parent_phone", "").strip()

    if not st_id or not st_name:
        return jsonify({"success": False, "message": "Student ID and Name are required!"}), 400

    df = get_students_df()
    # Find or assign serial number
    if not df.empty and st_id in df['ID'].astype(str).values:
        serial = int(df[df['ID'].astype(str) == st_id].iloc[0]['SERIAL NO.'])
    else:
        serial = len(df) + 1
        reg_date = datetime.datetime.now().strftime("%d-%m-%Y")
        new_row = {
            'SERIAL NO.': serial,
            'ID': st_id,
            'NAME': st_name,
            'CLASS_SUBJECT': st_subj,
            'REGISTRATION_DATE': reg_date,
            'PARENT_EMAIL': p_email,
            'PARENT_PHONE': p_phone
        }
        write_header = not os.path.isfile(STUDENT_DETAILS_CSV) or os.stat(STUDENT_DETAILS_CSV).st_size == 0
        with open(STUDENT_DETAILS_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['SERIAL NO.', 'ID', 'NAME', 'CLASS_SUBJECT', 'REGISTRATION_DATE', 'PARENT_EMAIL', 'PARENT_PHONE'])
            if write_header:
                writer.writeheader()
            writer.writerow(new_row)

    # Open webcam — EXACT port of old main.py take_student_images()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return jsonify({"success": False, "message": "Could not open Webcam hardware! Please check camera connection."}), 500

    # Use SAME detector as old Tkinter — fresh CascadeClassifier (most reliable)
    detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    if detector.empty():
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    sample_num = 0
    win_title = f"Taking Student Face Registration - Press Q to Cancel"

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # EXACT same params as old main.py: scaleFactor=1.3, minNeighbors=5
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Blue rectangle exactly like old Tkinter app
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample_num += 1
            img_path = os.path.join(TRAINING_IMG_DIR, f"{st_name}.{serial}.{st_id}.{sample_num}.jpg")
            cv2.imwrite(img_path, gray[y:y + h, x:x + w])
            # Show progress count on frame — green text like old app
            cv2.putText(img, f"Captured: {sample_num}/50", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the live window (exactly like old Tkinter cv2.imshow)
        cv2.imshow(win_title, img)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        if sample_num >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sample_num > 0:
        train_model()
        return jsonify({
            "success": True,
            "message": f"Success! Captured {sample_num}/50 face samples for {st_name} (ID: {st_id}) and retrained AI face model!"
        })
    else:
        return jsonify({"success": False, "message": f"No faces captured. Captured {sample_num} images. Make sure your face is clearly visible and well-lit, then try again."}), 400


@app.route("/api/subjects", methods=["GET"])
def api_get_subjects():
    if os.path.isfile(SUBJECTS_CSV):
        try:
            df = pd.read_csv(SUBJECTS_CSV)
            return jsonify(df.to_dict('records'))
        except:
            pass
    return jsonify([])


@app.route("/api/subjects/add", methods=["POST"])
def api_add_subject():
    data = request.json or {}
    s_name = data.get("subject_name", "").strip()
    t_name = data.get("teacher_name", "").strip()

    if not s_name:
        return jsonify({"success": False, "message": "Subject name is required!"}), 400

    write_header = not os.path.isfile(SUBJECTS_CSV) or os.stat(SUBJECTS_CSV).st_size == 0
    with open(SUBJECTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['SUBJECT_NAME', 'TEACHER_NAME'])
        writer.writerow([s_name, t_name])

    return jsonify({"success": True, "message": f"Subject '{s_name}' added successfully!"})


@app.route("/api/subjects/edit", methods=["POST"])
def api_edit_subject():
    data = request.json or {}
    old_name = data.get("old_subject_name", "").strip()
    s_name = data.get("subject_name", "").strip()
    t_name = data.get("teacher_name", "").strip()

    if not old_name or not s_name:
        return jsonify({"success": False, "message": "Original and new subject names are required!"}), 400

    if os.path.isfile(SUBJECTS_CSV):
        df = pd.read_csv(SUBJECTS_CSV)
        mask = df['SUBJECT_NAME'] == old_name
        if mask.any():
            df.loc[mask, 'SUBJECT_NAME'] = s_name
            df.loc[mask, 'TEACHER_NAME'] = t_name
            df.to_csv(SUBJECTS_CSV, index=False)

            if old_name != s_name:
                if os.path.isfile(STUDENT_DETAILS_CSV):
                    st_df = pd.read_csv(STUDENT_DETAILS_CSV)
                    st_df.loc[st_df['CLASS_SUBJECT'] == old_name, 'CLASS_SUBJECT'] = s_name
                    st_df.to_csv(STUDENT_DETAILS_CSV, index=False)

                if os.path.isfile(MASTER_ATTENDANCE_CSV):
                    m_df = pd.read_csv(MASTER_ATTENDANCE_CSV)
                    m_df.loc[m_df['CLASS_SUBJECT'] == old_name, 'CLASS_SUBJECT'] = s_name
                    m_df.to_csv(MASTER_ATTENDANCE_CSV, index=False)

            return jsonify({"success": True, "message": f"Subject '{s_name}' updated successfully!"})

    return jsonify({"success": False, "message": "Subject not found!"}), 404


@app.route("/api/subjects/delete", methods=["POST"])
def api_delete_subject():
    data = request.json or {}
    s_name = data.get("subject_name", "").strip()

    if os.path.isfile(SUBJECTS_CSV):
        df = pd.read_csv(SUBJECTS_CSV)
        df = df[df['SUBJECT_NAME'] != s_name]
        df.to_csv(SUBJECTS_CSV, index=False)

    # Purge Master Attendance records for deleted subject
    if os.path.isfile(MASTER_ATTENDANCE_CSV):
        mdf = pd.read_csv(MASTER_ATTENDANCE_CSV)
        mdf = mdf[mdf['CLASS_SUBJECT'] != s_name]
        mdf.to_csv(MASTER_ATTENDANCE_CSV, index=False)

    return jsonify({"success": True, "message": f"Subject '{s_name}' and attendance records deleted!"})


@app.route("/api/retrain")
def api_retrain():
    success, msg = train_model()
    return jsonify({"success": success, "message": msg})


@app.route("/download_pdf")
def download_pdf():
    subj = request.args.get("subject", None)
    pdf_path = generate_attendance_pdf(subject_name=subj)
    return send_file(pdf_path, as_attachment=True)


@app.route("/api/send_email_alerts", methods=["POST"])
def api_send_email_alerts():
    students_df = get_students_df()
    master_df = get_master_df()
    total_sessions = master_df['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0

    sent_count = 0
    if not students_df.empty and total_sessions > 0:
        for _, srow in students_df.iterrows():
            st_id = str(srow['ID'])
            st_name = str(srow['NAME'])
            p_email = srow.get('PARENT_EMAIL', '')
            p_cnt = master_df[master_df['ID'].astype(str) == st_id]['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
            att_pct = (p_cnt / total_sessions * 100.0)

            if att_pct < 75.0:
                ok, _ = send_defaulter_alert_email(st_id, st_name, p_email, att_pct, total_sessions, p_cnt)
                if ok:
                    sent_count += 1

    return jsonify({"success": True, "sent_count": sent_count, "message": f"Sent {sent_count} defaulter alert emails!"})


@app.route("/api/send_parent_email/<student_id>", methods=["POST"])
def api_send_parent_email(student_id):
    students_df = get_students_df()
    master_df = get_master_df()
    total_sessions = master_df['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0

    st_row = students_df[students_df['ID'].astype(str) == str(student_id)]
    if st_row.empty:
        return jsonify({"success": False, "message": f"Student ID {student_id} not found!"}), 404

    srow = st_row.iloc[0]
    st_name = str(srow['NAME'])
    p_email = str(srow.get('PARENT_EMAIL', '')).strip()
    p_cnt = master_df[master_df['ID'].astype(str) == str(student_id)]['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
    att_pct = (p_cnt / total_sessions * 100.0) if total_sessions > 0 else 0.0

    ok, msg = send_defaulter_alert_email(student_id, st_name, p_email, att_pct, total_sessions, p_cnt)
    if ok:
        target = p_email if (p_email and "@" in p_email) else "default configured email"
        return jsonify({"success": True, "message": f"Parent warning email sent to {target} for student {st_name} ({student_id})!"})
    else:
        return jsonify({"success": False, "message": f"Email error: {msg}"}), 400


@app.route("/api/settings", methods=["POST"])
def api_update_settings():
    data = request.json or {}
    sender = data.get("sender_email", "").strip()
    pwd = data.get("sender_password", "").strip()
    hod = data.get("hod_email", "").strip()

    update_email_config(sender, pwd, hod)
    return jsonify({"success": True, "message": "Settings updated successfully!"})


if __name__ == "__main__":
    print("🚀 Starting Smart Classroom Full-Stack Web Server on http://127.0.0.1:5000 ...")
    
# ─────────────────────────────────────────────────────────────────
# IN-MEMORY PENDING SESSION  (filled by start_session, cleared by save/discard)
# ─────────────────────────────────────────────────────────────────
_pending_session = {}   # { "records": [...], "subject": "...", "filename": "...", "filepath": "..." }


@app.route("/api/attendance/start_session", methods=["POST"])
def api_start_session():
    """
    Run OpenCV face recognition session. Returns detected records WITHOUT saving.
    Frontend must call /api/attendance/save_session to persist, or discard.
    """
    global _pending_session
    data = request.json or {}
    subject = data.get("subject", "General").strip() or "General"

    if not os.path.isfile(TRAINER_YML):
        return jsonify({"success": False, "message": "No trained AI model found! Please register students and train the model first."}), 400

    df_students = get_students_df()
    if df_students.empty:
        return jsonify({"success": False, "message": "No registered students found in database!"}), 400

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(TRAINER_YML)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading face model: {e}"}), 500

    detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    if detector.empty():
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return jsonify({"success": False, "message": "Could not open Webcam! Please check camera connection."}), 500

    session_dt = datetime.datetime.now()
    date_str = session_dt.strftime('%d-%m-%Y')
    marked = {}   # student_id -> record dict — deduplicates automatically
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            serial_pred, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 65:
                matched = df_students[df_students['SERIAL NO.'].astype(int) == int(serial_pred)]
                if not matched.empty:
                    st_id   = str(matched.iloc[0]['ID'])
                    st_name = str(matched.iloc[0]['NAME'])
                    st_cls  = str(matched.iloc[0].get('CLASS_SUBJECT', subject))
                    color   = (0, 255, 0)
                    label   = f"{st_name} ({st_id})"
                    if st_id not in marked:
                        marked[st_id] = {
                            'ID': st_id, 'NAME': st_name,
                            'CLASS_SUBJECT': subject,
                            'DATE': date_str,
                            'TIME': datetime.datetime.now().strftime('%H:%M:%S'),
                            'STATUS': 'Present'
                        }
                else:
                    label, color = "Unknown", (0, 0, 255)
            else:
                label, color = "Unknown", (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, label, (x+5, y-10), font, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Subject: {subject} | Present: {len(marked)} | Press Q to Finish", (10, 30), font, 0.65, (0,255,255), 2)
        cv2.imshow(f"Attendance Session — {subject}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    records = list(marked.values())

    if records:
        # Build file paths (DON'T write yet — pending user confirmation)
        ts = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        clean_subj = subject.replace(' ', '_').replace('/', '_')
        fname = f"Attendance_{clean_subj}_{ts}.csv"
        fpath = os.path.join(ATTENDANCE_DIR, fname)
        _pending_session = {"records": records, "subject": subject, "filename": fname, "filepath": fpath}
        return jsonify({"success": True, "pending": True, "count": len(records), "records": records,
                        "message": f"Session complete! Found {len(records)} student(s) present for '{subject}'."})
    else:
        _pending_session = {}
        return jsonify({"success": True, "pending": False, "count": 0, "records": [],
                        "message": "Session ended — no students were detected. Nothing to save."})


@app.route("/api/attendance/save_session", methods=["POST"])
def api_save_session():
    """User confirmed Save — write pending session records to CSV files."""
    global _pending_session
    if not _pending_session or not _pending_session.get("records"):
        return jsonify({"success": False, "message": "No pending session to save!"}), 400

    records  = _pending_session["records"]
    fpath    = _pending_session["filepath"]
    subject  = _pending_session["subject"]

    df = pd.DataFrame(records)

    # 1. Save timestamped session file
    df.to_csv(fpath, index=False)

    # 2. Append to Master_Attendance.csv
    master_exists = os.path.isfile(MASTER_ATTENDANCE_CSV)
    df.to_csv(MASTER_ATTENDANCE_CSV, mode='a', header=not master_exists, index=False)

    _pending_session = {}
    return jsonify({"success": True,
                    "message": f"Attendance saved! {len(records)} student(s) marked present for '{subject}'.",
                    "filename": os.path.basename(fpath)})


@app.route("/api/attendance/discard_session", methods=["POST"])
def api_discard_session():
    """User chose Discard — throw away pending session, nothing is written."""
    global _pending_session
    count = len(_pending_session.get("records", []))
    _pending_session = {}
    return jsonify({"success": True, "message": f"Session discarded. {count} record(s) NOT saved to attendance."})


@app.route("/api/students/search", methods=["GET"])
def api_search_students():
    """Search students by ID or Name (case-insensitive partial match)."""
    query = request.args.get("q", "").strip().lower()
    df = get_students_df()
    if df.empty:
        return jsonify([])
    if query:
        mask = (df['ID'].astype(str).str.lower().str.contains(query, na=False) |
                df['NAME'].astype(str).str.lower().str.contains(query, na=False))
        df = df[mask]
    return jsonify(df.fillna('').to_dict('records'))


@app.route("/api/admin/clear_data", methods=["POST"])
def api_clear_data():
    """Delete all attendance records (Master + session CSVs). Students are kept."""
    data = request.json or {}
    what = data.get("what", "attendance")   # "attendance" | "all"

    removed = []

    # Always clear attendance CSVs
    if os.path.isdir(ATTENDANCE_DIR):
        for f in os.listdir(ATTENDANCE_DIR):
            if f.endswith('.csv'):
                try:
                    os.remove(os.path.join(ATTENDANCE_DIR, f))
                    removed.append(f)
                except Exception as e:
                    print('Clear error:', e)

    if what == "all":
        # Also remove student details + training images
        try:
            if os.path.isfile(STUDENT_DETAILS_CSV):
                os.remove(STUDENT_DETAILS_CSV)
                removed.append('StudentDetails.csv')
        except: pass
        if os.path.isdir(TRAINING_IMG_DIR):
            for f in os.listdir(TRAINING_IMG_DIR):
                try:
                    os.remove(os.path.join(TRAINING_IMG_DIR, f))
                    removed.append(f)
                except: pass
        try:
            if os.path.isfile(TRAINER_YML):
                os.remove(TRAINER_YML)
        except: pass

    return jsonify({"success": True,
                    "message": f"Cleared {len(removed)} file(s) successfully.",
                    "removed_count": len(removed)})


app.run(host="127.0.0.1", port=5000, debug=True)
