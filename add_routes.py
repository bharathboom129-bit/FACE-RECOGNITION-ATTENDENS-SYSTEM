"""
Adds 4 new Flask routes to app_web_full.py:
1. /api/attendance/start_session  - runs cv2 tracking, returns found records without saving
2. /api/attendance/save_session   - user confirms: writes to CSV attendance files
3. /api/students/search           - search students by id/name
4. /api/admin/clear_data          - delete all attendance data
"""

with open('app_web_full.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_routes = '''
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

'''

# Insert before if __name__ == '__main__'
anchor = "if __name__ == '__main__'"
if anchor in content:
    content = content.replace(anchor, new_routes + '\n' + anchor)
    with open('app_web_full.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: All 6 new routes added to app_web_full.py")
else:
    # try app.run()
    anchor2 = "app.run("
    idx = content.rfind(anchor2)
    if idx != -1:
        content = content[:idx] + new_routes + '\n' + content[idx:]
        with open('app_web_full.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("SUCCESS (app.run anchor): routes added")
    else:
        # Append at end
        with open('app_web_full.py', 'a', encoding='utf-8') as f:
            f.write(new_routes)
        print("SUCCESS (appended at end): routes added")
