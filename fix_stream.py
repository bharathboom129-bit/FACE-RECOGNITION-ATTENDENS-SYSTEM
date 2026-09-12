import re

with open('app_web_full.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Insert MJPEG streaming route just before the capture_photos route
stream_route = '''@app.route("/api/students/live_feed")
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
            yield (b'--frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n' + buf.tobytes() + b'\\r\\n')
        cam.release()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


'''

target = '@app.route("/api/students/capture_photos", methods=["POST"])'
if target in content:
    content = content.replace(target, stream_route + target)
    with open('app_web_full.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: MJPEG live_feed route added")
else:
    print("ERROR: target route not found")
