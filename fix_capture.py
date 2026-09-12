import re

with open('app_web_full.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = '''    # Open webcam to capture 50 face samples
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return jsonify({"success": False, "message": "Could not open Webcam hardware! Please check camera connection."}), 500

    sample_num = 0
    start_time = time.time()

    while sample_num < 50:
        ret, frame = cam.read()
        if not ret:
            break
        if time.time() - start_time > 35:  # 35s timeout
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            img_path = os.path.join(TRAINING_IMG_DIR, f"{st_name}.{serial}.{st_id}.{sample_num}.jpg")
            cv2.imwrite(img_path, gray[y:y + h, x:x + w])
            
            # Show progress on screen
            cv2.putText(frame, f"Capturing: {sample_num}/50", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if sample_num >= 50:
                break
                
        cv2.imshow(f"Registering Face - {st_name}", frame)
        cv2.waitKey(1)
        time.sleep(0.04)

    cam.release()
    cv2.destroyAllWindows()

    if sample_num > 0:
        train_model()
        return jsonify({
            "success": True,
            "message": f"Captured {sample_num}/50 face samples for {st_name} (ID: {st_id}) and successfully retrained AI face model!"
        })
    else:
        return jsonify({"success": False, "message": "No faces detected in camera view. Please align face with camera and try again."}), 400'''

new_block = '''    # Open webcam — EXACT port of old main.py take_student_images()
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
        return jsonify({"success": False, "message": f"No faces captured. Captured {sample_num} images. Make sure your face is clearly visible and well-lit, then try again."}), 400'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('app_web_full.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Capture block replaced with exact main.py port")
else:
    print("BLOCK NOT FOUND — searching for key lines...")
    lines = content.split('\n')
    for i, l in enumerate(lines):
        if '35s timeout' in l or 'time.time() - start_time' in l:
            print(f"Line {i+1}: {l}")
