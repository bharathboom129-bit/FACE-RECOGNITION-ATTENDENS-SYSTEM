############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os, sys, glob
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import urllib.parse
import webbrowser
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Custom Production Modules
from pdf_generator import generate_attendance_pdf
from email_notifier import send_defaulter_alert_email, send_hod_summary_email, get_email_config, update_email_config
from app_web import run_flask_in_background

############################################# CONSTANTS & PATHS ################################################
STUDENT_DETAILS_DIR = "StudentDetails"
STUDENT_DETAILS_CSV = os.path.join(STUDENT_DETAILS_DIR, "StudentDetails.csv")
ATTENDANCE_DIR = "Attendance"
MASTER_ATTENDANCE_CSV = os.path.join(ATTENDANCE_DIR, "Master_Attendance.csv")
TRAINING_IMG_DIR = "TrainingImage"
TRAINING_LABEL_DIR = "TrainingImageLabel"
TRAINER_YML = os.path.join(TRAINING_LABEL_DIR, "Trainner.yml")
PSD_TXT = os.path.join(TRAINING_LABEL_DIR, "psd.txt")
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
SUBJECTS_DIR = "Subjects"
SUBJECTS_CSV = os.path.join(SUBJECTS_DIR, "Subjects.csv")

############################################# INITIALIZATION ################################################
def assure_path_exists(path):
    dir_name = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def init_environment():
    assure_path_exists(STUDENT_DETAILS_DIR)
    assure_path_exists(ATTENDANCE_DIR)
    assure_path_exists(TRAINING_IMG_DIR)
    assure_path_exists(TRAINING_LABEL_DIR)
    assure_path_exists(SUBJECTS_DIR)

    # Initialize StudentDetails.csv if missing
    if not os.path.isfile(STUDENT_DETAILS_CSV):
        with open(STUDENT_DETAILS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SERIAL NO.', 'ID', 'NAME', 'CLASS_SUBJECT', 'REGISTRATION_DATE'])

    # Initialize Subjects.csv if missing
    if not os.path.isfile(SUBJECTS_CSV):
        with open(SUBJECTS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SUBJECT_NAME', 'TEACHER_NAME'])
            writer.writerow(['Java Programming', 'Prof. Smith'])
            writer.writerow(['Machine Learning', 'Dr. Sharma'])
            writer.writerow(['C++ & Data Structures', 'Prof. Kumar'])
            writer.writerow(['Web Technology', 'Prof. Roy'])

    # Initialize Master_Attendance.csv if missing
    if not os.path.isfile(MASTER_ATTENDANCE_CSV):
        with open(MASTER_ATTENDANCE_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'NAME', 'CLASS_SUBJECT', 'DATE', 'TIME', 'STATUS'])

init_environment()

############################################# HELPER FUNCTIONS ################################################

def check_haarcascadefile():
    if not os.path.isfile(HAAR_CASCADE_FILE):
        mess.showerror(title='System File Missing', message=f'Required file {HAAR_CASCADE_FILE} is missing!')
        return False
    return True

def get_admin_password():
    if os.path.isfile(PSD_TXT):
        with open(PSD_TXT, "r") as f:
            return f.read().strip()
    return None

def set_admin_password(new_pass):
    with open(PSD_TXT, "w") as f:
        f.write(new_pass)

def prompt_admin_password():
    pwd = get_admin_password()
    if pwd is None:
        new_pas = tsd.askstring('Setup Admin Password', 'No password set. Enter a new Admin Password below:', show='*')
        if new_pas:
            set_admin_password(new_pas)
            mess.showinfo('Success', 'Admin Password saved successfully!')
            return True
        else:
            mess.showwarning('Warning', 'Password setup cancelled.')
            return False
    else:
        entered = tsd.askstring('Admin Authentication', 'Enter Admin Password:', show='*')
        if entered == pwd:
            return True
        elif entered is None:
            return False
        else:
            mess.showerror('Error', 'Incorrect Password!')
            return False

def get_student_df():
    if os.path.isfile(STUDENT_DETAILS_CSV):
        try:
            df = pd.read_csv(STUDENT_DETAILS_CSV)
            # Ensure proper string columns
            df['ID'] = df['ID'].astype(str)
            df['NAME'] = df['NAME'].astype(str)
            return df
        except Exception as e:
            print("Error reading StudentDetails.csv:", e)
    return pd.DataFrame(columns=['SERIAL NO.', 'ID', 'NAME', 'CLASS_SUBJECT', 'REGISTRATION_DATE'])

def get_subjects_list():
    if os.path.isfile(SUBJECTS_CSV):
        try:
            df = pd.read_csv(SUBJECTS_CSV)
            return df['SUBJECT_NAME'].dropna().unique().tolist()
        except Exception as e:
            print("Error reading Subjects.csv:", e)
    return ['Java Programming', 'Machine Learning', 'C++ & Data Structures', 'General']

############################################# FACE TRAINING & DATASET ################################################

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            # Filename format:  Name.serial.USN.sampleNum.jpg
            # parts[1] = serial (always integer) — used as LBPH label
            # parts[2] = USN/ID (can be alphanumeric like '1AM24MC015') — NOT used as label
            filename = os.path.split(imagePath)[-1].lstrip()
            parts = filename.split(".")
            if len(parts) >= 4:
                serial_label = int(parts[1])   # serial is always a clean integer
                faces.append(imageNp)
                Ids.append(serial_label)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")
    return faces, Ids

def train_images():
    if not check_haarcascadefile():
        return False
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, Ids = getImagesAndLabels(TRAINING_IMG_DIR)
        if len(faces) == 0:
            mess.showwarning(title='No Face Data', message='No registered student images found to train model!')
            return False
        recognizer.train(faces, np.array(Ids))
        recognizer.save(TRAINER_YML)
        return True
    except Exception as e:
        mess.showerror(title='Training Error', message=f'Failed to train model: {e}')
        return False

def take_student_images(txt_id, txt_name, txt_class, lbl_msg):
    if not check_haarcascadefile():
        return
    
    student_id = txt_id.get().strip()
    name = txt_name.get().strip()
    class_subj = txt_class.get().strip() or "General"

    if not student_id or not name:
        lbl_msg.configure(text="Error: ID and Name are required!", fg="red")
        mess.showwarning("Input Error", "Please enter both Student ID and Name!")
        return

    if not (name.replace(" ", "").isalpha()):
        lbl_msg.configure(text="Error: Name must contain letters only!", fg="red")
        mess.showwarning("Input Error", "Student Name should contain only alphabetic characters!")
        return

    # Read existing student data
    df = get_student_df()
    if not df.empty and student_id in df['ID'].values:
        mess.showerror("Duplicate ID", f"Student ID '{student_id}' is already registered!")
        return

    serial = len(df) + 1
    reg_date = datetime.datetime.now().strftime("%d-%m-%Y")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess.showerror("Camera Error", "Could not open Webcam. Please check camera connection!")
        return

    detector = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    sampleNum = 0
    lbl_msg.configure(text="Capturing images... Look at camera!", fg="blue")

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            # Save captured image
            img_path = os.path.join(TRAINING_IMG_DIR, f"{name}.{serial}.{student_id}.{sampleNum}.jpg")
            cv2.imwrite(img_path, gray[y:y + h, x:x + w])
            cv2.putText(img, f"Captured: {sampleNum}/50", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Taking Student Face Registration - Press 'Q' to Cancel", img)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        if sampleNum >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sampleNum > 0:
        # Append student to CSV
        new_row = [serial, student_id, name, class_subj, reg_date]
        with open(STUDENT_DETAILS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

        lbl_msg.configure(text=f"Images captured for ID: {student_id}. Training model...", fg="blue")
        if train_images():
            lbl_msg.configure(text=f"Success: Registered {name} (ID: {student_id})!", fg="green")
            mess.showinfo("Registration Complete", f"Student {name} (ID: {student_id}) successfully registered!")
            refresh_student_table()
            refresh_registration_count()
        else:
            lbl_msg.configure(text="Face model training failed!", fg="red")
    else:
        lbl_msg.configure(text="Registration cancelled or no faces detected.", fg="orange")

############################################# STUDENT CRUD (EDIT / DELETE) ################################################

def delete_student_action():
    if not prompt_admin_password():
        return

    selected = student_tree.selection()
    if not selected:
        mess.showwarning("Selection Error", "Please select a student from the table to delete.")
        return

    item = student_tree.item(selected[0])
    vals = item['values']
    student_id = str(vals[1])
    student_name = str(vals[2])

    if mess.askyesno("Confirm Delete", f"Are you sure you want to delete student {student_name} (ID: {student_id})?\nThis will remove face dataset and retrain model."):
        # Remove from CSV
        df = get_student_df()
        df = df[df['ID'].astype(str) != student_id]
        # Re-assign serial numbers
        df['SERIAL NO.'] = range(1, len(df) + 1)
        df.to_csv(STUDENT_DETAILS_CSV, index=False)

        # Delete image files
        imagePaths = glob.glob(os.path.join(TRAINING_IMG_DIR, f"*.{student_id}.*.jpg"))
        for p in imagePaths:
            try:
                os.remove(p)
            except Exception as e:
                print("Error deleting image:", e)

        # Retrain model
        train_images()
        refresh_student_table()
        refresh_registration_count()
        mess.showinfo("Deleted", f"Student {student_name} deleted successfully.")

def edit_student_action():
    if not prompt_admin_password():
        return

    selected = student_tree.selection()
    if not selected:
        mess.showwarning("Selection Error", "Please select a student from the table to edit.")
        return

    item = student_tree.item(selected[0])
    vals = item['values']
    current_id = str(vals[1])
    current_name = str(vals[2])
    current_class = str(vals[3]) if len(vals) > 3 else "General"

    # Dialog for editing
    edit_win = tk.Toplevel(window)
    edit_win.title("Edit Student Details")
    edit_win.geometry("380x250")
    edit_win.resizable(False, False)

    tk.Label(edit_win, text="Edit Student Details", font=('times', 14, 'bold')).pack(pady=10)
    
    frame = tk.Frame(edit_win)
    frame.pack(padx=20, pady=10)

    tk.Label(frame, text="Student ID:", font=('times', 11)).grid(row=0, column=0, sticky='e', pady=5)
    entry_id = tk.Entry(frame, font=('times', 11))
    entry_id.insert(0, current_id)
    entry_id.grid(row=0, column=1, pady=5)

    tk.Label(frame, text="Name:", font=('times', 11)).grid(row=1, column=0, sticky='e', pady=5)
    entry_name = tk.Entry(frame, font=('times', 11))
    entry_name.insert(0, current_name)
    entry_name.grid(row=1, column=1, pady=5)

    tk.Label(frame, text="Class/Subject:", font=('times', 11)).grid(row=2, column=0, sticky='e', pady=5)
    entry_class = tk.Entry(frame, font=('times', 11))
    entry_class.insert(0, current_class)
    entry_class.grid(row=2, column=1, pady=5)

    def save_edits():
        new_id = entry_id.get().strip()
        new_name = entry_name.get().strip()
        new_cls = entry_class.get().strip()

        if not new_id or not new_name:
            mess.showwarning("Input Error", "ID and Name cannot be empty.")
            return

        df = get_student_df()
        idx = df[df['ID'].astype(str) == current_id].index
        if not idx.empty:
            df.loc[idx, 'ID'] = new_id
            df.loc[idx, 'NAME'] = new_name
            df.loc[idx, 'CLASS_SUBJECT'] = new_cls
            df.to_csv(STUDENT_DETAILS_CSV, index=False)
            
            refresh_student_table()
            edit_win.destroy()
            mess.showinfo("Success", "Student details updated successfully!")

    tk.Button(edit_win, text="Save Changes", command=save_edits, bg="#3ece48", fg="white", font=('times', 11, 'bold')).pack(pady=15)

############################################# ATTENDANCE RECOGNITION SESSION ################################################

def TrackImages():
    if not check_haarcascadefile():
        return

    if not os.path.isfile(TRAINER_YML):
        mess.showerror('Missing Model', 'No trained model found! Please register a student and save profile first.')
        return

    subject_selected = subject_combo.get().strip() or "General"

    # Read registered students dataframe
    df_students = get_student_df()
    if df_students.empty:
        mess.showwarning('No Students', 'No student records found in database!')
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(TRAINER_YML)
    except Exception as e:
        mess.showerror('Model Error', f'Error reading face recognizer model: {e}')
        return

    faceCascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess.showerror("Camera Error", "Webcam could not be opened!")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    session_start_dt = datetime.datetime.now()
    date_str = session_start_dt.strftime('%d-%m-%Y')
    time_str = session_start_dt.strftime('%H:%M:%S')
    
    # Clean Session Deduplication Set
    marked_students = {}  # student_id -> dict(id, name, class, date, time)

    mess.showinfo("Starting Attendance Session", 
                  f"Class/Subject: {subject_selected}\nSession Date: {date_str} {time_str}\n\nPress 'Q' on camera window when finished taking attendance.")

    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

            # Confidence threshold (< 65 is recognized for LBPH)
            if conf < 65:
                # Recognizer returns the serial label (integer) — look up by SERIAL NO.
                matched = df_students[df_students['SERIAL NO.'].astype(int) == int(serial)]

                if not matched.empty:
                    # Clean scalar string extraction (eliminates numpy array string noise!)
                    student_id = str(matched.iloc[0]['ID'])
                    student_name = str(matched.iloc[0]['NAME'])
                    student_class = str(matched.iloc[0]['CLASS_SUBJECT']) if 'CLASS_SUBJECT' in matched.columns else subject_selected

                    disp_text = f"{student_name} ({student_id})"
                    color = (0, 255, 0)  # Green box for recognized

                    # Add to session attendance if not already marked
                    if student_id not in marked_students:
                        now_t = datetime.datetime.now().strftime('%H:%M:%S')
                        marked_students[student_id] = {
                            'ID': student_id,
                            'NAME': student_name,
                            'CLASS_SUBJECT': subject_selected,
                            'DATE': date_str,
                            'TIME': now_t,
                            'STATUS': 'Present'
                        }
                else:
                    disp_text = "Unknown Student"
                    color = (0, 0, 255)
            else:
                disp_text = "Unknown"
                color = (0, 0, 255)

            # Draw rectangle and clean text box
            cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(im, (x, y - 35), (x + w, y), color, cv2.FILLED)
            cv2.putText(im, disp_text, (x + 5, y - 10), font, 0.65, (255, 255, 255), 2)

        # Header overlay
        header_str = f"Subject: {subject_selected} | Present: {len(marked_students)} | Press 'Q' to Finish"
        cv2.putText(im, header_str, (10, 30), font, 0.7, (255, 255, 0), 2)

        cv2.imshow(f"CCTV Attendance Session - {subject_selected}", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save Session CSV and Master Log if students were present
    if marked_students:
        session_time_suffix = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        clean_subj_name = subject_selected.replace(" ", "_")
        session_filename = f"Attendance_{clean_subj_name}_{session_time_suffix}.csv"
        session_filepath = os.path.join(ATTENDANCE_DIR, session_filename)

        # Save timestamped session file
        session_df = pd.DataFrame(list(marked_students.values()))
        session_df.to_csv(session_filepath, index=False)

        # Append to Master_Attendance.csv cleanly
        master_exists = os.path.isfile(MASTER_ATTENDANCE_CSV)
        session_df.to_csv(MASTER_ATTENDANCE_CSV, mode='a', header=not master_exists, index=False)

        # Also update today's day file Attendance_DD-MM-YYYY.csv
        day_filepath = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
        if os.path.isfile(day_filepath):
            try:
                day_df = pd.read_csv(day_filepath)
                combined = pd.concat([day_df, session_df]).drop_duplicates(subset=['ID', 'DATE'])
                combined.to_csv(day_filepath, index=False)
            except Exception:
                session_df.to_csv(day_filepath, index=False)
        else:
            session_df.to_csv(day_filepath, index=False)

        # Update Treeview in GUI
        update_live_attendance_table(session_df)
        mess.showinfo("Session Completed", 
                      f"Attendance Session Saved!\n\nSubject: {subject_selected}\nPresent Students: {len(marked_students)}\nFile Saved: {session_filename}")
        
        # Trigger Analytics Refresh
        refresh_analytics_dashboard()
    else:
        mess.showinfo("Session Ended", "No students detected or marked present during this session.")

############################################# WHATSAPP & SHARE FUNCTIONS ################################################

def share_whatsapp_attendance():
    selected_subject = subject_combo.get().strip() or "General"
    date_str = datetime.datetime.now().strftime("%d-%m-%Y")
    time_str = datetime.datetime.now().strftime("%H:%M:%S")

    # Read latest session or today's attendance
    day_filepath = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
    present_df = pd.DataFrame()
    if os.path.isfile(day_filepath):
        try:
            present_df = pd.read_csv(day_filepath)
        except Exception as e:
            print("Error reading day attendance:", e)

    students_df = get_student_df()
    total_registered = len(students_df)
    total_present = len(present_df)
    total_absent = max(0, total_registered - total_present)

    # Format text message
    lines = [
        "📋 *SMART CLASSROOM ATTENDANCE REPORT*",
        f"📚 *Subject/Class:* {selected_subject}",
        f"📅 *Date:* {date_str} | *Time:* {time_str}",
        f"👥 *Total Capacity:* {total_registered}",
        f"✅ *Present Count:* {total_present}",
        f"❌ *Absent Count:* {total_absent}",
        f"📊 *Attendance Rate:* {((total_present/total_registered)*100 if total_registered>0 else 0):.1f}%",
        "",
        "*Present Students List:*"
    ]

    if not present_df.empty:
        for idx, row in present_df.iterrows():
            st_id = row.get('ID', '')
            st_name = row.get('NAME', '')
            st_time = row.get('TIME', '')
            lines.append(f"{idx+1}. `{st_id}` - *{st_name}* (at {st_time})")
    else:
        lines.append("_No attendance logged for today yet._")

    full_message = "\n".join(lines)

    # Copy to clipboard
    window.clipboard_clear()
    window.clipboard_append(full_message)

    # Encode URL for WhatsApp Web
    encoded_text = urllib.parse.quote(full_message)
    whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_text}"

    if mess.askyesno("Share via WhatsApp", "Report text has been copied to your clipboard!\n\nWould you like to open WhatsApp Web in your browser to send it to the Teacher/Group?"):
        webbrowser.open(whatsapp_url)

def open_csv_folder():
    if os.path.exists(ATTENDANCE_DIR):
        try:
            os.startfile(os.path.abspath(ATTENDANCE_DIR))
        except Exception as e:
            mess.showerror("Error", f"Could not open directory: {e}")

############################################# SUBJECT MANAGEMENT ################################################

def add_subject_action():
    subj_name = entry_subj.get().strip()
    teacher_name = entry_teacher.get().strip() or "Class Instructor"

    if not subj_name:
        mess.showwarning("Input Error", "Please enter Subject Name!")
        return

    subjects = get_subjects_list()
    if subj_name in subjects:
        mess.showwarning("Duplicate", "Subject already exists!")
        return

    with open(SUBJECTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([subj_name, teacher_name])

    entry_subj.delete(0, 'end')
    entry_teacher.delete(0, 'end')
    refresh_subjects_list()
    mess.showinfo("Success", f"Subject '{subj_name}' added successfully!")


def edit_subject_action():
    selected = subjects_tree.selection()
    if not selected:
        mess.showwarning("Selection Error", "Please select a subject from the list to edit.")
        return

    item = subjects_tree.item(selected[0])
    vals = item['values']
    current_name = str(vals[1])
    current_teacher = str(vals[2]) if len(vals) > 2 else ""

    edit_win = tk.Toplevel(window)
    edit_win.title("Edit Subject")
    edit_win.geometry("400x230")
    edit_win.resizable(False, False)
    edit_win.configure(bg="#2b2b2b")

    tk.Label(edit_win, text="Edit Subject Details", font=('times', 15, 'bold'), bg="#2b2b2b", fg="#00aeff").pack(pady=15)

    frm = tk.Frame(edit_win, bg="#2b2b2b")
    frm.pack(padx=20, fill='x')

    tk.Label(frm, text="Subject Name:", font=('times', 11, 'bold'), bg="#2b2b2b", fg="white").grid(row=0, column=0, sticky='e', pady=8)
    e_name = tk.Entry(frm, font=('times', 11), width=28)
    e_name.insert(0, current_name)
    e_name.grid(row=0, column=1, pady=8, padx=10)

    tk.Label(frm, text="Instructor:", font=('times', 11, 'bold'), bg="#2b2b2b", fg="white").grid(row=1, column=0, sticky='e', pady=8)
    e_teacher = tk.Entry(frm, font=('times', 11), width=28)
    e_teacher.insert(0, current_teacher)
    e_teacher.grid(row=1, column=1, pady=8, padx=10)

    def save_subject_edit():
        new_name = e_name.get().strip()
        new_teacher = e_teacher.get().strip() or "Class Instructor"
        if not new_name:
            mess.showwarning("Input Error", "Subject name cannot be empty.")
            return
        try:
            df = pd.read_csv(SUBJECTS_CSV)
            idx = df[df['SUBJECT_NAME'] == current_name].index
            if not idx.empty:
                df.loc[idx, 'SUBJECT_NAME'] = new_name
                df.loc[idx, 'TEACHER_NAME'] = new_teacher
                df.to_csv(SUBJECTS_CSV, index=False)
            refresh_subjects_list()
            edit_win.destroy()
            mess.showinfo("Updated", f"Subject updated to '{new_name}' successfully!")
        except Exception as ex:
            mess.showerror("Error", f"Failed to update: {ex}")

    tk.Button(edit_win, text="💾 Save Changes", command=save_subject_edit,
              bg="#3ece48", fg="white", font=('times', 12, 'bold')).pack(pady=15, ipadx=10)


def delete_subject_action():
    selected = subjects_tree.selection()
    if not selected:
        mess.showwarning("Selection Error", "Please select a subject from the list to delete.")
        return

    item = subjects_tree.item(selected[0])
    vals = item['values']
    subj_name = str(vals[1])

    if mess.askyesno("Confirm Delete", f"Are you sure you want to delete subject '{subj_name}'?"):
        try:
            df = pd.read_csv(SUBJECTS_CSV)
            df = df[df['SUBJECT_NAME'] != subj_name]
            df.to_csv(SUBJECTS_CSV, index=False)

            # Purge deleted subject from Master_Attendance.csv
            if os.path.isfile(MASTER_ATTENDANCE_CSV):
                df_m = pd.read_csv(MASTER_ATTENDANCE_CSV)
                df_m = df_m[df_m['CLASS_SUBJECT'] != subj_name]
                df_m.to_csv(MASTER_ATTENDANCE_CSV, index=False)

            refresh_subjects_list()
            refresh_analytics_dashboard()
            mess.showinfo("Deleted", f"Subject '{subj_name}' deleted successfully.")
        except Exception as ex:
            mess.showerror("Error", f"Failed to delete: {ex}")

def refresh_subjects_list():
    subjs = get_subjects_list()
    subject_combo['values'] = subjs
    if subjs:
        subject_combo.current(0)

    for item in subjects_tree.get_children():
        subjects_tree.delete(item)

    if os.path.isfile(SUBJECTS_CSV):
        try:
            df = pd.read_csv(SUBJECTS_CSV)
            for idx, r in df.iterrows():
                tag = 'even' if idx % 2 == 0 else 'odd'
                subjects_tree.insert('', 'end', values=(idx + 1, r.get('SUBJECT_NAME', ''), r.get('TEACHER_NAME', '')), tags=(tag,))
        except Exception as e:
            print(e)

############################################# ANALYTICAL DASHBOARD ENGINE ################################################

def _get_master_df():
    """Load Master_Attendance.csv, filter by active subjects, and assign SESSION_KEY
    using a 10-minute gap clustering algorithm (handles multiple sessions in the same hour)."""
    if os.path.isfile(MASTER_ATTENDANCE_CSV):
        try:
            df = pd.read_csv(MASTER_ATTENDANCE_CSV)
            if df.empty:
                return pd.DataFrame(columns=['ID','NAME','CLASS_SUBJECT','DATE','TIME','STATUS','SESSION_KEY'])

            # Filter out deleted/obsolete subjects not in active Subjects catalog
            active_subjs = get_subjects_list()
            if active_subjs and 'CLASS_SUBJECT' in df.columns:
                df = df[df['CLASS_SUBJECT'].isin(active_subjs)]

            if df.empty:
                return pd.DataFrame(columns=['ID','NAME','CLASS_SUBJECT','DATE','TIME','STATUS','SESSION_KEY'])

            # Convert DATE + TIME into DATETIME for gap-based session clustering
            if 'DATE' in df.columns and 'TIME' in df.columns:
                df['DATETIME'] = pd.to_datetime(
                    df['DATE'].astype(str) + ' ' + df['TIME'].astype(str),
                    format='%d-%m-%Y %H:%M:%S',
                    errors='coerce'
                )
            else:
                df['DATETIME'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')

            df = df.sort_values('DATETIME')

            # Cluster entries into sessions: gap > 10 mins (600s) = NEW SESSION
            session_keys = []
            last_seen = {}
            sess_counter = {}

            for _, row in df.iterrows():
                subj = str(row.get('CLASS_SUBJECT', ''))
                dt_val = row['DATETIME']
                dt_str = dt_val.strftime('%Y-%m-%d') if pd.notnull(dt_val) else str(row.get('DATE', ''))
                key = (subj, dt_str)

                if key not in last_seen or pd.isnull(dt_val) or (dt_val - last_seen[key]).total_seconds() > 600:
                    sess_counter[key] = sess_counter.get(key, 0) + 1
                    if pd.notnull(dt_val):
                        last_seen[key] = dt_val

                session_keys.append(f"{subj}|{dt_str}|S{sess_counter[key]}")

            df['SESSION_KEY'] = session_keys

            # Parse DATE column back to datetime object for Matplotlib chart filtering
            df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')

            # Deduplicate: keep only one record per student per session slot
            df = df.drop_duplicates(subset=['ID', 'SESSION_KEY'])

            return df
        except Exception as e:
            print("Master Attendance read error:", e)
    return pd.DataFrame(columns=['ID','NAME','CLASS_SUBJECT','DATE','TIME','STATUS','SESSION_KEY'])


def _draw_dark_chart(fig):
    """Apply dark theme to a Matplotlib figure."""
    fig.patch.set_facecolor('#1a1918')
    for ax in fig.get_axes():
        ax.set_facecolor('#252525')
        ax.tick_params(colors='#cccccc', labelsize=9)
        ax.title.set_color('white')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#444444')


def _embed_figure(fig, parent):
    """Embed matplotlib figure into a tkinter frame."""
    for w in parent.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    plt.close(fig)


# -------- Panel 1: Overview --------
def refresh_analytics_overview():
    df_students = get_student_df()
    total_registered = len(df_students)
    df = _get_master_df()

    total_sessions = 0
    avg_att = 0.0
    total_subjects = 0
    defaulters = 0

    if not df.empty:
        # Sessions = distinct (Subject + Date + Hour) slots — supports 2 classes same day
        total_sessions = df['SESSION_KEY'].nunique()
        total_subjects = df['CLASS_SUBJECT'].nunique()
        # Average attendance = mean sessions attended per student / total sessions
        p_counts = df.groupby('ID')['SESSION_KEY'].nunique()
        avg_att = (p_counts.mean() / total_sessions * 100) if total_sessions > 0 else 0.0
        if total_registered > 0:
            for _, row in df_students.iterrows():
                pc = df[df['ID'].astype(str) == str(row['ID'])]['SESSION_KEY'].nunique()
                if total_sessions > 0 and (pc / total_sessions * 100) < 75:
                    defaulters += 1

    # Update KPI cards
    card_total_students.config(text=str(total_registered))
    card_total_sessions.config(text=str(total_sessions))
    card_total_subjects.config(text=str(total_subjects))
    card_avg_att.config(text=f"{avg_att:.1f}%")
    card_defaulters.config(text=str(defaulters))

    # Charts — subject-wise uses unique session slots per subject
    subj_names, subj_rates = [], []
    if not df.empty and total_registered > 0:
        for s_name, grp in df.groupby('CLASS_SUBJECT'):
            subj_sessions = grp['SESSION_KEY'].nunique()   # actual class slots for this subject
            denom = total_registered * subj_sessions if subj_sessions > 0 else 1
            # count unique (student, session) pairs attended
            attended = grp.groupby('SESSION_KEY')['ID'].nunique().sum()
            rate = min(100.0, attended / denom * 100.0)
            subj_names.append(s_name[:14])
            subj_rates.append(round(rate, 1))

    if not subj_names:
        subj_names = ['No Data']
        subj_rates = [0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2), dpi=95)

    # Bar: subject-wise attendance
    colors = ['#00aeff' if r >= 75 else '#ea2a2a' for r in subj_rates]
    bars = ax1.bar(subj_names, subj_rates, color=colors, width=0.55, edgecolor='#333333')
    ax1.set_title('Subject-wise Attendance Rate (%)', fontweight='bold')
    ax1.set_ylabel('Attendance %')
    ax1.set_ylim(0, 110)
    ax1.axhline(75, color='orange', linewidth=1.2, linestyle='--', label='75% threshold')
    ax1.legend(fontsize=8, facecolor='#2b2b2b', labelcolor='white')
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.0f}%', xy=(bar.get_x()+bar.get_width()/2, h),
                     xytext=(0,3), textcoords='offset points', ha='center', color='white', fontsize=9)

    # Pie: Present vs Absent overall
    avg_rate = np.mean(subj_rates) if subj_rates else 0
    ax2.pie([max(avg_rate, 0.1), max(100-avg_rate, 0.1)],
            labels=['Present', 'Absent'],
            colors=['#3ece48', '#ea2a2a'],
            autopct='%1.1f%%', startangle=90,
            textprops={'color': 'white', 'fontsize': 10},
            wedgeprops={'edgecolor': '#1a1918', 'linewidth': 2})
    ax2.set_title('Overall Attendance Ratio', fontweight='bold')

    _draw_dark_chart(fig)
    fig.tight_layout(pad=2)
    _embed_figure(fig, overview_chart_frame)

    # Refresh student tracker table
    for item in student_stats_tree.get_children():
        student_stats_tree.delete(item)

    if not df_students.empty:
        for _, row in df_students.iterrows():
            st_id = str(row['ID'])
            st_name = str(row['NAME'])
            st_cls = str(row.get('CLASS_SUBJECT', ''))
            # Count unique session slots the student attended (not raw row count)
            p_cnt = df[df['ID'].astype(str) == st_id]['SESSION_KEY'].nunique() if not df.empty else 0
            att_pct = (p_cnt / total_sessions * 100.0) if total_sessions > 0 else 0.0
            status = '✅ Regular' if att_pct >= 75 else '⚠️ DEFAULTER (<75%)'
            iid = student_stats_tree.insert('', 'end', values=(st_id, st_name, st_cls, f"{p_cnt}/{total_sessions}", f"{att_pct:.1f}%", status))
            if att_pct < 75:
                student_stats_tree.item(iid, tags=('defaulter',))
            else:
                student_stats_tree.item(iid, tags=('ok',))

    student_stats_tree.tag_configure('defaulter', background='#5a1a1a', foreground='#ff6b6b')
    student_stats_tree.tag_configure('ok', background='#1a3a1a', foreground='#6bff6b')


# -------- Panel 2: Weekly Trend --------
def refresh_weekly_panel():
    df = _get_master_df()
    for w in weekly_chart_frame.winfo_children():
        w.destroy()

    if df.empty or 'DATE' not in df.columns:
        tk.Label(weekly_chart_frame, text='No attendance data yet.', bg='#1a1918', fg='#888888', font=('Helvetica', 13)).pack(expand=True)
        return

    df = df.dropna(subset=['DATE'])
    df['WEEK'] = df['DATE'].dt.isocalendar().week.astype(int)
    df['YEAR'] = df['DATE'].dt.year
    df['WEEK_LABEL'] = df['DATE'].dt.strftime('W%U\n%d %b')

    weekly_counts = df.groupby(['YEAR','WEEK','WEEK_LABEL'])['ID'].count().reset_index()
    weekly_counts = weekly_counts.sort_values(['YEAR','WEEK']).tail(8)  # last 8 weeks

    fig, ax = plt.subplots(figsize=(9.5, 3.5), dpi=95)
    weeks = weekly_counts['WEEK_LABEL'].tolist()
    counts = weekly_counts['ID'].tolist()
    bars = ax.bar(weeks, counts, color='#00aeff', width=0.5, edgecolor='#333')
    ax.plot(weeks, counts, color='#f5a623', marker='o', linewidth=2, markersize=6, label='Trend')
    ax.set_title('Weekly Attendance Count (Last 8 Weeks)', fontweight='bold')
    ax.set_ylabel('Number of Attendance Records')
    ax.legend(facecolor='#2b2b2b', labelcolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(str(int(h)), xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0,3), textcoords='offset points', ha='center', color='white', fontsize=9)
    _draw_dark_chart(fig)
    fig.tight_layout()
    _embed_figure(fig, weekly_chart_frame)


# -------- Panel 3: Monthly Trend --------
def refresh_monthly_panel():
    df = _get_master_df()
    for w in monthly_chart_frame.winfo_children():
        w.destroy()

    if df.empty or 'DATE' not in df.columns:
        tk.Label(monthly_chart_frame, text='No attendance data yet.', bg='#1a1918', fg='#888888', font=('Helvetica', 13)).pack(expand=True)
        return

    df = df.dropna(subset=['DATE'])
    df['MONTH_LABEL'] = df['DATE'].dt.strftime('%b %Y')
    df['MONTH_NUM'] = df['DATE'].dt.to_period('M')
    monthly = df.groupby('MONTH_NUM')['ID'].count().reset_index()
    monthly = monthly.sort_values('MONTH_NUM')
    labels = [str(p) for p in monthly['MONTH_NUM']]
    labels_short = [pd.Period(p, freq='M').strftime('%b\n%Y') for p in labels]
    counts = monthly['ID'].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5), dpi=95)
    # Bar
    ax1.bar(labels_short, counts, color='#9b59b6', width=0.5, edgecolor='#333')
    ax1.set_title('Monthly Attendance Volume', fontweight='bold')
    ax1.set_ylabel('Attendance Records')
    # Line trend
    ax2.plot(labels_short, counts, color='#00aeff', marker='o', linewidth=2.5, markersize=8)
    ax2.fill_between(range(len(counts)), counts, alpha=0.15, color='#00aeff')
    ax2.set_xticks(range(len(labels_short)))
    ax2.set_xticklabels(labels_short)
    ax2.set_title('Monthly Trend Line', fontweight='bold')
    ax2.set_ylabel('Attendance Records')
    _draw_dark_chart(fig)
    fig.tight_layout()
    _embed_figure(fig, monthly_chart_frame)


# -------- Panel 4: Subject Drill-Down --------
def refresh_subject_panel():
    df = _get_master_df()
    df_students = get_student_df()
    total_registered = len(df_students)

    for item in subject_drill_tree.get_children():
        subject_drill_tree.delete(item)

    if df.empty or total_registered == 0:
        return

    for s_name, grp in df.groupby('CLASS_SUBJECT'):
        # Use unique session slots (Subject+Date+Hour) not just unique dates
        sessions = grp['SESSION_KEY'].nunique()
        present_students = grp['ID'].nunique()
        # Total possible attendances = registered students × session slots
        denom = total_registered * sessions if sessions > 0 else 1
        attended = grp.groupby('SESSION_KEY')['ID'].nunique().sum()
        rate = min(100.0, attended / denom * 100)
        color_tag = 'good_row' if rate >= 75 else 'warn_row'
        subject_drill_tree.insert('', 'end',
            values=(s_name, sessions, present_students, total_registered, f"{rate:.1f}%"),
            tags=(color_tag,))

    subject_drill_tree.tag_configure('good_row', background='#1a3a1a', foreground='#6bff6b')
    subject_drill_tree.tag_configure('warn_row', background='#3a1a1a', foreground='#ff9f9f')


def on_subject_drill_click(event):
    """When user clicks a subject row — show all session CSV files for it."""
    selected = subject_drill_tree.selection()
    if not selected:
        return
    vals = subject_drill_tree.item(selected[0])['values']
    raw_subj = str(vals[0]).strip()
    subj_slug = raw_subj.replace(' ', '_').replace('/', '_').lower()

    # Find matching session CSV files with exact subject slug matching
    csv_files = []
    if os.path.isdir(ATTENDANCE_DIR):
        for f in os.listdir(ATTENDANCE_DIR):
            if f.startswith('Attendance_') and f.endswith('.csv') and f != 'Master_Attendance.csv':
                core = f[11:-4]  # Strip "Attendance_" and ".csv"
                parts = core.split('_')
                if len(parts) >= 3:
                    file_subj_slug = '_'.join(parts[:-2]).lower()
                    if file_subj_slug == subj_slug:
                        csv_files.append(f)

    for item in session_files_tree.get_children():
        session_files_tree.delete(item)

    if csv_files:
        session_lbl.config(text=f"Session Files for: {vals[0]} ({len(csv_files)} sessions found)")
        total_sessions_cnt = len(csv_files)
        total_registered = len(get_student_df())

        for i, fname in enumerate(sorted(csv_files)):
            fpath = os.path.join(ATTENDANCE_DIR, fname)
            # Parse Date & Time from filename e.g. "Attendance_Python_12-09-2026_18-03-21.csv"
            dt_str = "Unknown"
            core = fname[11:-4]
            parts = core.split('_')
            if len(parts) >= 2:
                d_part = parts[-2]
                t_part = parts[-1].replace('-', ':')
                dt_str = f"{d_part} {t_part}"

            row_cnt = 0
            try:
                fdf = pd.read_csv(fpath)
                row_cnt = len(fdf)
            except:
                pass

            session_slot = f"Session {i+1} of {total_sessions_cnt}"
            present_ratio = f"{row_cnt} / {total_registered}" if total_registered > 0 else f"{row_cnt}"
            pct_val = (row_cnt / total_registered * 100.0) if total_registered > 0 else 0.0
            pct_str = f"{pct_val:.1f}%"

            tag = 'even' if i % 2 == 0 else 'odd'
            session_files_tree.insert('', 'end', values=(
                i+1, session_slot, dt_str, present_ratio, pct_str, fname, fpath
            ), tags=(tag,))
    else:
        session_lbl.config(text=f"No session files found for: {vals[0]}")

    session_files_tree.tag_configure('even', background='#1e1e1e', foreground='#f0f0f0')
    session_files_tree.tag_configure('odd', background='#252525', foreground='#f0f0f0')


def open_session_csv(event):
    selected = session_files_tree.selection()
    if not selected:
        return
    vals = session_files_tree.item(selected[0])['values']
    fpath = str(vals[6])
    if os.path.isfile(fpath):
        os.startfile(fpath)


# -------- Panel 5: Class/Period-wise Analytics --------
def refresh_classwise_panel():
    df = _get_master_df()
    for w in classwise_chart_frame.winfo_children():
        w.destroy()

    if df.empty:
        tk.Label(classwise_chart_frame, text='No attendance data yet.', bg='#1a1918', fg='#888888', font=('Helvetica', 13)).pack(expand=True)
        return

    # Period breakdown by hour (from TIME column)
    period_counts = {}
    if 'TIME' in df.columns:
        for t in df['TIME'].dropna():
            try:
                hour = int(str(t).split(':')[0])
                label = f"{hour:02d}:00"
                period_counts[label] = period_counts.get(label, 0) + 1
            except:
                pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5), dpi=95)

    # Chart 1: Hour/Period-wise attendance volume
    if period_counts:
        periods = sorted(period_counts.keys())
        pcounts = [period_counts[p] for p in periods]
        cmap = plt.cm.get_cmap('Blues', len(periods) + 3)
        bar_colors = [cmap(i + 3) for i in range(len(periods))]
        ax1.bar(periods, pcounts, color=bar_colors, edgecolor='#333')
        ax1.set_title('Attendance by Class Hour (Period-wise)', fontweight='bold')
        ax1.set_ylabel('Students Recorded')
        ax1.set_xlabel('Hour of Day')
    else:
        ax1.text(0.5, 0.5, 'No time-slot data', ha='center', va='center', color='gray', transform=ax1.transAxes)
        ax1.set_title('Period-wise Attendance', fontweight='bold')

    # Chart 2: Day-of-week distribution
    if 'DATE' in df.columns:
        df2 = df.dropna(subset=['DATE'])
        df2 = df2.copy()
        df2['DOW'] = df2['DATE'].dt.day_name()
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow_counts = df2['DOW'].value_counts().reindex(dow_order, fill_value=0)
        colors_dow = ['#00aeff','#3ece48','#f5a623','#ea2a2a','#9b59b6','#1abc9c','#e74c3c']
        ax2.bar(dow_counts.index, dow_counts.values, color=colors_dow[:len(dow_counts)], edgecolor='#333')
        ax2.set_title('Attendance by Day of Week', fontweight='bold')
        ax2.set_ylabel('Records')
        ax2.set_xlabel('Day')
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No date data', ha='center', va='center', color='gray', transform=ax2.transAxes)

    _draw_dark_chart(fig)
    fig.tight_layout()
    _embed_figure(fig, classwise_chart_frame)


def export_pdf_report_action():
    """Generate and open official PDF Attendance Report."""
    try:
        pdf_path = generate_attendance_pdf()
        if os.path.exists(pdf_path):
            os.startfile(pdf_path)
            mess.showinfo("PDF Generated", f"Official PDF Attendance Report generated successfully!\nSaved to: {pdf_path}")
        else:
            mess.showerror("PDF Error", "Failed to generate PDF report file.")
    except Exception as e:
        mess.showerror("PDF Error", f"Error creating PDF: {e}")

def send_email_defaulters_action():
    """Send HTML warning emails to students & parents with <75% attendance."""
    df_students = get_student_df()
    df_master = _get_master_df()
    total_sessions = df_master['SESSION_KEY'].nunique() if not df_master.empty and 'SESSION_KEY' in df_master.columns else 0

    if df_students.empty or total_sessions == 0:
        mess.showwarning("Email Warning", "No student records or class sessions available to send alerts.")
        return

    config = get_email_config()
    if not config["SENDER_EMAIL"] or not config["SENDER_PASSWORD"] or "app_password" in config["SENDER_PASSWORD"]:
        if mess.askyesno("Email Config Required", "Email settings are not configured yet! Would you like to configure your Gmail & App Password now?"):
            open_email_config_dialog()
        return

    sent_count = 0
    fail_count = 0

    for _, row in df_students.iterrows():
        st_id = str(row['ID'])
        st_name = str(row['NAME'])
        p_email = row.get('PARENT_EMAIL', '')
        p_cnt = df_master[df_master['ID'].astype(str) == st_id]['SESSION_KEY'].nunique() if not df_master.empty and 'SESSION_KEY' in df_master.columns else 0
        att_pct = (p_cnt / total_sessions * 100.0)

        if att_pct < 75.0:
            success, msg = send_defaulter_alert_email(st_id, st_name, p_email, att_pct, total_sessions, p_cnt)
            if success:
                sent_count += 1
            else:
                fail_count += 1

    if sent_count > 0:
        mess.showinfo("Email Alerts Sent", f"Successfully sent {sent_count} HTML warning email alerts to defaulter parents/students!")
    elif fail_count > 0:
        mess.showerror("Email Alert Failed", "Failed to send email alerts. Please verify your SMTP / App Password settings.")
    else:
        mess.showinfo("No Defaulters", "Great news! All registered students currently have >= 75% attendance.")

def open_web_portal_action():
    """Open Flask Web & Mobile Dashboard in browser."""
    webbrowser.open("http://127.0.0.1:5000")

def open_email_config_dialog():
    """Open configuration dialog for Gmail SMTP & HOD settings."""
    cfg = get_email_config()
    top = tk.Toplevel(window)
    top.title("📧 Email & HOD SMTP Settings")
    top.geometry("450x340")
    top.configure(bg="#1a1918")
    top.grab_set()

    tk.Label(top, text="📧 Email & HOD Configuration", font=('Helvetica', 12, 'bold'), bg="#1a1918", fg="#00aeff").pack(pady=12)

    f = tk.Frame(top, bg="#1a1918")
    f.pack(padx=20, pady=10, fill='x')

    tk.Label(f, text="Sender Gmail:", font=('Helvetica', 9, 'bold'), bg="#1a1918", fg="white").grid(row=0, column=0, sticky='w', pady=6)
    e_sender = tk.Entry(f, width=32)
    e_sender.insert(0, cfg["SENDER_EMAIL"])
    e_sender.grid(row=0, column=1, pady=6)

    tk.Label(f, text="Gmail App Password:", font=('Helvetica', 9, 'bold'), bg="#1a1918", fg="white").grid(row=1, column=0, sticky='w', pady=6)
    e_pass = tk.Entry(f, width=32, show="*")
    e_pass.insert(0, cfg["SENDER_PASSWORD"])
    e_pass.grid(row=1, column=1, pady=6)

    tk.Label(f, text="HOD Email:", font=('Helvetica', 9, 'bold'), bg="#1a1918", fg="white").grid(row=2, column=0, sticky='w', pady=6)
    e_hod = tk.Entry(f, width=32)
    e_hod.insert(0, cfg["HOD_EMAIL"])
    e_hod.grid(row=2, column=1, pady=6)

    def save_cfg():
        s = e_sender.get().strip()
        p = e_pass.get().strip()
        h = e_hod.get().strip()
        update_email_config(s, p, h)
        mess.showinfo("Saved", "Email configuration saved successfully!")
        top.destroy()

    tk.Button(top, text="💾 Save Configuration", command=save_cfg, bg="#00aeff", fg="black", font=('Helvetica', 10, 'bold'), relief='flat', padx=12, pady=6).pack(pady=15)
    tk.Label(top, text="Note: Use a 16-character Gmail App Password (myaccount.google.com/apppasswords)", font=('Helvetica', 8, 'italic'), bg="#1a1918", fg="#aaaaaa").pack()

def refresh_analytics_dashboard():
    """Master refresh — updates all 5 analytics panels."""
    refresh_analytics_overview()
    refresh_weekly_panel()
    refresh_monthly_panel()
    refresh_subject_panel()
    refresh_classwise_panel()

############################################# GUI FRONTEND LAYOUT ################################################

window = tk.Tk()
window.geometry("1280x760")
window.resizable(True, True)
window.title("Smart Classroom Biometric Face Recognition & Analytics Platform")
window.configure(background='#262523')

# Apply TTK Theme Styles
style = ttk.Style()
style.theme_use('clam')

# Notebook tabs
style.configure("TNotebook", background='#1a1918', borderwidth=0)
style.configure("TNotebook.Tab", background='#2d2c2a', foreground='#aaaaaa',
                font=('Helvetica', 11, 'bold'), padding=[18, 9])
style.map("TNotebook.Tab",
          background=[("selected", "#00aeff"), ("active", "#3a3935")],
          foreground=[("selected", "#000000"), ("active", "white")])

# Base Treeview
style.configure("Treeview", font=('Helvetica', 11), rowheight=28,
                background="#1e1e1e", foreground="#f0f0f0", fieldbackground="#1e1e1e")
style.configure("Treeview.Heading", font=('Helvetica', 11, 'bold'),
                background="#00aeff", foreground="#000000")
style.map("Treeview", background=[("selected", "#00aeff")], foreground=[("selected", "black")])

# Combobox
style.configure("TCombobox", fieldbackground="#2d2c2a", background="#2d2c2a",
                foreground="white", selectbackground="#00aeff")

# Scrollbar
style.configure("Vertical.TScrollbar", background="#3a3935", troughcolor="#1e1e1e",
                arrowcolor="#00aeff")

# Header Frame
header_frame = tk.Frame(window, bg="#1a1918", height=70)
header_frame.pack(side='top', fill='x')

title_lbl = tk.Label(header_frame, text="🎓 Smart Classroom Attendance & Analytics System", font=('times', 22, 'bold'), fg="#00aeff", bg="#1a1918")
title_lbl.pack(side='left', padx=20, pady=10)

clock_lbl = tk.Label(header_frame, font=('times', 16, 'bold'), fg="orange", bg="#1a1918")
clock_lbl.pack(side='right', padx=20)

def update_clock():
    time_str = time.strftime('%d-%m-%Y  |  %H:%M:%S')
    clock_lbl.config(text=time_str)
    clock_lbl.after(500, update_clock)

update_clock()

# Tabbed Layout Notebook
notebook = ttk.Notebook(window)
notebook.pack(fill='both', expand=True, padx=10, pady=10)

# Tab 1: Live Attendance Camera Session
tab_live = tk.Frame(notebook, bg="#262523")
notebook.add(tab_live, text=" 🎥 Live Attendance Session ")

# Tab 2: Student Registration & Management
tab_students = tk.Frame(notebook, bg="#262523")
notebook.add(tab_students, text=" 👨‍🎓 Student Management ")

# Tab 3: Subjects & Classes
tab_subjects = tk.Frame(notebook, bg="#262523")
notebook.add(tab_subjects, text=" 📚 Subjects & Classes ")

# Tab 4: Analytics Dashboard
tab_analytics = tk.Frame(notebook, bg="#262523")
notebook.add(tab_analytics, text=" 📊 Analytical Dashboard ")

############################################# TAB 1: LIVE ATTENDANCE ################################################

frame_live_left = tk.Frame(tab_live, bg="#00aeff", width=420)
frame_live_left.pack(side='left', fill='both', expand=False, padx=10, pady=10)

frame_live_right = tk.Frame(tab_live, bg="#3b3a36")
frame_live_right.pack(side='right', fill='both', expand=True, padx=10, pady=10)

# Controls Panel
tk.Label(frame_live_left, text="Class Attendance Controls", font=('times', 16, 'bold'), bg="#00aeff", fg="black").pack(pady=15)

tk.Label(frame_live_left, text="Select Subject / Class:", font=('times', 12, 'bold'), bg="#00aeff", fg="black").pack(anchor='w', padx=25, pady=5)
subject_combo = ttk.Combobox(frame_live_left, font=('times', 12, 'bold'), state="readonly")
subject_combo.pack(fill='x', padx=25, pady=5)

btn_start_cam = tk.Button(frame_live_left, text="📷 Start CCTV / Mobile Camera Session", command=TrackImages, bg="yellow", fg="black", font=('times', 14, 'bold'), height=2)
btn_start_cam.pack(fill='x', padx=25, pady=20)

btn_share_wa = tk.Button(frame_live_left, text="📲 Share Attendance Report via WhatsApp", command=share_whatsapp_attendance, bg="#25D366", fg="white", font=('times', 12, 'bold'), height=2)
btn_share_wa.pack(fill='x', padx=25, pady=10)

btn_open_folder = tk.Button(frame_live_left, text="📁 Open Attendance CSV Files Folder", command=open_csv_folder, bg="#3ece48", fg="black", font=('times', 12, 'bold'), height=2)
btn_open_folder.pack(fill='x', padx=25, pady=10)

# Live Table Panel
tk.Label(frame_live_right, text="Current Session Attendance Records", font=('times', 16, 'bold'), bg="#3b3a36", fg="white").pack(pady=10)

live_tree = ttk.Treeview(frame_live_right, columns=('id', 'name', 'subject', 'date', 'time', 'status'), show='headings')
live_tree.heading('id', text='Student ID')
live_tree.heading('name', text='Name')
live_tree.heading('subject', text='Subject')
live_tree.heading('date', text='Date')
live_tree.heading('time', text='Time')
live_tree.heading('status', text='Status')

live_tree.column('id', width=110)
live_tree.column('name', width=160)
live_tree.column('subject', width=140)
live_tree.column('date', width=110)
live_tree.column('time', width=110)
live_tree.column('status', width=90)

live_scroll = ttk.Scrollbar(frame_live_right, orient='vertical', command=live_tree.yview)
live_tree.configure(yscrollcommand=live_scroll.set)
live_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
live_scroll.pack(side='right', fill='y', pady=10, padx=(0, 10))

def update_live_attendance_table(df_session):
    for item in live_tree.get_children():
        live_tree.delete(item)
    for idx, row in df_session.iterrows():
        live_tree.insert('', 'end', values=(row.get('ID', ''), row.get('NAME', ''), row.get('CLASS_SUBJECT', ''), row.get('DATE', ''), row.get('TIME', ''), row.get('STATUS', 'Present')))

############################################# TAB 2: STUDENT MANAGEMENT ################################################

frame_reg_left = tk.Frame(tab_students, bg="#00aeff", width=420)
frame_reg_left.pack(side='left', fill='both', expand=False, padx=10, pady=10)

frame_reg_right = tk.Frame(tab_students, bg="#3b3a36")
frame_reg_right.pack(side='right', fill='both', expand=True, padx=10, pady=10)

# Registration Form
tk.Label(frame_reg_left, text="Register New Student", font=('times', 16, 'bold'), bg="#00aeff", fg="black").pack(pady=15)

tk.Label(frame_reg_left, text="Student ID (e.g. 1AM24MC015):", font=('times', 11, 'bold'), bg="#00aeff", fg="black").pack(anchor='w', padx=25, pady=3)
txt_id = tk.Entry(frame_reg_left, font=('times', 12))
txt_id.pack(fill='x', padx=25, pady=3)

tk.Label(frame_reg_left, text="Student Full Name:", font=('times', 11, 'bold'), bg="#00aeff", fg="black").pack(anchor='w', padx=25, pady=3)
txt_name = tk.Entry(frame_reg_left, font=('times', 12))
txt_name.pack(fill='x', padx=25, pady=3)

tk.Label(frame_reg_left, text="Class / Section:", font=('times', 11, 'bold'), bg="#00aeff", fg="black").pack(anchor='w', padx=25, pady=3)
txt_class = tk.Entry(frame_reg_left, font=('times', 12))
txt_class.insert(0, "MCA Section A")
txt_class.pack(fill='x', padx=25, pady=3)

lbl_reg_msg = tk.Label(frame_reg_left, text="Fill details & click Take Images", font=('times', 11, 'bold'), bg="#00aeff", fg="black")
lbl_reg_msg.pack(pady=10)

btn_take_img = tk.Button(frame_reg_left, text="📸 Take Face Images & Register", 
                         command=lambda: take_student_images(txt_id, txt_name, txt_class, lbl_reg_msg), 
                         bg="blue", fg="white", font=('times', 13, 'bold'), height=2)
btn_take_img.pack(fill='x', padx=25, pady=10)

lbl_total_registered = tk.Label(frame_reg_left, text="Total Registrations: 0", font=('times', 13, 'bold'), bg="#00aeff", fg="black")
lbl_total_registered.pack(pady=15)

# Registered Students Table & Edit/Delete Buttons
table_bar = tk.Frame(frame_reg_right, bg="#3b3a36")
table_bar.pack(fill='x', padx=10, pady=10)

tk.Label(table_bar, text="Registered Students Database", font=('times', 16, 'bold'), bg="#3b3a36", fg="white").pack(side='left')

btn_edit = tk.Button(table_bar, text="✏️ Edit Student", command=edit_student_action, bg="orange", fg="black", font=('times', 11, 'bold'))
btn_edit.pack(side='right', padx=5)

btn_delete = tk.Button(table_bar, text="🗑️ Delete Student", command=delete_student_action, bg="#ea2a2a", fg="white", font=('times', 11, 'bold'))
btn_delete.pack(side='right', padx=5)

student_tree = ttk.Treeview(frame_reg_right, columns=('serial', 'id', 'name', 'class', 'date'), show='headings')
student_tree.heading('serial', text='S.No')
student_tree.heading('id', text='Student ID')
student_tree.heading('name', text='Name')
student_tree.heading('class', text='Class/Subject')
student_tree.heading('date', text='Reg. Date')

student_tree.column('serial', width=50)
student_tree.column('id', width=130)
student_tree.column('name', width=180)
student_tree.column('class', width=140)
student_tree.column('date', width=120)

student_scroll = ttk.Scrollbar(frame_reg_right, orient='vertical', command=student_tree.yview)
student_tree.configure(yscrollcommand=student_scroll.set)
student_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=(0, 10))
student_scroll.pack(side='right', fill='y', pady=(0, 10), padx=(0, 10))

def refresh_student_table():
    for item in student_tree.get_children():
        student_tree.delete(item)
    df = get_student_df()
    if not df.empty:
        for idx, row in df.iterrows():
            student_tree.insert('', 'end', values=(row.get('SERIAL NO.', idx+1), row.get('ID', ''), row.get('NAME', ''), row.get('CLASS_SUBJECT', ''), row.get('REGISTRATION_DATE', '')))

def refresh_registration_count():
    df = get_student_df()
    lbl_total_registered.config(text=f"Total Registrations: {len(df)}")

############################################# TAB 3: SUBJECTS & CLASSES ################################################

# LEFT PANEL - Add Subject Form
frame_subj_left = tk.Frame(tab_subjects, bg="#1a1918", width=400)
frame_subj_left.pack(side='left', fill='both', expand=False, padx=(10, 5), pady=10)
frame_subj_left.pack_propagate(False)

# Left header
tk.Label(frame_subj_left, text="📚 Subject Management",
         font=('Helvetica', 15, 'bold'), bg="#1a1918", fg="#00aeff").pack(pady=(20, 5))
tk.Label(frame_subj_left, text="Add, Edit or Delete Course Subjects",
         font=('Helvetica', 9), bg="#1a1918", fg="#888888").pack(pady=(0, 15))

# Form card
card_form = tk.Frame(frame_subj_left, bg="#2b2b2b", bd=0)
card_form.pack(fill='x', padx=20, pady=10)

tk.Label(card_form, text="Subject / Course Name",
         font=('Helvetica', 10, 'bold'), bg="#2b2b2b", fg="#aaaaaa").pack(anchor='w', padx=15, pady=(15, 2))
entry_subj = tk.Entry(card_form, font=('Helvetica', 12), bg="#3a3a3a", fg="white",
                      insertbackground='white', relief='flat', bd=5)
entry_subj.pack(fill='x', padx=15, pady=(0, 10))

tk.Label(card_form, text="Instructor / Teacher Name",
         font=('Helvetica', 10, 'bold'), bg="#2b2b2b", fg="#aaaaaa").pack(anchor='w', padx=15, pady=(5, 2))
entry_teacher = tk.Entry(card_form, font=('Helvetica', 12), bg="#3a3a3a", fg="white",
                         insertbackground='white', relief='flat', bd=5)
entry_teacher.pack(fill='x', padx=15, pady=(0, 15))

btn_add_subj = tk.Button(frame_subj_left, text="➕  Add Subject",
                         command=add_subject_action,
                         bg="#00aeff", fg="black", font=('Helvetica', 12, 'bold'),
                         relief='flat', height=2, cursor='hand2')
btn_add_subj.pack(fill='x', padx=20, pady=(5, 5))

# Quick info tip
tk.Label(frame_subj_left,
         text="💡 Tip: Select a row in the table\nto Edit or Delete a subject.",
         font=('Helvetica', 9), bg="#1a1918", fg="#666666", justify='left').pack(padx=20, pady=15, anchor='w')

# RIGHT PANEL - Subject Catalog Table
frame_subj_right = tk.Frame(tab_subjects, bg="#262523")
frame_subj_right.pack(side='right', fill='both', expand=True, padx=(5, 10), pady=10)

# Top bar with title + action buttons
subj_bar = tk.Frame(frame_subj_right, bg="#1a1918", pady=8)
subj_bar.pack(fill='x', padx=10)

tk.Label(subj_bar, text="📋  Active Subject Catalog",
         font=('Helvetica', 14, 'bold'), bg="#1a1918", fg="white").pack(side='left', padx=10)

btn_del_subj = tk.Button(subj_bar, text="🗑️  Delete Subject",
                         command=delete_subject_action,
                         bg="#ea2a2a", fg="white", font=('Helvetica', 10, 'bold'),
                         relief='flat', padx=10, cursor='hand2')
btn_del_subj.pack(side='right', padx=5)

btn_edit_subj = tk.Button(subj_bar, text="✏️  Edit Subject",
                          command=edit_subject_action,
                          bg="#f5a623", fg="black", font=('Helvetica', 10, 'bold'),
                          relief='flat', padx=10, cursor='hand2')
btn_edit_subj.pack(side='right', padx=5)

# Treeview table
subjects_tree = ttk.Treeview(frame_subj_right,
                              columns=('sno', 'name', 'teacher'), show='headings')
subjects_tree.heading('sno', text='S.No')
subjects_tree.heading('name', text='Subject / Course Name')
subjects_tree.heading('teacher', text='Instructor')

subjects_tree.column('sno', width=55, anchor='center')
subjects_tree.column('name', width=320)
subjects_tree.column('teacher', width=260)

# Alternating row colors via tags
subjects_tree.tag_configure('even', background='#1e1e1e', foreground='#f0f0f0')
subjects_tree.tag_configure('odd', background='#252525', foreground='#f0f0f0')

subj_scroll = ttk.Scrollbar(frame_subj_right, orient='vertical', command=subjects_tree.yview)
subjects_tree.configure(yscrollcommand=subj_scroll.set)
subjects_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
subj_scroll.pack(side='right', fill='y', pady=10, padx=(0, 10))

############################################# TAB 4: ANALYTICAL DASHBOARD ################################################

# ---- KPI Cards Row ----
top_metrics_frame = tk.Frame(tab_analytics, bg="#1a1918")
top_metrics_frame.pack(fill='x', padx=10, pady=(10, 4))

def _make_kpi_card(parent, label, bg_color, side='left'):
    f = tk.Frame(parent, bg=bg_color, width=170, height=80)
    f.pack(side=side, padx=8, pady=6)
    f.pack_propagate(False)
    tk.Label(f, text=label, font=('Helvetica', 9, 'bold'), bg=bg_color, fg='black').pack(pady=(8,0))
    val_lbl = tk.Label(f, text='--', font=('Helvetica', 24, 'bold'), bg=bg_color, fg='black')
    val_lbl.pack()
    return val_lbl

card_total_students = _make_kpi_card(top_metrics_frame, '👥 Total Students', '#00aeff')
card_total_sessions = _make_kpi_card(top_metrics_frame, '🎓 Total Sessions', '#3ece48')
card_total_subjects = _make_kpi_card(top_metrics_frame, '📚 Subjects Tracked', '#f5a623')
card_avg_att       = _make_kpi_card(top_metrics_frame, '📊 Avg. Attendance %', '#9b59b6')
card_defaulters    = _make_kpi_card(top_metrics_frame, '⚠️ Defaulters (<75%)', '#ea2a2a')

btn_refresh_analytics = tk.Button(top_metrics_frame, text='🔄 Refresh',
    command=refresh_analytics_dashboard, bg='#2d2c2a', fg='#00aeff',
    font=('Helvetica', 9, 'bold'), relief='flat', cursor='hand2', padx=8)
btn_refresh_analytics.pack(side='right', padx=4, pady=20)

btn_pdf_report = tk.Button(top_metrics_frame, text='📄 Export PDF Report',
    command=export_pdf_report_action, bg='#00aeff', fg='black',
    font=('Helvetica', 9, 'bold'), relief='flat', cursor='hand2', padx=8)
btn_pdf_report.pack(side='right', padx=4, pady=20)

btn_email_alert = tk.Button(top_metrics_frame, text='📧 Email Alert Defaulters',
    command=send_email_defaulters_action, bg='#e71d36', fg='white',
    font=('Helvetica', 9, 'bold'), relief='flat', cursor='hand2', padx=8)
btn_email_alert.pack(side='right', padx=4, pady=20)

btn_web_portal = tk.Button(top_metrics_frame, text='🌐 Open Web Portal',
    command=open_web_portal_action, bg='#9b59b6', fg='white',
    font=('Helvetica', 9, 'bold'), relief='flat', cursor='hand2', padx=8)
btn_web_portal.pack(side='right', padx=4, pady=20)

btn_email_cfg = tk.Button(top_metrics_frame, text='⚙️ Email Config',
    command=open_email_config_dialog, bg='#444444', fg='white',
    font=('Helvetica', 9, 'bold'), relief='flat', cursor='hand2', padx=8)
btn_email_cfg.pack(side='right', padx=4, pady=20)

# ---- Inner Sub-Notebook (5 analytical panels) ----
analytics_notebook = ttk.Notebook(tab_analytics)
analytics_notebook.pack(fill='both', expand=True, padx=10, pady=4)

# Sub-tab 1: Overview
subtab_overview = tk.Frame(analytics_notebook, bg='#1a1918')
analytics_notebook.add(subtab_overview, text='  📈 Overview  ')

# Sub-tab 2: Weekly
subtab_weekly = tk.Frame(analytics_notebook, bg='#1a1918')
analytics_notebook.add(subtab_weekly, text='  📅 Weekly Trend  ')

# Sub-tab 3: Monthly
subtab_monthly = tk.Frame(analytics_notebook, bg='#1a1918')
analytics_notebook.add(subtab_monthly, text='  🗓️ Monthly Trend  ')

# Sub-tab 4: Subject Drill-Down
subtab_subject = tk.Frame(analytics_notebook, bg='#1a1918')
analytics_notebook.add(subtab_subject, text='  📚 Subject Drill-Down  ')

# Sub-tab 5: Class/Period-wise
subtab_classwise = tk.Frame(analytics_notebook, bg='#1a1918')
analytics_notebook.add(subtab_classwise, text='  🕐 Class/Period-wise  ')

# -------- Overview: chart + student tracker table --------
overview_chart_frame = tk.Frame(subtab_overview, bg='#1a1918', height=300)
overview_chart_frame.pack(fill='x', expand=False, padx=5, pady=5)
overview_chart_frame.pack_propagate(False)

tracker_lbl_frame = tk.Frame(subtab_overview, bg='#1a1918')
tracker_lbl_frame.pack(fill='x', padx=10, pady=(2,0))
tk.Label(tracker_lbl_frame, text='Student Attendance Tracker & Defaulter Warnings',
         font=('Helvetica', 11, 'bold'), bg='#1a1918', fg='white').pack(side='left')

tracker_frame = tk.Frame(subtab_overview, bg='#1a1918')
tracker_frame.pack(fill='both', expand=True, padx=10, pady=4)

student_stats_tree = ttk.Treeview(tracker_frame,
    columns=('id','name','class','present','pct','status'), show='headings', height=6)
for col, hdr, w in [('id','Student ID',120),('name','Name',160),('class','Class',100),
                     ('present','Present/Total',120),('pct','Att %',80),('status','Status',180)]:
    student_stats_tree.heading(col, text=hdr)
    student_stats_tree.column(col, width=w)
stats_scroll = ttk.Scrollbar(tracker_frame, orient='vertical', command=student_stats_tree.yview)
student_stats_tree.configure(yscrollcommand=stats_scroll.set)
student_stats_tree.pack(side='left', fill='both', expand=True)
stats_scroll.pack(side='right', fill='y')

# -------- Weekly panel --------
weekly_chart_frame = tk.Frame(subtab_weekly, bg='#1a1918')
weekly_chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

# -------- Monthly panel --------
monthly_chart_frame = tk.Frame(subtab_monthly, bg='#1a1918')
monthly_chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

# -------- Subject Drill-Down panel --------
subj_drill_top  = tk.Frame(subtab_subject, bg='#1a1918')
subj_drill_top.pack(fill='both', expand=False, padx=10, pady=(10,4))
tk.Label(subj_drill_top, text='📚 Click any subject row below to view its session CSV files',
         font=('Helvetica', 10, 'italic'), bg='#1a1918', fg='#aaaaaa').pack(anchor='w')

subject_drill_tree = ttk.Treeview(subj_drill_top,
    columns=('subject','sessions','present_stu','capacity','rate'), show='headings', height=6)
for col, hdr, w in [('subject','Subject Name',220),('sessions','Sessions',80),
                     ('present_stu','Unique Students',110),('capacity','Capacity',80),('rate','Attendance %',100)]:
    subject_drill_tree.heading(col, text=hdr)
    subject_drill_tree.column(col, width=w)
subject_drill_tree.pack(fill='x', expand=False)
subject_drill_tree.bind('<<TreeviewSelect>>', on_subject_drill_click)

session_lbl = tk.Label(subtab_subject, text='Select a subject above to see CSV sessions →',
    font=('Helvetica', 10, 'italic'), bg='#1a1918', fg='#aaaaaa')
session_lbl.pack(anchor='w', padx=10, pady=(6, 2))

session_files_frame = tk.Frame(subtab_subject, bg='#1a1918')
session_files_frame.pack(fill='both', expand=True, padx=10, pady=(0,10))

session_files_tree = ttk.Treeview(session_files_frame,
    columns=('sno','session_num','datetime','records','pct','filename','path'), show='headings')
for col, hdr, w in [
    ('sno','#',35),
    ('session_num','Session Slot',120),
    ('datetime','Date & Time',150),
    ('records','Present / Total',120),
    ('pct','Att %',80),
    ('filename','Session CSV File',280),
    ('path','Full Path',280)
]:
    session_files_tree.heading(col, text=hdr)
    session_files_tree.column(col, width=w)
sf_scroll = ttk.Scrollbar(session_files_frame, orient='vertical', command=session_files_tree.yview)
session_files_tree.configure(yscrollcommand=sf_scroll.set)
session_files_tree.pack(side='left', fill='both', expand=True)
sf_scroll.pack(side='right', fill='y')
session_files_tree.bind('<Double-Button-1>', open_session_csv)
tk.Label(subtab_subject, text='💡 Double-click a CSV row to open the file',
         font=('Helvetica', 9), bg='#1a1918', fg='#555555').pack(anchor='w', padx=10)

# -------- Class/Period-wise panel --------
classwise_chart_frame = tk.Frame(subtab_classwise, bg='#1a1918')
classwise_chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Auto-refresh on tab switch
def on_analytics_tab_changed(event):
    idx = analytics_notebook.index(analytics_notebook.select())
    if idx == 0: refresh_analytics_overview()
    elif idx == 1: refresh_weekly_panel()
    elif idx == 2: refresh_monthly_panel()
    elif idx == 3: refresh_subject_panel()
    elif idx == 4: refresh_classwise_panel()

analytics_notebook.bind('<<NotebookTabChanged>>', on_analytics_tab_changed)

############################################# INITIALIZATION TRIGGERS ################################################

refresh_student_table()
refresh_registration_count()
refresh_subjects_list()
refresh_analytics_dashboard()

# Launch Flask Web Server in Background
try:
    run_flask_in_background(host="127.0.0.1", port=5000)
except Exception as ex:
    print("Flask background start error:", ex)

window.mainloop()
