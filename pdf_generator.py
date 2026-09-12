import os
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

REPORTS_DIR = "Reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

MASTER_CSV = os.path.join("Attendance", "Master_Attendance.csv")
STUDENTS_CSV = os.path.join("StudentDetails", "StudentDetails.csv")
SUBJECTS_CSV = os.path.join("Subjects", "Subjects.csv")


def _get_active_subjects():
    if os.path.isfile(SUBJECTS_CSV):
        try:
            df = pd.read_csv(SUBJECTS_CSV)
            return df['SUBJECT_NAME'].dropna().unique().tolist()
        except:
            pass
    return []


def _load_data(subject_filter=None):
    """Load students and master attendance, applying subject filter if provided."""
    students_df = pd.DataFrame()
    if os.path.isfile(STUDENTS_CSV):
        try:
            students_df = pd.read_csv(STUDENTS_CSV)
        except Exception as e:
            print("PDF generator error reading students:", e)

    master_df = pd.DataFrame()
    if os.path.isfile(MASTER_CSV):
        try:
            master_df = pd.read_csv(MASTER_CSV)
            active_subjs = _get_active_subjects()
            if active_subjs and 'CLASS_SUBJECT' in master_df.columns:
                master_df = master_df[master_df['CLASS_SUBJECT'].isin(active_subjs)]

            if subject_filter and subject_filter != "All Subjects" and 'CLASS_SUBJECT' in master_df.columns:
                master_df = master_df[master_df['CLASS_SUBJECT'] == subject_filter]

            if not master_df.empty and 'DATE' in master_df.columns and 'TIME' in master_df.columns:
                master_df['DATETIME'] = pd.to_datetime(
                    master_df['DATE'].astype(str) + ' ' + master_df['TIME'].astype(str),
                    format='%d-%m-%Y %H:%M:%S', errors='coerce'
                )
                master_df = master_df.sort_values('DATETIME')

                session_keys = []
                last_seen = {}
                sess_cnt = {}
                for _, row in master_df.iterrows():
                    subj = str(row.get('CLASS_SUBJECT', ''))
                    dt_val = row['DATETIME']
                    dt_str = dt_val.strftime('%Y-%m-%d') if pd.notnull(dt_val) else str(row.get('DATE', ''))
                    key = (subj, dt_str)
                    if key not in last_seen or pd.isnull(dt_val) or (dt_val - last_seen[key]).total_seconds() > 600:
                        sess_cnt[key] = sess_cnt.get(key, 0) + 1
                        if pd.notnull(dt_val):
                            last_seen[key] = dt_val
                    session_keys.append(f"{subj}|{dt_str}|S{sess_cnt[key]}")
                master_df['SESSION_KEY'] = session_keys
                master_df = master_df.drop_duplicates(subset=['ID', 'SESSION_KEY'])
        except Exception as e:
            print("PDF generator error reading master:", e)

    return students_df, master_df


def generate_attendance_pdf(subject_name=None, college_name="SMART CLASSROOM ACADEMIC PLATFORM"):
    """Generate a clean, professional PDF attendance report."""
    now_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    subj_label = subject_name if (subject_name and subject_name != "All Subjects") else "All_Subjects"
    subj_slug = subj_label.replace(" ", "_").replace("/", "_")
    filename = f"Attendance_Report_{subj_slug}_{now_str}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    students_df, master_df = _load_data(subject_filter=subject_name)

    doc = SimpleDocTemplate(
        filepath, pagesize=letter,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    story = []
    styles = getSampleStyleSheet()

    # Title & Subtitle Styles
    style_college = ParagraphStyle(
        'CollegeTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor('#003366'),
        alignment=1, spaceAfter=4
    )
    style_subtitle = ParagraphStyle(
        'ReportSubtitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#444444'),
        alignment=1, spaceAfter=15
    )
    style_normal = styles['Normal']
    style_bold = ParagraphStyle('BoldText', parent=styles['Normal'], fontName='Helvetica-Bold')

    story.append(Paragraph(college_name.upper(), style_college))
    story.append(Paragraph(f"OFFICIAL ATTENDANCE REPORT — {subj_label.upper()}", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#003366'), spaceAfter=15))

    # General Information Table
    total_sessions = master_df['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
    total_students = len(students_df)
    defaulter_count = 0

    if not students_df.empty and total_sessions > 0:
        for _, srow in students_df.iterrows():
            st_id = str(srow['ID'])
            p_cnt = len(master_df[master_df['ID'].astype(str) == st_id]) if not master_df.empty else 0
            if (p_cnt / total_sessions * 100.0) < 75.0:
                defaulter_count += 1

    info_data = [
        [Paragraph("<b>Report Date:</b>", style_normal), Paragraph(datetime.datetime.now().strftime("%d %B %Y, %I:%M %p"), style_normal),
         Paragraph("<b>Total Sessions Tracked:</b>", style_normal), Paragraph(str(total_sessions), style_normal)],
        [Paragraph("<b>Subject / Course:</b>", style_normal), Paragraph(subj_label, style_normal),
         Paragraph("<b>Registered Students:</b>", style_normal), Paragraph(str(total_students), style_normal)],
        [Paragraph("<b>Defaulter Threshold:</b>", style_normal), Paragraph("<font color='#cc0000'><b>75.0% Minimum</b></font>", style_normal),
         Paragraph("<b>Defaulter Count (<75%):</b>", style_normal), Paragraph(f"<font color='#cc0000'><b>{defaulter_count} Students</b></font>", style_normal)]
    ]
    info_table = Table(info_data, colWidths=[120, 150, 140, 130])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F6F8')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#D0D7DE')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E1E4E8')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))

    # Student Attendance Performance Table
    story.append(Paragraph("<b>STUDENT ATTENDANCE PERFORMANCE BREAKDOWN</b>", style_bold))
    story.append(Spacer(1, 6))

    table_data = [[
        Paragraph("<b>#</b>", style_bold),
        Paragraph("<b>Student ID</b>", style_bold),
        Paragraph("<b>Student Name</b>", style_bold),
        Paragraph("<b>Attended / Total</b>", style_bold),
        Paragraph("<b>Attendance %</b>", style_bold),
        Paragraph("<b>Status / Alert</b>", style_bold)
    ]]

    if not students_df.empty:
        for idx, srow in students_df.iterrows():
            st_id = str(srow['ID'])
            st_name = str(srow['NAME'])
            p_cnt = master_df[master_df['ID'].astype(str) == st_id]['SESSION_KEY'].nunique() if not master_df.empty and 'SESSION_KEY' in master_df.columns else 0
            att_pct = (p_cnt / total_sessions * 100.0) if total_sessions > 0 else 0.0

            if att_pct >= 75.0:
                status_p = Paragraph("<font color='#008800'><b>REGULAR</b></font>", style_normal)
                pct_p = Paragraph(f"<b>{att_pct:.1f}%</b>", style_normal)
            else:
                status_p = Paragraph("<font color='#cc0000'><b>⚠️ DEFAULTER (<75%)</b></font>", style_normal)
                pct_p = Paragraph(f"<font color='#cc0000'><b>{att_pct:.1f}%</b></font>", style_normal)

            table_data.append([
                str(idx + 1),
                st_id,
                st_name,
                f"{p_cnt} / {total_sessions}",
                pct_p,
                status_p
            ])
    else:
        table_data.append(["-", "No Data", "No Students Registered", "0/0", "0.0%", "N/A"])

    att_table = Table(table_data, colWidths=[30, 110, 160, 100, 75, 125])
    att_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))
    story.append(att_table)
    story.append(Spacer(1, 40))

    # Official Signatures Section
    sig_data = [
        [Paragraph("___________________________<br/><b>Subject Teacher Signature</b>", style_normal),
         Paragraph("___________________________<br/><b>Head of Department (HOD)</b>", style_normal),
         Paragraph("___________________________<br/><b>Principal / Academic Dean</b>", style_normal)]
    ]
    sig_table = Table(sig_data, colWidths=[180, 180, 180])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM')
    ]))
    story.append(sig_table)

    doc.build(story)
    return filepath


if __name__ == "__main__":
    pdf_path = generate_attendance_pdf()
    print("Generated PDF:", pdf_path)
