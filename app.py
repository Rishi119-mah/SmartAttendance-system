from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import cv2
import base64
from datetime import datetime, timedelta

import database
from face_recognition_module import FaceRecognizer
from nlg_engine import AttendanceNLG
from ml_analytics import AttendanceML

app = Flask(__name__)
CORS(app)

# Initialize modules
face_recognizer = FaceRecognizer()
nlg_engine = AttendanceNLG()
ml_engine = AttendanceML()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/register/start', methods=['POST'])
def register_start():
    name = request.json.get('name')
    if not name:
        return jsonify({"success": False, "message": "Name is required"}), 400
        
    # Clear old samples
    for f in os.listdir(face_recognizer.dataset_dir):
        if f.startswith(f"{name}_"):
            try:
                os.remove(os.path.join(face_recognizer.dataset_dir, f))
            except:
                pass
    return jsonify({"success": True})

@app.route('/api/register/frame', methods=['POST'])
def register_frame():
    data = request.json
    name = data.get('name')
    count = data.get('count')
    b64_img = data.get('image')
    
    if not name or b64_img is None:
        return jsonify({"success": False}), 400
        
    if ',' in b64_img:
        b64_img = b64_img.split(',')[1]
    img_data = base64.b64decode(b64_img)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    success = face_recognizer.process_registration_frame(name, count, frame)
    return jsonify({"success": success})

@app.route('/api/register/train', methods=['POST'])
def register_train():
    name = request.json.get('name')
    success = face_recognizer.train_model()
    if not success:
        return jsonify({"success": False, "message": "Model training failed"}), 500
    
    if name:
        database.insert_student(name)
        
    return jsonify({"success": True})

@app.route('/api/capture-frame', methods=['POST'])
def capture_frame():
    data = request.json
    b64_img = data.get('image')
    if not b64_img:
        return jsonify({"error": "No image provided"}), 400
        
    try:
        # Decode base64
        if ',' in b64_img:
            b64_img = b64_img.split(',')[1]
        img_data = base64.b64decode(b64_img)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces = face_recognizer.recognize_frame(frame)
        return jsonify({"faces": faces})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mark-attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    name = data.get('student_name')
    confidence = data.get('confidence')
    
    if not name or name == "Unknown":
        return jsonify({"success": False, "message": "Invalid student name"}), 400
        
    if confidence is None or confidence < 50:
        return jsonify({"success": False, "message": "Confidence too low to mark attendance"}), 400
        
    student_id = database.get_student_by_name(name)
    if not student_id:
        return jsonify({"success": False, "message": "Student not found in database"}), 404
        
    success, msg = database.insert_attendance(student_id, name, confidence)
    return jsonify({"success": success, "message": msg, "already_marked": not success})

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    students = database.get_all_students()
    total_students = len(students)
    
    today = datetime.now().strftime("%Y-%m-%d")
    att = database.get_attendance(date_filter=today)
    today_count = len(att)
    
    df = database.get_all_attendance_for_analytics()
    overall_rate = 0
    if not df.empty and total_students > 0:
        total_days = max(1, df['date'].nunique())
        total_presents = len(df)
        overall_rate = round((total_presents / (total_students * total_days)) * 100, 2)
        
    # At risk count
    at_risk_count = 0
    if not df.empty:
        for st in students:
            feat = ml_engine._get_student_features(st['name'], df)
            if feat['attendance_rate'] < 0.75:
                at_risk_count += 1
                
    # Trend last 14 days
    trend_last_14 = []
    weekly_bar = []
    if not df.empty:
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts = daily_counts.sort_values('date')
        
        last_14 = daily_counts.tail(14)
        for _, row in last_14.iterrows():
            trend_last_14.append({"date": row['date'].strftime("%Y-%m-%d"), "count": int(row['count'])})
            
        last_7 = daily_counts.tail(7)
        for _, row in last_7.iterrows():
            day_name = row['date'].strftime("%a")
            weekly_bar.append({"day": day_name, "count": int(row['count'])})
            
    recent_activity = database.get_attendance()[:10]
    
    return jsonify({
        "total_students": total_students,
        "today_count": today_count,
        "overall_rate": overall_rate,
        "at_risk_count": at_risk_count,
        "trend_last_14_days": trend_last_14,
        "weekly_bar": weekly_bar,
        "recent_activity": recent_activity
    })

@app.route('/api/students', methods=['GET'])
def get_students():
    return jsonify(database.get_all_students())

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    date_filter = request.args.get('date')
    name_filter = request.args.get('name')
    return jsonify(database.get_attendance(date_filter, name_filter))

@app.route('/api/analytics/', methods=['GET'])
def get_analytics():
    student_name = request.args.get('name')
    df = database.get_all_attendance_for_analytics()
    
    # Global trend
    trend_res = ml_engine.project_trend(df)
    
    if student_name:
        history = database.get_student_attendance_history(student_name)
        risk = ml_engine.predict_risk(student_name, df)
        return jsonify({
            "history": history,
            "trend_slope": trend_res['slope'],
            "projection": trend_res['projection'],
            "risk_level": risk['risk_level'],
            "risk_confidence": risk['risk_confidence'],
            "feature_importances": risk['feature_importances'],
            "attendance_rate": risk['attendance_rate'],
            "streak_absent": risk['streak_absent']
        })
    return jsonify(trend_res)

@app.route('/api/anomalies', methods=['GET'])
def get_anomalies():
    df = database.get_all_attendance_for_analytics()
    res = ml_engine.detect_anomalies(df)
    return jsonify(res)

@app.route('/api/heatmap', methods=['GET'])
def get_heatmap():
    df = database.get_all_attendance_for_analytics()
    return jsonify(ml_engine.class_heatmap_data(df))

@app.route('/api/generate-report', methods=['GET'])
def generate_report():
    students = database.get_all_students()
    df = database.get_all_attendance_for_analytics()
    
    total_students = len(students)
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = len(database.get_attendance(date_filter=today))
    
    overall_rate = 0
    if not df.empty and total_students > 0:
        total_days = max(1, df['date'].nunique())
        overall_rate = round((len(df) / (total_students * total_days)) * 100, 2)
        
    at_risk_list = []
    top_students = []
    if not df.empty:
        for st in students:
            feat = ml_engine._get_student_features(st['name'], df)
            rate = feat['attendance_rate']
            streak = feat['streak_absent']
            if streak >= 3:
                at_risk_list.append({"name": st['name'], "risk_level": "critical"})
            elif rate < 0.60:
                at_risk_list.append({"name": st['name'], "risk_level": "severe"})
            elif rate < 0.75:
                at_risk_list.append({"name": st['name'], "risk_level": "mild"})
            elif rate > 0.90:
                top_students.append(st['name'])
                
    trend_res = ml_engine.project_trend(df)
    
    day_counts = {}
    if not df.empty:
        df_temp = df.copy()
        df_temp['weekday'] = pd.to_datetime(df_temp['date']).dt.day_name()
        day_counts = df_temp['weekday'].value_counts().to_dict()

    stats = {
        "total_students": total_students,
        "today_count": today_count,
        "overall_rate": overall_rate,
        "at_risk_list": at_risk_list,
        "top_students": top_students[:5],
        "trend_direction": trend_res['trend_direction'],
        "day_counts": day_counts
    }
    
    report_text = nlg_engine.generate_report(stats)
    
    # Sections (splitting simple string for frontend use if needed)
    sections = {
        "summary": "Included in report_text",
        "risks": "Included in report_text",
        "highlights": "Included in report_text",
        "trend": "Included in report_text",
        "actions": "Included in report_text"
    }
    
    # Find similar week
    db_summaries = database.get_all_weekly_summaries()
    similar_week = nlg_engine.find_similar_week(stats, db_summaries)
    
    # Save to db
    week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")
    database.save_weekly_summary(week_start, overall_rate, report_text)
    
    return jsonify({
        "report_text": report_text,
        "similar_week": similar_week,
        "sections": sections
    })

@app.route('/api/generate-letter', methods=['POST'])
def generate_letter():
    name = request.json.get('student_name')
    df = database.get_all_attendance_for_analytics()
    
    feat = ml_engine._get_student_features(name, df)
    
    # calculate missed days
    missed_days = []
    if not df.empty:
        all_dates = sorted(df['date'].unique())
        st_df = df[df['student_name'] == name]
        st_dates = st_df['date'].unique()
        for d in all_dates:
            if d not in st_dates:
                missed_days.append(d)
                
    stats = {
        "name": name,
        "attendance_rate": round(feat['attendance_rate']*100, 2),
        "missed_days": missed_days,
        "streak_absent": feat['streak_absent'],
        "total_days": len(df['date'].unique()) if not df.empty else 0
    }
    
    letter = nlg_engine.generate_warning_letter(stats)
    return jsonify({"letter_text": letter})

@app.route('/api/generate-parent-note', methods=['POST'])
def generate_parent_note():
    name = request.json.get('student_name')
    df = database.get_all_attendance_for_analytics()
    
    feat = ml_engine._get_student_features(name, df)
    
    missed_days = []
    if not df.empty:
        all_dates = sorted(df['date'].unique())
        st_df = df[df['student_name'] == name]
        st_dates = st_df['date'].unique()
        for d in all_dates:
            if d not in st_dates:
                missed_days.append(d)
                
    stats = {
        "name": name,
        "attendance_rate": round(feat['attendance_rate']*100, 2),
        "missed_days": missed_days,
        "total_days": len(df['date'].unique()) if not df.empty else 0
    }
    
    note = nlg_engine.generate_parent_note(stats)
    return jsonify({"note_text": note})

@app.route('/api/export/students')
def export_students():
    df = pd.DataFrame(database.get_all_students())
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=students.csv"}
    )

@app.route('/api/export/attendance')
def export_attendance():
    df = database.get_all_attendance_for_analytics()
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=attendance.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
