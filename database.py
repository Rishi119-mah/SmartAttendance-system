import sqlite3
import pandas as pd
from datetime import datetime

DB_FILE = 'attendance.db'

def get_connection():
    return sqlite3.connect(DB_FILE)

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                registered_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                student_name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence INTEGER,
                FOREIGN KEY(student_id) REFERENCES Students(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS WeeklySummaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start TEXT NOT NULL,
                avg_attendance_rate REAL NOT NULL,
                summary_text TEXT NOT NULL
            )
        ''')
        conn.commit()

def insert_student(name):
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute("INSERT INTO Students (name, registered_at) VALUES (?, ?)", (name, now))
            conn.commit()
            return True, "Student registered successfully."
        except sqlite3.IntegrityError:
            return False, "Student already exists."

def get_all_students():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, registered_at FROM Students")
        rows = cursor.fetchall()
        return [{"id": r[0], "name": r[1], "registered_at": r[2]} for r in rows]

def get_student_by_name(name):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM Students WHERE name = ?", (name,))
        res = cursor.fetchone()
        return res[0] if res else None

def already_marked_today(student_id, date_str=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM Attendance WHERE student_id = ? AND date = ?", (student_id, date_str))
        return cursor.fetchone() is not None

def insert_attendance(student_id, student_name, confidence):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    
    if already_marked_today(student_id, date_str):
        return False, "Already marked today"
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Attendance (student_id, student_name, date, time, confidence) VALUES (?, ?, ?, ?, ?)",
            (student_id, student_name, date_str, time_str, confidence)
        )
        conn.commit()
        return True, "Attendance marked"

def get_attendance(date_filter=None, name_filter=None):
    query = "SELECT id, student_id, student_name, date, time, confidence FROM Attendance WHERE 1=1"
    params = []
    if date_filter:
        query += " AND date = ?"
        params.append(date_filter)
    if name_filter:
        query += " AND student_name LIKE ?"
        params.append(f"%{name_filter}%")
        
    query += " ORDER BY date DESC, time DESC"
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [{"id": r[0], "student_id": r[1], "student_name": r[2], "date": r[3], "time": r[4], "confidence": r[5]} for r in rows]

def get_student_attendance_history(student_name):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT date, time, confidence FROM Attendance WHERE student_name = ? ORDER BY date ASC", (student_name,))
        rows = cursor.fetchall()
        return [{"date": r[0], "time": r[1], "confidence": r[2]} for r in rows]

def get_all_attendance_for_analytics():
    # Returns pandas DataFrame for ML component
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT student_id, student_name, date, time FROM Attendance", conn)
        return df

def save_weekly_summary(week_start, avg_attendance_rate, summary_text):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO WeeklySummaries (week_start, avg_attendance_rate, summary_text) VALUES (?, ?, ?)",
            (week_start, avg_attendance_rate, summary_text)
        )
        conn.commit()

def get_all_weekly_summaries():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, week_start, avg_attendance_rate, summary_text FROM WeeklySummaries ORDER BY week_start DESC")
        rows = cursor.fetchall()
        return [{"id": r[0], "week_start": r[1], "avg_attendance_rate": r[2], "summary_text": r[3]} for r in rows]

# Initialize DB on import
init_db()
