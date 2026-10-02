"""
Database functions.
By default the project uses SQLite (no setup needed).
To use MySQL, set the environment variable DB_BACKEND=mysql (see README).
"""
import os
import sqlite3
import datetime
from werkzeug.security import generate_password_hash

USE_MYSQL = os.getenv("DB_BACKEND", "sqlite") == "mysql"

if USE_MYSQL:
    ID_COLUMN = "INT PRIMARY KEY AUTO_INCREMENT"
else:
    ID_COLUMN = "INTEGER PRIMARY KEY AUTOINCREMENT"


def connect():
    if USE_MYSQL:
        import pymysql
        return pymysql.connect(host=os.getenv("DB_HOST", "localhost"),
                               user=os.getenv("DB_USER", "root"),
                               password=os.getenv("DB_PASSWORD", ""),
                               database=os.getenv("DB_NAME", "exam_forensics"),
                               cursorclass=pymysql.cursors.DictCursor)
    con = sqlite3.connect(os.getenv("SQLITE_PATH", "exam_forensics.db"))
    con.row_factory = sqlite3.Row
    return con


def fix(sql):
    """SQLite uses ? and MySQL uses %s in SQL commands."""
    if USE_MYSQL:
        return sql.replace("?", "%s")
    return sql


def query(sql, params=(), one=False):
    """Run a SELECT and return a list of dictionaries (or one dictionary)."""
    con = connect()
    cur = con.cursor()
    cur.execute(fix(sql), params)
    rows = []
    for r in cur.fetchall():
        rows.append(dict(r))
    con.close()
    if one:
        if len(rows) > 0:
            return rows[0]
        return None
    return rows


def init_db():
    """Create the tables and the default admin user."""
    con = connect()
    cur = con.cursor()
    cur.execute(f"CREATE TABLE IF NOT EXISTS admins (id {ID_COLUMN}, username VARCHAR(50) UNIQUE NOT NULL, password_hash VARCHAR(255) NOT NULL)")
    cur.execute(f"CREATE TABLE IF NOT EXISTS runs (id {ID_COLUMN}, filename VARCHAR(255), created_at VARCHAR(30), n_students INT, n_centers INT, n_questions INT, n_high_students INT, n_high_centers INT)")
    cur.execute(f"CREATE TABLE IF NOT EXISTS student_risk (id {ID_COLUMN}, run_id INT, student_id VARCHAR(64), center_id VARCHAR(64), score DOUBLE, center_z DOUBLE, max_similarity DOUBLE, max_same_wrong INT, n_partners INT, risk_score DOUBLE, risk_level VARCHAR(10), reasons TEXT)")
    cur.execute(f"CREATE TABLE IF NOT EXISTS center_risk (id {ID_COLUMN}, run_id INT, center_id VARCHAR(64), n_students INT, avg_score DOUBLE, max_score DOUBLE, score_std DOUBLE, pct_flagged DOUBLE, mean_similarity DOUBLE, risk_score DOUBLE, risk_level VARCHAR(10), reasons TEXT)")
    cur.execute(f"CREATE TABLE IF NOT EXISTS similar_pairs (id {ID_COLUMN}, run_id INT, student_a VARCHAR(64), student_b VARCHAR(64), center_id VARCHAR(64), similarity DOUBLE, same_wrong INT)")

    username = os.getenv("ADMIN_USER", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin123")
    cur.execute(fix("SELECT id FROM admins WHERE username=?"), (username,))
    if cur.fetchone() is None:
        cur.execute(fix("INSERT INTO admins (username, password_hash) VALUES (?, ?)"),
                    (username, generate_password_hash(password)))
    con.commit()
    con.close()


def save_run(filename, result):
    """Save one analysis (students, centres, pairs) and return its run id."""
    students = result["students"]
    centers = result["centers"]
    pairs = result["pairs"]

    con = connect()
    cur = con.cursor()

    n_high_students = int((students["risk_level"] == "High").sum())
    n_high_centers = int((centers["risk_level"] == "High").sum())
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cur.execute(fix("INSERT INTO runs (filename, created_at, n_students, n_centers, n_questions, n_high_students, n_high_centers) VALUES (?,?,?,?,?,?,?)"),
                (filename, now, len(students), len(centers), int(result["n_questions"]), n_high_students, n_high_centers))
    run_id = cur.lastrowid

    for i in range(len(students)):
        r = students.iloc[i]
        cur.execute(fix("INSERT INTO student_risk (run_id, student_id, center_id, score, center_z, max_similarity, max_same_wrong, n_partners, risk_score, risk_level, reasons) VALUES (?,?,?,?,?,?,?,?,?,?,?)"),
                    (run_id, str(r["student_id"]), str(r["center_id"]), float(r["score"]), float(r["center_z"]),
                     float(r["max_similarity"]), int(r["max_same_wrong"]), int(r["n_partners"]),
                     float(r["risk_score"]), str(r["risk_level"]), str(r["reasons"])))

    for i in range(len(centers)):
        r = centers.iloc[i]
        cur.execute(fix("INSERT INTO center_risk (run_id, center_id, n_students, avg_score, max_score, score_std, pct_flagged, mean_similarity, risk_score, risk_level, reasons) VALUES (?,?,?,?,?,?,?,?,?,?,?)"),
                    (run_id, str(r["center_id"]), int(r["n_students"]), float(r["avg_score"]), float(r["max_score"]),
                     float(r["score_std"]), float(r["pct_flagged"]), float(r["mean_similarity"]),
                     float(r["risk_score"]), str(r["risk_level"]), str(r["reasons"])))

    for i in range(len(pairs)):
        r = pairs.iloc[i]
        cur.execute(fix("INSERT INTO similar_pairs (run_id, student_a, student_b, center_id, similarity, same_wrong) VALUES (?,?,?,?,?,?)"),
                    (run_id, str(r["student_a"]), str(r["student_b"]), str(r["center_id"]),
                     float(r["similarity"]), int(r["same_wrong"])))

    con.commit()
    con.close()
    return run_id
