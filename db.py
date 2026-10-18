"""
Unified Database Access Layer for AI-Based Forensic Answer Sheet Analysis.
Provides zero-configuration SQLite for development/demo and seamless MySQL support.
Tracks full multi-run history, examinee forensics, center risk, pairwise collusion, and raw answer cache.
"""
import os
import sqlite3
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

import config

USE_MYSQL = config.DB_TYPE == "mysql"

if USE_MYSQL:
    ID_COL = "INT PRIMARY KEY AUTO_INCREMENT"
    TEXT_COL = "TEXT"
else:
    ID_COL = "INTEGER PRIMARY KEY AUTOINCREMENT"
    TEXT_COL = "TEXT"


def get_connection():
    """Returns a database connection for SQLite or MySQL."""
    if USE_MYSQL:
        import pymysql
        return pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DB,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
    os.makedirs(os.path.dirname(config.SQLITE_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(config.SQLITE_PATH)
    con.row_factory = sqlite3.Row
    return con


def _fix_sql(sql):
    """Translates parameter placeholders between SQLite (?) and MySQL (%s)."""
    if USE_MYSQL:
        return sql.replace("?", "%s")
    return sql


def query(sql, params=(), one=False):
    """Executes a SELECT query and returns dictionaries."""
    con = get_connection()
    cur = con.cursor()
    cur.execute(_fix_sql(sql), params)
    rows = cur.fetchall()
    con.close()
    res = [dict(r) for r in rows]
    if one:
        return res[0] if res else None
    return res


def execute(sql, params=()):
    """Executes an INSERT/UPDATE/DELETE query and commits."""
    con = get_connection()
    cur = con.cursor()
    cur.execute(_fix_sql(sql), params)
    if not USE_MYSQL:
        con.commit()
    last_id = cur.lastrowid
    con.close()
    return last_id


def init_db():
    """Initializes all database tables and seeds the default admin user."""
    con = get_connection()
    cur = con.cursor()

    # Admins Table
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS admins (
            id {ID_COL},
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL
        )
        """
    )

    # Runs Table (Tracks analysis sessions)
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS runs (
            id {ID_COL},
            filename VARCHAR(255) NOT NULL,
            created_at VARCHAR(30) NOT NULL,
            n_students INT DEFAULT 0,
            n_centers INT DEFAULT 0,
            n_questions INT DEFAULT 0,
            n_high_students INT DEFAULT 0,
            n_high_centers INT DEFAULT 0
        )
        """
    )

    # Student Forensic Analysis Table
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS student_risk (
            id {ID_COL},
            run_id INT NOT NULL,
            student_id VARCHAR(64) NOT NULL,
            center_id VARCHAR(64) NOT NULL,
            score DOUBLE NOT NULL,
            center_z DOUBLE DEFAULT 0,
            overall_z DOUBLE DEFAULT 0,
            max_similarity DOUBLE DEFAULT 0,
            max_same_wrong INT DEFAULT 0,
            n_partners INT DEFAULT 0,
            anomaly_score DOUBLE DEFAULT 0,
            risk_score DOUBLE NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            reasons {TEXT_COL},
            aberrance_score DOUBLE DEFAULT 0
        )
        """
    )

    # Center Forensic Analysis Table
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS center_risk (
            id {ID_COL},
            run_id INT NOT NULL,
            center_id VARCHAR(64) NOT NULL,
            n_students INT NOT NULL,
            avg_score DOUBLE NOT NULL,
            max_score DOUBLE NOT NULL,
            score_std DOUBLE DEFAULT 0,
            pct_flagged DOUBLE DEFAULT 0,
            mean_similarity DOUBLE DEFAULT 0,
            mean_wrong DOUBLE DEFAULT 0,
            risk_score DOUBLE NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            reasons {TEXT_COL}
        )
        """
    )

    # Pairwise Collusion Table
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS similar_pairs (
            id {ID_COL},
            run_id INT NOT NULL,
            student_a VARCHAR(64) NOT NULL,
            student_b VARCHAR(64) NOT NULL,
            center_id VARCHAR(64) NOT NULL,
            similarity DOUBLE NOT NULL,
            same_wrong INT NOT NULL,
            p_value DOUBLE DEFAULT 1.0
        )
        """
    )

    # Raw Answers Cache Table
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS answers_cache (
            id {ID_COL},
            run_id INT NOT NULL,
            student_id VARCHAR(64) NOT NULL,
            answers {TEXT_COL} NOT NULL
        )
        """
    )

    # Column migrations for existing databases
    try:
        cur.execute("ALTER TABLE student_risk ADD COLUMN aberrance_score DOUBLE DEFAULT 0")
    except Exception:
        pass

    try:
        cur.execute("ALTER TABLE similar_pairs ADD COLUMN p_value DOUBLE DEFAULT 1.0")
    except Exception:
        pass

    # Seed Default Admin Account if not present
    cur.execute(_fix_sql("SELECT id FROM admins WHERE username=?"), (config.ADMIN_USER,))
    if cur.fetchone() is None:
        cur.execute(
            _fix_sql("INSERT INTO admins (username, password_hash) VALUES (?, ?)"),
            (config.ADMIN_USER, generate_password_hash(config.ADMIN_PASS)),
        )

    if not USE_MYSQL:
        con.commit()
    con.close()


def verify_user(username, password):
    """Verifies username and password hash."""
    admin = query("SELECT * FROM admins WHERE username=?", (username,), one=True)
    if admin and check_password_hash(admin["password_hash"], password):
        return admin
    if admin and admin["password_hash"] == password:
        return admin
    return None


def save_run(filename, result):
    """
    Saves an entire forensic analysis run atomically.
    Returns the newly created run_id.
    """
    students_df = result["students"]
    centers_df = result["centers"]
    pairs_df = result["pairs"]
    answers_dict = result.get("answers_dict", {})
    n_questions = int(result.get("n_questions", 0))

    n_high_students = int((students_df["risk_level"] == "High").sum())
    n_high_centers = int((centers_df["risk_level"] == "High").sum())
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    con = get_connection()
    cur = con.cursor()

    # 1. Insert Run record
    cur.execute(
        _fix_sql(
            """
            INSERT INTO runs (filename, created_at, n_students, n_centers, n_questions, n_high_students, n_high_centers)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
        ),
        (filename, now, len(students_df), len(centers_df), n_questions, n_high_students, n_high_centers),
    )
    run_id = cur.lastrowid

    # 2. Insert Students
    st_records = []
    for _, r in students_df.iterrows():
        st_records.append(
            (
                run_id,
                str(r["student_id"]),
                str(r["center_id"]),
                float(r["score"]),
                float(r.get("center_z", 0.0)),
                float(r.get("overall_z", 0.0)),
                float(r.get("max_similarity", 0.0)),
                int(r.get("max_same_wrong", 0)),
                int(r.get("n_partners", 0)),
                float(r.get("anomaly_score", 0.0)),
                float(r["risk_score"]),
                str(r["risk_level"]),
                str(r.get("reasons", "")),
                float(r.get("aberrance_score", 0.0)),
            )
        )
    insert_student_sql = _fix_sql(
        """
        INSERT INTO student_risk (run_id, student_id, center_id, score, center_z, overall_z, max_similarity,
                                  max_same_wrong, n_partners, anomaly_score, risk_score, risk_level, reasons, aberrance_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    )
    cur.executemany(insert_student_sql, st_records)

    # 3. Insert Centers
    c_records = []
    for _, r in centers_df.iterrows():
        c_records.append(
            (
                run_id,
                str(r["center_id"]),
                int(r["n_students"]),
                float(r["avg_score"]),
                float(r["max_score"]),
                float(r.get("score_std", 0.0)),
                float(r.get("pct_flagged", 0.0)),
                float(r.get("mean_similarity", 0.0)),
                float(r.get("mean_wrong", 0.0)),
                float(r["risk_score"]),
                str(r["risk_level"]),
                str(r.get("reasons", "")),
            )
        )
    insert_center_sql = _fix_sql(
        """
        INSERT INTO center_risk (run_id, center_id, n_students, avg_score, max_score, score_std,
                                 pct_flagged, mean_similarity, mean_wrong, risk_score, risk_level, reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    )
    cur.executemany(insert_center_sql, c_records)

    # 4. Insert Similar Pairs
    if not pairs_df.empty:
        p_records = []
        for _, r in pairs_df.iterrows():
            p_records.append(
                (
                    run_id,
                    str(r["student_a"]),
                    str(r["student_b"]),
                    str(r["center_id"]),
                    float(r["similarity"]),
                    int(r["same_wrong"]),
                    float(r.get("p_value", 1.0)),
                )
            )
        insert_pair_sql = _fix_sql(
            """
            INSERT INTO similar_pairs (run_id, student_a, student_b, center_id, similarity, same_wrong, p_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
        )
        cur.executemany(insert_pair_sql, p_records)

    # 5. Insert Raw Answers Cache
    if answers_dict:
        ans_records = [(run_id, sid, ans) for sid, ans in answers_dict.items()]
        insert_ans_sql = _fix_sql(
            """
            INSERT INTO answers_cache (run_id, student_id, answers)
            VALUES (?, ?, ?)
            """
        )
        cur.executemany(insert_ans_sql, ans_records)

    if not USE_MYSQL:
        con.commit()
    con.close()
    return run_id


def get_latest_run():
    """Returns the most recent analysis run, or None."""
    return query("SELECT * FROM runs ORDER BY id DESC LIMIT 1", one=True)


def get_run(run_id):
    """Retrieves a specific run by ID."""
    return query("SELECT * FROM runs WHERE id=?", (run_id,), one=True)


def get_all_runs():
    """Returns all analysis runs ordered newest first."""
    return query("SELECT * FROM runs ORDER BY id DESC")


def get_students(run_id, risk_level=None, center_id=None, search=None, limit=None):
    """Retrieves students for a given run with optional filtering."""
    sql = "SELECT * FROM student_risk WHERE run_id=?"
    params = [run_id]

    if risk_level:
        sql += " AND risk_level=?"
        params.append(risk_level)
    if center_id:
        sql += " AND center_id=?"
        params.append(center_id)
    if search:
        sql += " AND (student_id LIKE ? OR center_id LIKE ?)"
        wild = f"%{search}%"
        params.extend([wild, wild])

    sql += " ORDER BY risk_score DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"

    return query(sql, tuple(params))


def get_student(run_id, student_id):
    """Retrieves a single student's forensic record."""
    return query("SELECT * FROM student_risk WHERE run_id=? AND student_id=?", (run_id, student_id), one=True)


def get_centers(run_id, risk_level=None, limit=None):
    """Retrieves centers for a given run."""
    sql = "SELECT * FROM center_risk WHERE run_id=?"
    params = [run_id]
    if risk_level:
        sql += " AND risk_level=?"
        params.append(risk_level)
    sql += " ORDER BY risk_score DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return query(sql, tuple(params))


def get_center(run_id, center_id):
    """Retrieves a single center's forensic record."""
    return query("SELECT * FROM center_risk WHERE run_id=? AND center_id=?", (run_id, center_id), one=True)


def get_pairs(run_id, student_id=None, center_id=None, limit=None):
    """Retrieves collusion pairs for a run."""
    sql = "SELECT * FROM similar_pairs WHERE run_id=?"
    params = [run_id]
    if student_id:
        sql += " AND (student_a=? OR student_b=?)"
        params.extend([student_id, student_id])
    if center_id:
        sql += " AND center_id=?"
        params.append(center_id)
    sql += " ORDER BY same_wrong DESC, similarity DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return query(sql, tuple(params))


def get_answers_for_student(run_id, student_id):
    """Retrieves cached raw answer string for a student."""
    row = query("SELECT answers FROM answers_cache WHERE run_id=? AND student_id=?", (run_id, student_id), one=True)
    return row["answers"] if row else ""


def delete_run(run_id):
    """Deletes an entire run and all related forensic tables."""
    con = get_connection()
    cur = con.cursor()
    for table in ["runs", "student_risk", "center_risk", "similar_pairs", "answers_cache"]:
        cur.execute(_fix_sql(f"DELETE FROM {table} WHERE " + ("id=?" if table == "runs" else "run_id=?")), (run_id,))
    if not USE_MYSQL:
        con.commit()
    con.close()
