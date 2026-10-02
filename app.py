import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, abort
from werkzeug.security import check_password_hash
import db
import analysis
import report

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")
db.init_db()


def get_run(run_id):
    """Get one analysis run from the database (404 page if not found)."""
    run = db.query("SELECT * FROM runs WHERE id=?", (run_id,), one=True)
    if run is None:
        abort(404)
    return run


def color_of(level):
    if level == "High":
        return "#B3261E"
    if level == "Medium":
        return "#B7791F"
    return "#2E7D5B"


# ---------------- Login / Logout ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        admin = db.query("SELECT * FROM admins WHERE username=?", (username,), one=True)
        if admin is not None and check_password_hash(admin["password_hash"], password):
            session["user"] = admin["username"]
            return redirect(url_for("home"))
        flash("Wrong username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- Upload page ----------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    runs = db.query("SELECT * FROM runs ORDER BY id DESC LIMIT 20")
    return render_template("upload.html", runs=runs)


@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    exam_file = request.files.get("exam")
    key_file = request.files.get("key")
    if exam_file is None or exam_file.filename == "":
        flash("Choose the examination CSV file first.")
        return redirect(url_for("home"))

    try:
        df = pd.read_csv(exam_file)
        key = None
        if key_file is not None and key_file.filename != "":
            key = pd.read_csv(key_file).iloc[0].tolist()
        result = analysis.run_analysis(df, key)
    except Exception as e:
        flash(f"Could not analyse the file: {e}")
        return redirect(url_for("home"))

    run_id = db.save_run(exam_file.filename, result)
    return redirect(url_for("dashboard", run_id=run_id))


# ---------------- Dashboard ----------------
@app.route("/dashboard/<int:run_id>")
def dashboard(run_id):
    if "user" not in session:
        return redirect(url_for("login"))
    run = get_run(run_id)

    students = db.query("SELECT * FROM student_risk WHERE run_id=? ORDER BY risk_score DESC", (run_id,))
    centers = db.query("SELECT * FROM center_risk WHERE run_id=? ORDER BY risk_score DESC", (run_id,))
    pairs = db.query("SELECT * FROM similar_pairs WHERE run_id=? ORDER BY same_wrong DESC, similarity DESC LIMIT 15", (run_id,))

    # data for the three charts
    risk_scores = [s["risk_score"] for s in students]
    top_centers = centers[:15]
    center_names = [c["center_id"] for c in top_centers]
    center_risks = [c["risk_score"] for c in top_centers]
    center_colors = [color_of(c["risk_level"]) for c in top_centers]
    scatter_x = [s["max_same_wrong"] for s in students]
    scatter_y = [s["score"] for s in students]
    scatter_colors = [color_of(s["risk_level"]) for s in students]

    return render_template("dashboard.html", run=run, students=students[:20], centers=centers[:10], pairs=pairs,
                           risk_scores=risk_scores, center_names=center_names, center_risks=center_risks,
                           center_colors=center_colors, scatter_x=scatter_x, scatter_y=scatter_y,
                           scatter_colors=scatter_colors)


# ---------------- One student ----------------
@app.route("/student/<int:run_id>/<sid>")
def student(run_id, sid):
    if "user" not in session:
        return redirect(url_for("login"))
    run = get_run(run_id)
    s = db.query("SELECT * FROM student_risk WHERE run_id=? AND student_id=?", (run_id, sid), one=True)
    if s is None:
        abort(404)
    pairs = db.query("SELECT * FROM similar_pairs WHERE run_id=? AND (student_a=? OR student_b=?) ORDER BY same_wrong DESC",
                     (run_id, sid, sid))
    return render_template("student.html", run=run, s=s, pairs=pairs)


# ---------------- PDF report ----------------
@app.route("/report/<int:run_id>.pdf")
def pdf(run_id):
    if "user" not in session:
        return redirect(url_for("login"))
    run = get_run(run_id)
    students = pd.DataFrame(db.query("SELECT * FROM student_risk WHERE run_id=? ORDER BY risk_score DESC", (run_id,)))
    centers = pd.DataFrame(db.query("SELECT * FROM center_risk WHERE run_id=? ORDER BY risk_score DESC", (run_id,)))
    pairs = pd.DataFrame(db.query("SELECT * FROM similar_pairs WHERE run_id=? ORDER BY same_wrong DESC, similarity DESC", (run_id,)))
    pdf_file = report.build_pdf(run, students, centers, pairs)
    return send_file(pdf_file, mimetype="application/pdf", as_attachment=True,
                     download_name=f"irregularity_report_run{run_id}.pdf")


if __name__ == "__main__":
    app.run(debug=True)
