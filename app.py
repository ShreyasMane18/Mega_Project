"""
AI-Based Forensic Answer Sheet Analysis System - Web Application.
Final Year B.Tech CSE Mega Project, ADCoET Ashta.

Provides:
- Anomaly Detection Dashboard with Interactive Plotly Visualizations
- Examinee Forensic Dossiers & Question-by-Question Response Matrix Audits
- Examination Center Malpractice & Question Leak Analytics
- Automated ReportLab PDF Investigation Audit Dossier Downloads
- Research Benchmark & Empirical Evaluation Suite
- In-App Full Research Paper Reader & CSV Data Exports
"""

import os
import io
import csv
from functools import wraps
import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    send_file,
    abort,
)

import config
import db
import analysis
import report
import evaluate

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Ensure database tables and initial storage exist
db.init_db()


# -------------------------------------------------------------
# Security & Context Helpers
# -------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_global_vars():
    """Provides active_run_id globally to templates for navigation links."""
    active_id = session.get("active_run_id")
    if not active_id:
        latest = db.get_latest_run()
        if latest:
            active_id = latest["id"]
            session["active_run_id"] = active_id
    return {"active_run_id": active_id}


def resolve_run(run_id=None):
    """Resolves target run by ID or falls back to latest run."""
    if run_id:
        r = db.get_run(run_id)
        if not r:
            abort(404)
        session["active_run_id"] = r["id"]
        return r
    latest = db.get_latest_run()
    if latest:
        session["active_run_id"] = latest["id"]
        return latest
    return None


def get_risk_color(level):
    if level == "High":
        return "#dc2626"
    if level == "Medium":
        return "#d97706"
    return "#16a34a"


# -------------------------------------------------------------
# Authentication Routes
# -------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        admin = db.verify_user(username, password)
        if admin:
            session["user"] = admin["username"]
            flash(f"Authenticated successfully as {admin['username']}.")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials. Please verify username and password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.")
    return redirect(url_for("login"))


# -------------------------------------------------------------
# Dashboard & Overview
# -------------------------------------------------------------
@app.route("/")
@app.route("/dashboard")
@app.route("/dashboard/<int:run_id>")
@login_required
def dashboard(run_id=None):
    run = resolve_run(run_id)
    if not run:
        return redirect(url_for("upload"))

    rid = run["id"]
    students = db.get_students(rid, limit=2000)
    top_students = students[:15]
    centers = db.get_centers(rid)
    top_centers = centers[:10]
    pairs = db.get_pairs(rid, limit=30)

    # Plotly Visualizations Data Preparation
    risk_scores = [float(s["risk_score"]) for s in students]

    chart_centers = centers[:15]
    center_names = [str(c["center_id"]) for c in chart_centers]
    center_risks = [float(c["risk_score"]) for c in chart_centers]
    center_colors = [get_risk_color(c["risk_level"]) for c in chart_centers]

    scatter_x = [int(s["max_same_wrong"]) for s in students]
    scatter_y = [float(s["score"]) for s in students]
    scatter_colors = [get_risk_color(s["risk_level"]) for s in students]
    scatter_texts = [f"{s['student_id']} ({s['risk_level']})" for s in students]

    chart_data = {
        "risk_scores": risk_scores,
        "center_names": center_names,
        "center_risks": center_risks,
        "center_colors": center_colors,
        "scatter_x": scatter_x,
        "scatter_y": scatter_y,
        "scatter_colors": scatter_colors,
        "scatter_texts": scatter_texts,
    }

    return render_template(
        "dashboard.html",
        run=run,
        top_students=top_students,
        top_centers=top_centers,
        pairs=pairs,
        chart_data=chart_data,
    )


# -------------------------------------------------------------
# Upload & Dataset Ingestion
# -------------------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        exam_file = request.files.get("exam")
        key_file = request.files.get("key")

        if not exam_file or exam_file.filename == "":
            flash("Please select an examination CSV dataset file.")
            return redirect(url_for("upload"))

        try:
            exam_df = pd.read_csv(exam_file)
            key = None
            if key_file and key_file.filename != "":
                key_df = pd.read_csv(key_file)
                key = key_df.iloc[0].tolist()

            result = analysis.run_analysis(exam_df, key)
            run_id = db.save_run(exam_file.filename, result)
            session["active_run_id"] = run_id
            flash(f"Forensic analysis completed for {exam_file.filename} (Run #{run_id}).")
            return redirect(url_for("dashboard", run_id=run_id))
        except Exception as exc:
            flash(f"Analysis error: {exc}")
            return redirect(url_for("upload"))

    recent_runs = db.get_all_runs()[:5]
    return render_template("upload.html", recent_runs=recent_runs)


@app.route("/run_preset", methods=["POST"])
@login_required
def run_preset():
    preset = request.form.get("preset", "research")
    try:
        if preset == "research":
            csv_path = config.SAMPLE_EXAM_CSV
            key_path = config.SAMPLE_KEY_CSV
            filename = "Research_Benchmark_Exam.csv"
        else:
            csv_path = os.path.join(config.DATA_DIR, "sample_exam.csv")
            key_path = None
            filename = "Sample_Exam_Viva.csv"

        if not os.path.exists(csv_path):
            flash(f"Preset file '{csv_path}' not found on disk.")
            return redirect(url_for("upload"))

        exam_df = pd.read_csv(csv_path)
        key = None
        if key_path and os.path.exists(key_path):
            key = pd.read_csv(key_path).iloc[0].tolist()

        result = analysis.run_analysis(exam_df, key)
        run_id = db.save_run(filename, result)
        session["active_run_id"] = run_id
        flash(f"Preset '{preset}' analyzed successfully as Run #{run_id}.")
        return redirect(url_for("dashboard", run_id=run_id))
    except Exception as exc:
        flash(f"Failed to execute preset: {exc}")
        return redirect(url_for("upload"))


# -------------------------------------------------------------
# Examinee Roster & Forensic Dossier
# -------------------------------------------------------------
@app.route("/students")
@app.route("/students/<int:run_id>")
@login_required
def students(run_id=None):
    run = resolve_run(run_id)
    if not run:
        return redirect(url_for("upload"))

    risk_filter = request.args.get("risk_level")
    search_query = request.args.get("search", "").strip()

    students_list = db.get_students(
        run["id"],
        risk_level=risk_filter if risk_filter in ["High", "Medium", "Low"] else None,
        search=search_query if search_query else None,
    )

    return render_template(
        "students.html",
        run=run,
        students=students_list,
        current_risk=risk_filter,
        current_search=search_query,
    )


@app.route("/student/<int:run_id>/<sid>")
@login_required
def student_detail(run_id, sid):
    run = resolve_run(run_id)
    s = db.get_student(run_id, sid)
    if not s:
        abort(404)

    pairs = db.get_pairs(run_id, student_id=sid)
    answers_str = db.get_answers_for_student(run_id, sid)

    return render_template(
        "student_detail.html",
        run=run,
        s=s,
        pairs=pairs,
        answers_str=answers_str,
    )


# -------------------------------------------------------------
# Question-by-Question Pair Inspector Matrix
# -------------------------------------------------------------
@app.route("/pair_compare/<int:run_id>/<sid1>/<sid2>")
@login_required
def pair_compare(run_id, sid1, sid2):
    run = resolve_run(run_id)
    s1 = db.get_student(run_id, sid1)
    s2 = db.get_student(run_id, sid2)
    if not (s1 and s2):
        abort(404)

    ans1 = db.get_answers_for_student(run_id, sid1)
    ans2 = db.get_answers_for_student(run_id, sid2)

    n_q = max(len(ans1), len(ans2), run["n_questions"])

    key = []
    if os.path.exists(config.SAMPLE_KEY_CSV):
        key = pd.read_csv(config.SAMPLE_KEY_CSV).iloc[0].tolist()

    matrix = []
    same_correct = 0
    same_wrong = 0
    wrong_trials = 0
    for i in range(n_q):
        a1 = ans1[i] if i < len(ans1) else "-"
        a2 = ans2[i] if i < len(ans2) else "-"
        k = key[i] if i < len(key) else ""

        match_correct = False
        match_wrong = False

        if a1 != "-" and a2 != "-":
            if a1 == a2:
                if k and a1 == k:
                    match_correct = True
                    same_correct += 1
                else:
                    match_wrong = True
                    same_wrong += 1
                    wrong_trials += 1
            else:
                if (k and a1 != k) or (k and a2 != k):
                    wrong_trials += 1

        matrix.append({
            "q_num": i + 1,
            "a1": a1,
            "a2": a2,
            "key": k,
            "match_correct": match_correct,
            "match_wrong": match_wrong,
        })

    similarity = ((same_correct + same_wrong) / max(n_q, 1)) * 100.0
    p_value = analysis.calculate_collusion_p_value(same_wrong, max(wrong_trials, same_wrong))

    return render_template(
        "pair_compare.html",
        run=run,
        s1_id=sid1,
        s2_id=sid2,
        center_id=s1["center_id"],
        matrix=matrix,
        total_questions=n_q,
        same_wrong=same_wrong,
        same_correct=same_correct,
        similarity=similarity,
        p_value=p_value,
    )


# -------------------------------------------------------------
# Centers Roster & Center Dossier
# -------------------------------------------------------------
@app.route("/centers")
@app.route("/centers/<int:run_id>")
@login_required
def centers(run_id=None):
    run = resolve_run(run_id)
    if not run:
        return redirect(url_for("upload"))

    centers_list = db.get_centers(run["id"])
    return render_template("centers.html", run=run, centers=centers_list)


@app.route("/center/<int:run_id>/<cid>")
@login_required
def center_detail(run_id, cid):
    run = resolve_run(run_id)
    center = db.get_center(run_id, cid)
    if not center:
        abort(404)

    students_in_center = db.get_students(run_id, center_id=cid)
    pairs_in_center = db.get_pairs(run_id, center_id=cid)

    return render_template(
        "center_detail.html",
        run=run,
        center=center,
        students=students_in_center,
        pairs=pairs_in_center,
    )


# -------------------------------------------------------------
# Collusion Pairs Explorer
# -------------------------------------------------------------
@app.route("/pairs")
@app.route("/pairs/<int:run_id>")
@login_required
def pairs(run_id=None):
    run = resolve_run(run_id)
    if not run:
        return redirect(url_for("upload"))

    pairs_list = db.get_pairs(run["id"], limit=100)
    return render_template("pairs.html", run=run, pairs=pairs_list)


# -------------------------------------------------------------
# Reports: HTML & ReportLab PDF
# -------------------------------------------------------------
@app.route("/report")
@app.route("/report/<int:run_id>")
@login_required
def report_view(run_id=None):
    run = resolve_run(run_id)
    if not run:
        return redirect(url_for("upload"))

    students_list = db.get_students(run["id"], limit=50)
    centers_list = db.get_centers(run["id"])
    pairs_list = db.get_pairs(run["id"], limit=30)

    return render_template(
        "report.html",
        run=run,
        students=students_list,
        centers=centers_list,
        pairs=pairs_list,
    )


@app.route("/report/<int:run_id>.pdf")
@login_required
def report_pdf(run_id):
    run = resolve_run(run_id)
    if not run:
        abort(404)

    students_df = pd.DataFrame(db.get_students(run_id))
    centers_df = pd.DataFrame(db.get_centers(run_id))
    pairs_df = pd.DataFrame(db.get_pairs(run_id))

    pdf_buffer = report.build_pdf(run, students_df, centers_df, pairs_df)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"Forensic_Investigation_Report_Run_{run_id}.pdf",
    )


# -------------------------------------------------------------
# Research Benchmark & Empirical Evaluation Suite
# -------------------------------------------------------------
@app.route("/evaluation")
@app.route("/evaluation/<int:run_id>")
@login_required
def evaluation_view(run_id=None):
    run = resolve_run(run_id)
    try:
        eval_res = evaluate.evaluate_dataset()
    except Exception as exc:
        flash(f"Evaluation benchmark notice: {exc}")
        return redirect(url_for("dashboard"))

    return render_template("evaluation.html", run=run, eval_res=eval_res)


# -------------------------------------------------------------
# Research Paper Manuscript Reader
# -------------------------------------------------------------
@app.route("/paper")
@login_required
def paper_view():
    return render_template("paper.html")


# -------------------------------------------------------------
# CSV Data Export Endpoints
# -------------------------------------------------------------
@app.route("/export/students/<int:run_id>")
@login_required
def export_students(run_id):
    run = resolve_run(run_id)
    students = db.get_students(run["id"])
    if not students:
        abort(404)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(students[0].keys()))
    writer.writeheader()
    writer.writerows(students)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=Examinees_Run_{run_id}.csv"},
    )


@app.route("/export/centers/<int:run_id>")
@login_required
def export_centers(run_id):
    run = resolve_run(run_id)
    centers = db.get_centers(run["id"])
    if not centers:
        abort(404)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(centers[0].keys()))
    writer.writeheader()
    writer.writerows(centers)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=Centers_Run_{run_id}.csv"},
    )


# -------------------------------------------------------------
# Run History Management
# -------------------------------------------------------------
@app.route("/runs")
@login_required
def runs_list():
    all_runs = db.get_all_runs()
    return render_template("runs.html", runs=all_runs)


@app.route("/delete_run/<int:run_id>", methods=["POST"])
@login_required
def delete_run_route(run_id):
    db.delete_run(run_id)
    flash(f"Run #{run_id} was removed.")
    if session.get("active_run_id") == run_id:
        session.pop("active_run_id", None)
    return redirect(url_for("runs_list"))


# -------------------------------------------------------------
# Application Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    db.init_db()
    print("AI Forensic Answer Sheet Analysis Server starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
