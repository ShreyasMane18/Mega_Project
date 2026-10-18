"""
End-to-End Test Suite for Integrated Forensic Examination System.
Verifies all algorithmic pipelines, database operations, PDF generation, and web endpoints.
"""

import os
import io
import pandas as pd
import config
import db
import analysis
import report
import evaluate
from app import app


def test_system():
    print("[1/6] Testing Database Initialization & Admin Authentication...")
    db.init_db()
    admin = db.verify_user(config.ADMIN_USER, config.ADMIN_PASS)
    assert admin is not None, "Failed to verify admin credentials!"
    assert admin["username"] == config.ADMIN_USER
    print("  -> DB initialized and admin verified successfully.")

    print("\n[2/6] Testing Universal Forensic Engine on Compact Format (sample_exam.csv)...")
    sample_df = pd.read_csv("data/sample_exam.csv")
    res_sample = analysis.run_analysis(sample_df)
    assert len(res_sample["students"]) == 160
    assert len(res_sample["centers"]) == 8
    assert res_sample["n_questions"] == 20
    print(f"  -> Compact format processed: {len(res_sample['students'])} students across {len(res_sample['centers'])} centers.")

    print("\n[3/6] Testing Universal Forensic Engine on Wide Format (exam_data.csv)...")
    exam_df = pd.read_csv("data/exam_data.csv")
    key = pd.read_csv("data/answer_key.csv").iloc[0].tolist()
    res_wide = analysis.run_analysis(exam_df, key)
    assert len(res_wide["students"]) == 1800
    assert len(res_wide["centers"]) == 30
    assert res_wide["n_questions"] == 60
    run_id = db.save_run("Research_Benchmark_Exam.csv", res_wide)
    assert run_id is not None and run_id > 0
    print(f"  -> Wide format processed: {len(res_wide['students'])} students saved as Run #{run_id}.")

    print("\n[4/6] Testing PDF Audit Dossier Generation (ReportLab)...")
    run_rec = db.get_run(run_id)
    st_df = pd.DataFrame(db.get_students(run_id))
    c_df = pd.DataFrame(db.get_centers(run_id))
    p_df = pd.DataFrame(db.get_pairs(run_id))
    pdf_buf = report.build_pdf(run_rec, st_df, c_df, p_df)
    pdf_bytes = pdf_buf.getvalue()
    assert pdf_bytes.startswith(b"%PDF"), "Generated file is not a valid PDF!"
    assert len(pdf_bytes) > 5000, f"PDF suspiciously small: {len(pdf_bytes)} bytes"
    print(f"  -> PDF generated cleanly ({len(pdf_bytes):,} bytes).")

    print("\n[5/6] Testing Research Benchmark Evaluation Suite...")
    eval_res = evaluate.evaluate_dataset()
    assert eval_res["total_examinees"] == 1800
    assert eval_res["injected_cheaters"] == 96
    assert eval_res["metrics_high"]["precision"] > 0.90
    print(f"  -> Benchmark evaluation passed: High-Risk Precision = {eval_res['metrics_high']['precision']:.2f}, F1 = {eval_res['metrics_high']['f1']:.2f}")

    print("\n[6/6] Testing Flask Web Application Routes & Endpoints...")
    client = app.test_client()

    # 1. Unauthenticated redirect
    r = client.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]

    # 2. Login
    r = client.post("/login", data={"username": config.ADMIN_USER, "password": config.ADMIN_PASS}, follow_redirects=True)
    assert r.status_code == 200
    assert b"Forensic Intelligence Dashboard" in r.data

    # 3. Dashboard
    r = client.get(f"/dashboard/{run_id}")
    assert r.status_code == 200
    assert b"Examinees Audited" in r.data
    assert b"Plotly.newPlot" in r.data

    # 4. Students
    r = client.get(f"/students/{run_id}")
    assert r.status_code == 200
    assert b"Examinee Forensic Audit Roster" in r.data

    # 5. Student Detail
    top_student = db.get_students(run_id, limit=1)[0]["student_id"]
    r = client.get(f"/student/{run_id}/{top_student}")
    assert r.status_code == 200
    assert bytes(top_student, "utf-8") in r.data

    # 6. Centers
    r = client.get(f"/centers/{run_id}")
    assert r.status_code == 200
    assert b"Examination Center Forensic Ranking" in r.data

    # 7. Center Detail
    top_center = db.get_centers(run_id, limit=1)[0]["center_id"]
    r = client.get(f"/center/{run_id}/{top_center}")
    assert r.status_code == 200
    assert bytes(top_center, "utf-8") in r.data

    # 8. Pairs
    r = client.get(f"/pairs/{run_id}")
    assert r.status_code == 200
    assert b"Collusion Pairs Forensic Analysis" in r.data

    # 9. Pair Compare Matrix
    pairs = db.get_pairs(run_id, limit=1)
    if pairs:
        p = pairs[0]
        r = client.get(f"/pair_compare/{run_id}/{p['student_a']}/{p['student_b']}")
        assert r.status_code == 200
        assert b"Item-by-Item Comparative Matrix" in r.data

    # 10. Printable Report
    r = client.get(f"/report/{run_id}")
    assert r.status_code == 200
    assert b"Forensic Investigation Audit Dossier" in r.data

    # 11. PDF Download
    r = client.get(f"/report/{run_id}.pdf")
    assert r.status_code == 200
    assert r.data.startswith(b"%PDF")

    # 12. Research Benchmark View
    r = client.get(f"/evaluation/{run_id}")
    assert r.status_code == 200
    assert b"Research Validation" in r.data

    # 13. Runs View
    r = client.get("/runs")
    assert r.status_code == 200
    assert b"Forensic Analysis History" in r.data

    # 14. Research Paper View
    r = client.get("/paper")
    assert r.status_code == 200
    assert b"Research Paper Manuscript" in r.data

    # 15. CSV Data Exports
    r = client.get(f"/export/students/{run_id}")
    assert r.status_code == 200
    assert "text/csv" in r.headers["Content-Type"]

    r = client.get(f"/export/centers/{run_id}")
    assert r.status_code == 200
    assert "text/csv" in r.headers["Content-Type"]

    print("  -> All 16 web routes, paper view, and CSV exports validated successfully (HTTP 200).")
    print("\n==================================================================")
    print(" [SUCCESS] ALL SYSTEM TESTS PASSED CLEANLY! System is 100% operational.")
    print("==================================================================")


if __name__ == "__main__":
    test_system()
