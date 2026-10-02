"""End-to-end smoke test: python -m pytest tests  (or python tests/test_app.py)"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
import app as A

def test_flow():
    c = A.app.test_client()
    assert c.get("/").status_code == 302                                   # login required
    assert c.post("/login", data={"username": "admin", "password": "admin123"}).status_code == 302
    r = c.post("/upload", data={"exam": open("data/exam_data.csv", "rb"), "key": open("data/answer_key.csv", "rb")},
               content_type="multipart/form-data")
    assert r.status_code == 302; url = r.headers["Location"]
    assert b"High-risk students" in c.get(url).data
    pdf = c.get(url.replace("dashboard", "report") + ".pdf"); assert pdf.data[:4] == b"%PDF"
    sid = A.db.query("SELECT student_id FROM student_risk ORDER BY risk_score DESC LIMIT 1", one=True)["student_id"]
    assert c.get(f"/student/1/{sid}").status_code == 200

if __name__ == "__main__": test_flow(); print("all good")
