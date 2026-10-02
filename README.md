# AI-Based Forensic Answer Sheet Analysis System

Flask + Scikit-learn (Isolation Forest) + MySQL/SQLite + Plotly + ReportLab.

## Run it (5 minutes)
```bash
pip install -r requirements.txt
python generate_data.py          # makes data/exam_data.csv, answer_key.csv, ground_truth.csv
python evaluate.py               # precision / recall against the injected cheating
python app.py                    # open http://127.0.0.1:5000  (admin / admin123)
```
Upload `data/exam_data.csv` (+ `data/answer_key.csv`) on the home page.

## Use MySQL (as in your synopsis)
```bash
# run schema.sql once (or let the app create tables), then:
set DB_BACKEND=mysql & set DB_HOST=localhost & set DB_USER=root & set DB_PASSWORD=yourpw & set DB_NAME=exam_forensics   (Windows)
python app.py
```
Change the admin login with `ADMIN_USER` / `ADMIN_PASSWORD` and set `SECRET_KEY` before first run.

## Files
| File | Job |
|---|---|
| `generate_data.py` | Synthetic exam data with injected copying pairs and 2 "leak" centres |
| `analysis.py` | Cleaning, similarity features, Isolation Forest, risk scores (0-100) |
| `evaluate.py` | Precision / recall / F1 versus ground truth (use in your results chapter) |
| `db.py`, `schema.sql` | MySQL/SQLite storage: admins, runs, student_risk, center_risk, similar_pairs |
| `app.py`, `templates/`, `static/` | Login, upload, dashboard, student detail, PDF download |
| `report.py` | PDF investigation report |
| `tests/test_app.py` | End-to-end smoke test |

## How it works (for your viva)
1. **Clean**: drop duplicates, fill blanks.
2. **Similarity features**: within each centre, compare every pair of students. Keep % identical answers and, most importantly, identical WRONG answers (two students rarely make the same mistakes by chance).
3. **Student features**: score vs centre average, score vs everyone, max similarity, max identical wrong answers, number of suspicious partners.
4. **Isolation Forest** scores how easy each student is to isolate, and the score is rescaled to 0-100. Centres get the same treatment on centre-level features (average score, % flagged students, ...).
5. **Levels**: High >= 70, Medium >= 45 (tunable at the top of `analysis.py`).
6. Results are saved to the database and shown on the dashboard, with a PDF report.

## Limits to state honestly
- Isolation Forest is unsupervised: it finds unusual patterns, not proof of cheating. Reports are leads for investigation.
- Tested on synthetic data; run it on a real (anonymised) dataset if you can get one.
- Pair comparison is per centre and grows with centre size squared (fine up to a few thousand per centre).
- Set `MIN_SIM` / `MIN_SAME_WRONG` in `analysis.py` to match your real exam's number of questions and options.
