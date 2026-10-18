# AI-Based Forensic Answer Sheet Analysis System for Detecting Examination Irregularities

Final Year B.Tech CSE Mega Project  
**Annasaheb Dange College of Engineering & Technology, Ashta**  
**Affiliated to Shivaji University, Kolhapur**  
**Under the Guidance of:** Prof. Mrs. A. M. Babar  
**Team:** Swapnali Jadhav, Shreyas Mane, Prafulla Olekar, Shantanu Jadhav  

---

## 📌 Project Overview
This project is an AI/ML post-examination forensic investigation system designed to detect examination malpractice, pairwise answer copying (collusion), and organized question paper leaks.

### Key Capabilities & Scientific Features
- **Universal Data Ingestion**: Seamlessly ingests both Item-Level Matrices (`q1, q2... qN`) and Compact Strings (`answers="ABCD..."`).
- **Psychometric Collusion Engine**: Computes Hamming similarity and **Identical Incorrect Responses (IIR / K-index principle)**.
- **Dual-Stage Isolation Forest Anomaly Detection**:
  - *Stage 1 (Examinee Level)*: Identifies anomalous candidate behavior combining score deviations ($Z_{\text{center}}, Z_{\text{overall}}$) and collusion indices.
  - *Stage 2 (Center Level)*: Evaluates examination centers for organized malpractice and question paper leaks.
- **Explainable AI (XAI)**: Generates human-interpretable forensic justifications for every flagged examinee and center.
- **Interactive Visualizations (Plotly + Matplotlib)**: Dynamic interactive scatter plots, center risk rankings, and risk score distributions.
- **Question-by-Question Comparative Matrix**: Visual side-by-side audit of test sheets highlighting matching correct answers (green) and matching identical wrong answers (red).
- **Official Forensic PDF Dossier**: Downloadable publication-grade PDF investigation reports built with ReportLab.
- **Research Benchmarking Suite (`evaluate.py`)**: Computes Precision, Recall, F1-Score, and Confusion Matrix against known ground truth.

---

## 🚀 How to Run the Project (Step-by-Step)

### Step 1: Open Terminal / PowerShell
Open PowerShell or Command Prompt and navigate to the project directory:
```powershell
cd C:\Users\JSwapnali\Desktop\MegaProject
```

### Step 2: Activate the Virtual Environment
```powershell
.\venv\Scripts\activate
```
*(You will see `(venv)` appear at the start of your terminal line).*

### Step 3: Verify / Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: (Optional) Generate Fresh Datasets
```powershell
# Generates 1,800-candidate research benchmark with injected cheaters:
python generate_data.py

# Generates 160-candidate compact viva demo dataset:
python generate_sample.py
```

### Step 5: (Optional) Run the Scientific Research Benchmark
To see the empirical evaluation metrics (Precision, Recall, F1, and baseline comparisons):
```powershell
python evaluate.py
```

### Step 6: Start the Web Application
```powershell
python app.py
```

### Step 7: Open the Dashboard in your Browser
Open Google Chrome or Microsoft Edge and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔑 Login Credentials
- **Username:** `admin`
- **Password:** `admin123`

---

## 🧭 Navigation & Demo Walkthrough for Viva / Review

1. **Upload & Run (`/upload`)**:
   - Click **"Research Benchmark Dataset"** to analyze 1,800 examinees across 30 centers in 1 click!
   - Or upload your own custom examination CSV file.
2. **Dashboard (`/dashboard`)**:
   - Inspect summary KPIs, interactive Plotly charts, and top suspect candidates & centers.
3. **Examinee Roster (`/students`)**:
   - Filter candidates by **High**, **Medium**, or **Low** risk, or search by ID.
4. **Examinee Forensic Dossier (`/student/<run_id>/<sid>`)**:
   - Click on any candidate to inspect their Z-deviation, collusion partners, and explainable AI reasons.
5. **Question Matrix Comparator (`/pair_compare/<run_id>/<sid1>/<sid2>`)**:
   - Click **"Compare Sheets Side-by-Side"** to show the examiners the exact test questions where two candidates shared identical incorrect choices!
6. **Examination Centers (`/centers`)**:
   - View center malpractice rates, score standard deviations, and identify leak centers (e.g. C03, C19).
7. **Download PDF Report (`/report/<run_id>.pdf`)**:
   - Click to download the formal ReportLab PDF audit dossier.
8. **Research Evaluation (`/evaluation`)**:
   - Show examiners the quantitative metrics: **Precision = 1.00**, **F1-Score = 0.81**, and baseline ablation tables!

---

## 🗄️ Database Configuration (SQLite vs MySQL)
- **Default (Zero-Setup Demo):** Uses SQLite (`data/forensic_exam.db`). No configuration or database installation needed.
- **MySQL (College Stack):**
  1. Open `config.py` and set `DB_TYPE = "mysql"`
  2. Ensure MySQL is running on `localhost:3306` with database `exam_forensics`
  3. Start `python app.py` (tables are created automatically).

---

## 📁 Key File Structure
| File | Role / Purpose |
| :--- | :--- |
| `app.py` | Flask web server, authentication, routing, and Plotly visualization endpoints |
| `analysis.py` | Universal CSV parser, psychometric IIR collusion engine, dual-stage Isolation Forest |
| `db.py` | Relational storage for runs, examinees, centers, pairs, and answers cache (SQLite & MySQL) |
| `config.py` | Configuration settings, database toggle, and algorithm thresholds |
| `report.py` | Institutional forensic audit PDF generator using ReportLab |
| `evaluate.py` | Academic benchmark suite measuring Precision, Recall, F1 against ground truth |
| `generate_data.py` | Realistic synthetic test simulator with latent examinee abilities and injected cheating |
| `generate_sample.py` | Compact format test data generator |
| `test_system.py` | Comprehensive automated test suite verifying 100% system health |
