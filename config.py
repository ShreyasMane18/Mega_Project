"""
Configuration settings for AI-Based Forensic Answer Sheet Analysis System.
Supports both SQLite (out-of-the-box demo/viva) and MySQL (production/college stack).
"""
import os

# Application Security
SECRET_KEY = os.getenv("SECRET_KEY", "forensic-exam-ai-secret-key-2026")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

# Database Backend: 'sqlite' or 'mysql'
DB_TYPE = os.getenv("DB_BACKEND", os.getenv("DB_TYPE", "sqlite")).lower()
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join("data", "forensic_exam.db"))

# MySQL Connection Details (used when DB_TYPE == 'mysql')
MYSQL_HOST = os.getenv("DB_HOST", "localhost")
MYSQL_PORT = int(os.getenv("DB_PORT", 3306))
MYSQL_USER = os.getenv("DB_USER", "root")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD", "")
MYSQL_DB = os.getenv("DB_NAME", "exam_forensics")

# Forensic Detection Thresholds
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 75.0))  # Minimum % identical answers to flag a pair
MIN_SAME_WRONG = int(os.getenv("MIN_SAME_WRONG", 6))       # Minimum count of identical incorrect answers (IIR)
HIGH_RISK_THRESHOLD = 70.0                                 # Risk score >= 70 -> High Risk
MEDIUM_RISK_THRESHOLD = 45.0                               # Risk score >= 45 -> Medium Risk

# Isolation Forest Hyperparameters
IF_ESTIMATORS = 250
IF_CONTAMINATION = "auto"
IF_RANDOM_STATE = 42

# File Paths
DATA_DIR = "data"
CHARTS_DIR = os.path.join("static", "charts")
SAMPLE_EXAM_CSV = os.path.join("data", "exam_data.csv")
SAMPLE_KEY_CSV = os.path.join("data", "answer_key.csv")
GROUND_TRUTH_CSV = os.path.join("data", "ground_truth.csv")
