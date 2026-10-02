-- MySQL schema (the app creates these automatically on first start; this file is for reference / manual setup)
CREATE DATABASE IF NOT EXISTS exam_forensics CHARACTER SET utf8mb4;
USE exam_forensics;

CREATE TABLE IF NOT EXISTS admins (
  id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(50) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
  id INT PRIMARY KEY AUTO_INCREMENT,
  filename VARCHAR(255), created_at VARCHAR(30),
  n_students INT, n_centers INT, n_questions INT,
  n_high_students INT, n_high_centers INT);

CREATE TABLE IF NOT EXISTS student_risk (
  id INT PRIMARY KEY AUTO_INCREMENT, run_id INT, student_id VARCHAR(64), center_id VARCHAR(64),
  score DOUBLE, center_z DOUBLE, max_similarity DOUBLE, max_same_wrong INT, n_partners INT,
  risk_score DOUBLE, risk_level VARCHAR(10), reasons TEXT, INDEX(run_id));

CREATE TABLE IF NOT EXISTS center_risk (
  id INT PRIMARY KEY AUTO_INCREMENT, run_id INT, center_id VARCHAR(64), n_students INT,
  avg_score DOUBLE, max_score DOUBLE, score_std DOUBLE, pct_flagged DOUBLE, mean_similarity DOUBLE,
  risk_score DOUBLE, risk_level VARCHAR(10), reasons TEXT, INDEX(run_id));

CREATE TABLE IF NOT EXISTS similar_pairs (
  id INT PRIMARY KEY AUTO_INCREMENT, run_id INT, student_a VARCHAR(64), student_b VARCHAR(64),
  center_id VARCHAR(64), similarity DOUBLE, same_wrong INT, INDEX(run_id));
