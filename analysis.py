"""
AI-Based Forensic Answer Sheet Analysis Engine.
Implements Psychometric Collusion Detection & Dual-Stage Isolation Forest Anomaly Detection.

Key Capabilities:
1. Universal Ingestion: Wide item-matrix (q1, q2.. qN) or compact string (answers="ABCD...")
2. Psychometric Identical Incorrect Responses (IIR / K-index Principle) with exact Binomial p-values
3. Item Difficulty Facility Value (FV) & Guttman Aberrance Index (Preknowledge Indicator)
4. Multi-tier Standardized Z-Scores (Center & Cohort level)
5. Dual-Stage Isolation Forest Anomaly Scoring (0-100)
6. Center-Level Leak Homogeneity Index (LHI) & Score Dispersion Analysis
7. Explainable AI (XAI) Forensic Reasoning Generation
8. Offline Matplotlib Charts + Data Structures for Plotly Interactive Visualizations
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


# -------------------------------------------------------------
# 1. Universal Data Loading & Preprocessing
# -------------------------------------------------------------
def load_and_preprocess(csv_path_or_df, key=None):
    """
    Parses and standardizes examination CSV datasets in either:
    - Wide format: student_id, center_id, [seat_no], q1, q2, ... qN
    - Compact format: student_id, center_id, [score], answers (string of length N)
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df)

    # Clean column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Required identifiers
    for col in ["student_id", "center_id"]:
        if col not in df.columns:
            matches = [c for c in df.columns if c.lower() == col]
            if matches:
                df.rename(columns={matches[0]: col}, inplace=True)
            else:
                raise ValueError(f"CSV must contain '{col}' column.")

    df = df.dropna(subset=["student_id", "center_id"])
    df = df.drop_duplicates(subset=["student_id"])
    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["center_id"] = df["center_id"].astype(str).str.strip()

    # Detect if Wide format (q1, q2, ...) or Compact format (answers column)
    q_cols = []
    for c in df.columns:
        if re.match(r"^q\d+$", c.lower()):
            q_cols.append(c)

    if len(q_cols) >= 5:
        q_cols.sort(key=lambda c: int(c[1:]))
        for c in q_cols:
            df[c] = df[c].fillna("-").astype(str).str.strip().str.upper()
    elif "answers" in [c.lower() for c in df.columns]:
        ans_col = [c for c in df.columns if c.lower() == "answers"][0]
        first_valid = df[ans_col].dropna().iloc[0] if not df[ans_col].dropna().empty else ""
        n_q = len(str(first_valid).strip())
        if n_q < 5:
            raise ValueError("Answers string must be at least 5 questions long.")
        q_cols = [f"q{i+1}" for i in range(n_q)]
        expanded = []
        for ans_str in df[ans_col]:
            cleaned_str = str(ans_str).strip().upper()
            chars = list(cleaned_str)
            if len(chars) < n_q:
                chars += ["-"] * (n_q - len(chars))
            else:
                chars = chars[:n_q]
            expanded.append(chars)
        expanded_df = pd.DataFrame(expanded, columns=q_cols, index=df.index)
        for c in q_cols:
            df[c] = expanded_df[c]
    else:
        raise ValueError(
            "Unrecognized format. CSV must contain either question columns (q1, q2...) "
            "or an 'answers' string column (e.g. ABCD...)."
        )

    # Resolve or Infer Answer Key
    if key is None or len(key) == 0:
        key = []
        for c in q_cols:
            valid_responses = df[c][df[c] != "-"]
            if not valid_responses.empty:
                key.append(valid_responses.mode()[0])
            else:
                key.append("A")
    else:
        if isinstance(key, str):
            key = list(key.strip().upper())
        elif isinstance(key, (list, tuple, np.ndarray, pd.Series)):
            key = [str(k).strip().upper() for k in key]
        key = key[: len(q_cols)]
        if len(key) < len(q_cols):
            key += ["A"] * (len(q_cols) - len(key))

    return df.reset_index(drop=True), q_cols, key


# -------------------------------------------------------------
# 2. Psychometric Collusion & Response Similarity
# -------------------------------------------------------------
def calculate_collusion_p_value(same_wrong, wrong_trials, num_options=4):
    """
    Computes the exact psychometric binomial p-value of observing >= same_wrong
    identical distractor selections under the null hypothesis H0 of independent guessing.
    Under 4 options, p_distractor = 1 / (num_options - 1) = 1/3.
    """
    if same_wrong <= 0 or wrong_trials <= 0:
        return 1.0
    p_distractor = 1.0 / max(num_options - 1, 1)
    # Survival function: P(X >= same_wrong) = binom.sf(same_wrong - 1, wrong_trials, p_distractor)
    p_val = float(binom.sf(same_wrong - 1, wrong_trials, p_distractor))
    return max(min(p_val, 1.0), 1e-15)


def compare_examinee_pair(ans_a, ans_b, key):
    """
    Computes pairwise psychometric indices between examinee A and B:
    - Hamming similarity percentage: % of identical choices (correct or wrong)
    - Identical Incorrect Responses (IIR): Count of identical wrong options
    - Exact binomial p-value under null hypothesis H0
    """
    n = min(len(ans_a), len(ans_b), len(key))
    if n == 0:
        return 0.0, 0, 1.0

    same = 0
    same_wrong = 0
    wrong_trials = 0
    for i in range(n):
        # Questions where at least one candidate answered
        if ans_a[i] != "-" and ans_b[i] != "-":
            if ans_a[i] == ans_b[i]:
                same += 1
                if ans_a[i] != key[i]:
                    same_wrong += 1
                    wrong_trials += 1
            else:
                if ans_a[i] != key[i] or ans_b[i] != key[i]:
                    wrong_trials += 1

    similarity_pct = (same / n) * 100.0
    p_val = calculate_collusion_p_value(same_wrong, max(wrong_trials, same_wrong))
    return round(similarity_pct, 2), same_wrong, p_val


def compute_collusion_features(df, q_cols, key):
    """
    Evaluates all intra-center examinee pairs for aberrant collusion patterns.
    Extracts maximum similarity, maximum identical wrong answers, collusion network degrees,
    and exact binomial p-values.
    """
    answers = df[q_cols].values.tolist()
    ids = df["student_id"].tolist()
    centers = df["center_id"].tolist()
    n = len(df)

    max_similarity = [0.0] * n
    max_same_wrong = [0] * n
    n_partners = [0] * n
    min_p_value = [1.0] * n
    pairs_list = []

    answers_dict = {}
    for i in range(n):
        answers_dict[ids[i]] = "".join(answers[i])

    unique_centers = df["center_id"].unique()
    for center in unique_centers:
        members = df.index[df["center_id"] == center].tolist()
        m_len = len(members)

        for x in range(m_len):
            for y in range(x + 1, m_len):
                idx_a = members[x]
                idx_b = members[y]

                sim, wrong, p_val = compare_examinee_pair(answers[idx_a], answers[idx_b], key)

                for s in (idx_a, idx_b):
                    if sim > max_similarity[s]:
                        max_similarity[s] = sim
                    if wrong > max_same_wrong[s]:
                        max_same_wrong[s] = wrong
                    if p_val < min_p_value[s]:
                        min_p_value[s] = p_val

                # Flag statistically suspicious collusion pair
                if sim >= config.MIN_SIMILARITY and wrong >= config.MIN_SAME_WRONG:
                    n_partners[idx_a] += 1
                    n_partners[idx_b] += 1
                    pairs_list.append({
                        "student_a": ids[idx_a],
                        "student_b": ids[idx_b],
                        "center_id": center,
                        "similarity": sim,
                        "same_wrong": wrong,
                        "p_value": p_val,
                    })

    pairs_df = pd.DataFrame(pairs_list)
    if pairs_df.empty:
        pairs_df = pd.DataFrame(columns=["student_a", "student_b", "center_id", "similarity", "same_wrong", "p_value"])

    return max_similarity, max_same_wrong, n_partners, pairs_df, answers_dict, min_p_value


# -------------------------------------------------------------
# 3. Item Aberrance & Facility Value Analysis (Guttman Aberrance)
# -------------------------------------------------------------
def compute_item_aberrance(df, q_cols, key):
    """
    Computes question facility values (difficulty) and determines examinee
    item-difficulty aberrance. An examinee who solves difficult questions
    correctly while failing easy questions exhibits classic item preknowledge aberrance.
    """
    n_students = len(df)
    n_questions = len(q_cols)

    # 1. Facility Value per item = % correct across population
    correct_matrix = np.zeros((n_students, n_questions), dtype=int)
    for q_idx, col in enumerate(q_cols):
        k = key[q_idx]
        correct_matrix[:, q_idx] = (df[col] == k).astype(int).values

    facility_values = correct_matrix.mean(axis=0)  # High FV = easy, Low FV = hard

    # Define Hard items (FV <= 35th percentile) and Easy items (FV >= 65th percentile)
    hard_cutoff = np.percentile(facility_values, 35)
    easy_cutoff = np.percentile(facility_values, 65)

    hard_mask = facility_values <= hard_cutoff
    easy_mask = facility_values >= easy_cutoff

    n_hard = max(int(np.sum(hard_mask)), 1)
    n_easy = max(int(np.sum(easy_mask)), 1)

    aberrance_scores = []
    for i in range(n_students):
        hard_acc = np.sum(correct_matrix[i, hard_mask]) / n_hard
        easy_acc = np.sum(correct_matrix[i, easy_mask]) / n_easy
        # Aberrance: High hard accuracy combined with low easy accuracy
        diff = hard_acc - easy_acc
        aberrance_scores.append(round(float(max(diff, 0.0) * 100.0), 2))

    return aberrance_scores, facility_values.tolist()


# -------------------------------------------------------------
# 4. Dual-Stage Isolation Forest Anomaly Detection
# -------------------------------------------------------------
def train_isolation_forest_risk(features_matrix, contamination=config.IF_CONTAMINATION, n_estimators=config.IF_ESTIMATORS):
    """
    Applies the Isolation Forest algorithm to detect anomalies in multi-dimensional space.
    Decision function inverted: outliers get highest scores, normalized to [0, 100].
    """
    if len(features_matrix) == 0:
        return np.array([])

    n_est = min(n_estimators, max(50, len(features_matrix) * 2))
    model = IsolationForest(
        n_estimators=n_est,
        contamination=contamination,
        random_state=config.IF_RANDOM_STATE,
    )
    model.fit(features_matrix)

    raw_scores = -model.score_samples(features_matrix)
    s_min, s_max = raw_scores.min(), raw_scores.max()

    if s_max - s_min > 1e-9:
        scaled = 100.0 * (raw_scores - s_min) / (s_max - s_min)
    else:
        scaled = np.zeros_like(raw_scores)

    return scaled


def assign_risk_level(score):
    if score >= config.HIGH_RISK_THRESHOLD:
        return "High"
    if score >= config.MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


# -------------------------------------------------------------
# 5. End-to-End Forensic Analysis Pipeline
# -------------------------------------------------------------
def run_analysis(csv_path_or_df, key=None):
    """
    Executes complete examination forensic investigation pipeline:
    1. Ingestion & Preprocessing
    2. Scoring & Answer Key Alignment
    3. Pairwise Collusion & IIR Feature Engineering with Binomial p-values
    4. Item Difficulty Facility Values & Guttman Aberrance Scoring
    5. Center & Cohort Standardization (Z-Scores)
    6. Dual-Stage Isolation Forest (Examinee + Center level)
    7. Explainable AI Reasoning Generation
    8. Offline Matplotlib Charts Export
    """
    df, q_cols, key = load_and_preprocess(csv_path_or_df, key)
    n_questions = len(q_cols)

    # Compute examinee scores (% correct)
    scores = []
    for row in df[q_cols].values.tolist():
        correct = sum(1 for i in range(n_questions) if row[i] == key[i])
        scores.append(round((correct / n_questions) * 100.0, 2))
    df["score"] = scores

    # Compute Collusion Features & exact p-values
    max_sim, max_wrong, partners, pairs_df, answers_dict, min_p_val = compute_collusion_features(df, q_cols, key)

    # Compute Item Aberrance Index
    aberrance_scores, facility_values = compute_item_aberrance(df, q_cols, key)

    students_df = pd.DataFrame({
        "student_id": df["student_id"],
        "center_id": df["center_id"],
        "score": scores,
        "max_similarity": max_sim,
        "max_same_wrong": max_wrong,
        "n_partners": partners,
        "aberrance_score": aberrance_scores,
        "min_p_value": min_p_val,
    })

    # Statistical Standardization (Center Z-score and Cohort Z-score)
    center_means = students_df.groupby("center_id")["score"].transform("mean")
    center_stds = students_df.groupby("center_id")["score"].transform("std").fillna(1.0).clip(lower=1.0)
    students_df["center_z"] = ((students_df["score"] - center_means) / center_stds).round(2)

    overall_mean = students_df["score"].mean()
    overall_std = max(float(students_df["score"].std()), 1.0)
    students_df["overall_z"] = ((students_df["score"] - overall_mean) / overall_std).round(2)

    # --- Stage 1: Examinee Anomaly Detection ---
    pos_center_z = students_df["center_z"].clip(lower=0.0).values
    pos_overall_z = students_df["overall_z"].clip(lower=0.0).values
    sim_norm = (students_df["max_similarity"] / 100.0).values
    wrong_norm = (students_df["max_same_wrong"] / max(n_questions, 1)).values
    partners_norm = (students_df["n_partners"] / 5.0).clip(upper=2.0).values
    aberr_norm = (students_df["aberrance_score"] / 100.0).values

    st_features = np.column_stack([pos_center_z, pos_overall_z, sim_norm, wrong_norm, partners_norm, aberr_norm])
    st_anomaly_scores = train_isolation_forest_risk(st_features)

    # Composite Calibration: Psychometric collusion evidence + statistical anomalies
    heuristic_boost = (
        (students_df["max_same_wrong"] >= config.MIN_SAME_WRONG).astype(int) * 20.0 +
        (students_df["n_partners"] > 0).astype(int) * 15.0 +
        (students_df["max_similarity"] >= config.MIN_SIMILARITY).astype(int) * 15.0
    )
    calibrated_student_risk = np.clip(st_anomaly_scores * 0.65 + heuristic_boost * 0.35, 0.0, 100.0).round(1)

    students_df["anomaly_score"] = st_anomaly_scores.round(1)
    students_df["risk_score"] = calibrated_student_risk
    students_df["risk_level"] = students_df["risk_score"].apply(assign_risk_level)

    # Generate Explainable AI (XAI) Forensic Justifications
    st_reasons = []
    for _, r in students_df.iterrows():
        reasons_list = []
        if r["n_partners"] > 0:
            p_val_str = f"p < {r['min_p_value']:.1e}" if r["min_p_value"] < 0.001 else f"p = {r['min_p_value']:.3f}"
            reasons_list.append(
                f"Answers match {int(r['n_partners'])} peer(s) with up to {int(r['max_same_wrong'])} identical errors ({p_val_str})"
            )
        elif r["max_similarity"] >= config.MIN_SIMILARITY:
            reasons_list.append(f"Elevated similarity ({r['max_similarity']:.1f}%) with peer")

        if r["center_z"] >= 2.0:
            reasons_list.append(f"Score is +{r['center_z']:.1f} SD above center mean")
        if r["aberrance_score"] >= 25.0:
            reasons_list.append(f"Aberrant response pattern: solves hard items while failing easy items")

        st_reasons.append("; ".join(reasons_list) if reasons_list else "No significant anomaly detected")
    students_df["reasons"] = st_reasons

    # --- Stage 2: Center Anomaly Detection ---
    students_df["flagged_int"] = (students_df["risk_level"] != "Low").astype(int)
    centers_agg = students_df.groupby("center_id").agg(
        n_students=("student_id", "count"),
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        score_std=("score", "std"),
        pct_flagged=("flagged_int", lambda s: (s.mean() * 100.0)),
        mean_similarity=("max_similarity", "mean"),
        mean_wrong=("max_same_wrong", "mean"),
    ).reset_index()

    centers_agg["score_std"] = centers_agg["score_std"].fillna(0.0).round(2)
    centers_agg["avg_score"] = centers_agg["avg_score"].round(2)
    centers_agg["pct_flagged"] = centers_agg["pct_flagged"].round(1)
    centers_agg["mean_similarity"] = centers_agg["mean_similarity"].round(2)
    centers_agg["mean_wrong"] = centers_agg["mean_wrong"].round(2)

    center_avg_z = ((centers_agg["avg_score"] - centers_agg["avg_score"].mean()) /
                    max(float(centers_agg["avg_score"].std()), 1.0)).clip(lower=0.0).values
    flagged_norm = (centers_agg["pct_flagged"] / 100.0).values
    c_wrong_norm = (centers_agg["mean_wrong"] / max(n_questions, 1)).values
    c_sim_norm = (centers_agg["mean_similarity"] / 100.0).values

    c_features = np.column_stack([center_avg_z, flagged_norm, c_wrong_norm, c_sim_norm])
    c_anomaly_scores = train_isolation_forest_risk(c_features)

    c_heuristic = (
        (centers_agg["pct_flagged"] >= 40.0).astype(int) * 30.0 +
        (centers_agg["mean_wrong"] >= 4.0).astype(int) * 20.0 +
        (center_avg_z >= 1.5).astype(int) * 20.0
    )
    calibrated_center_risk = np.clip(c_anomaly_scores * 0.60 + c_heuristic * 0.40, 0.0, 100.0).round(1)

    centers_agg["risk_score"] = calibrated_center_risk
    centers_agg["risk_level"] = centers_agg["risk_score"].apply(assign_risk_level)

    c_reasons = []
    for _, r in centers_agg.iterrows():
        reasons_list = []
        if r["pct_flagged"] >= 30.0:
            reasons_list.append(f"{r['pct_flagged']:.0f}% examinees exhibit collusion patterns")
        if r["avg_score"] >= 80.0:
            reasons_list.append(f"Abnormally high center average ({r['avg_score']:.1f}%)")
        if r["mean_wrong"] >= 3.0:
            reasons_list.append(f"Average identical errors per student: {r['mean_wrong']:.1f}")
        c_reasons.append("; ".join(reasons_list) if reasons_list else "Normal center performance distribution")
    centers_agg["reasons"] = c_reasons

    students_df.drop(columns=["flagged_int"], inplace=True)

    save_offline_charts(students_df, centers_agg)

    return {
        "students": students_df.sort_values("risk_score", ascending=False).reset_index(drop=True),
        "centers": centers_agg.sort_values("risk_score", ascending=False).reset_index(drop=True),
        "pairs": pairs_df.sort_values(["same_wrong", "similarity"], ascending=False).reset_index(drop=True),
        "n_questions": n_questions,
        "answer_key": key,
        "answers_dict": answers_dict,
        "facility_values": facility_values,
    }


# -------------------------------------------------------------
# 6. Offline Static Charts Generation (Matplotlib)
# -------------------------------------------------------------
def save_offline_charts(students_df, centers_df, folder=config.CHARTS_DIR):
    """Generates clean publication-ready charts saved to static/charts."""
    os.makedirs(folder, exist_ok=True)

    # 1. Student Risk Distribution
    plt.figure(figsize=(6, 3.5))
    risks = students_df["risk_score"].values
    plt.hist(risks, bins=15, color="#1e3a8a", edgecolor="white", alpha=0.9)
    plt.title("Examinee Forensic Risk Score Distribution", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Forensic Risk Score (0 - 100)", fontsize=9)
    plt.ylabel("Examinee Count", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "student_risk.png"), dpi=130)
    plt.close()

    # 2. Riskiest Examination Centers
    plt.figure(figsize=(7, 3.5))
    top_centers = centers_df.head(10)
    colors = ["#dc2626" if r == "High" else "#d97706" if r == "Medium" else "#16a34a" for r in top_centers["risk_level"]]
    plt.bar(top_centers["center_id"], top_centers["risk_score"], color=colors, edgecolor="black", linewidth=0.5)
    plt.title("Top High-Risk Examination Centers", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Examination Center ID", fontsize=9)
    plt.ylabel("Center Risk Score (0 - 100)", fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "center_risk.png"), dpi=130)
    plt.close()

    # 3. Score vs Identical Wrong Answers Scatter
    plt.figure(figsize=(6, 3.5))
    scatter_colors = ["#dc2626" if r == "High" else "#d97706" if r == "Medium" else "#16a34a" for r in students_df["risk_level"]]
    plt.scatter(students_df["max_same_wrong"], students_df["score"], c=scatter_colors, alpha=0.7, edgecolors="none", s=25)
    plt.title("Score vs. Identical Incorrect Responses (IIR)", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Max Identical Wrong Answers with Peer", fontsize=9)
    plt.ylabel("Examinee Score (%)", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "collusion_scatter.png"), dpi=130)
    plt.close()
