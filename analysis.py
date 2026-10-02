"""
Forensic analysis of answer sheets.
Steps: 1) clean data  2) compare students  3) build features  4) Isolation Forest  5) risk score
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

# Settings you can change
MIN_SIMILARITY = 75     # two students are "similar" if this % of answers are identical
MIN_SAME_WRONG = 8      # ...and they have at least this many identical WRONG answers
HIGH_LIMIT = 70         # risk score >= 70 -> High
MEDIUM_LIMIT = 45       # risk score >= 45 -> Medium


# ---------- Step 1: clean the data ----------
def clean_data(df):
    df.columns = [c.strip() for c in df.columns]

    # find the question columns: q1, q2, q3 ...
    q_cols = []
    for c in df.columns:
        if c.lower().startswith("q") and c[1:].isdigit():
            q_cols.append(c)
    q_cols.sort(key=lambda c: int(c[1:]))

    if "student_id" not in df.columns or "center_id" not in df.columns or len(q_cols) < 10:
        raise ValueError("CSV needs columns student_id, center_id and at least 10 answer columns named q1, q2, ...")

    df = df.dropna(subset=["student_id", "center_id"])      # remove rows with missing id
    df = df.drop_duplicates("student_id")                    # remove duplicate students
    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["center_id"] = df["center_id"].astype(str).str.strip()
    for c in q_cols:
        df[c] = df[c].fillna("-").astype(str).str.strip().str.upper()   # blank answer becomes "-"
    return df.reset_index(drop=True), q_cols


# ---------- Answer key ----------
def find_answer_key(df, q_cols):
    """If no answer key is uploaded, use the most common answer of each question."""
    key = []
    for c in q_cols:
        answers = df[c][df[c] != "-"]
        common = answers.mode()
        key.append(common[0] if len(common) > 0 else "-")
    return key


# ---------- Step 2: compare two students ----------
def compare_students(ans_a, ans_b, key):
    """Returns (similarity %, number of identical WRONG answers)."""
    same = 0
    same_wrong = 0
    for i in range(len(key)):
        if ans_a[i] == ans_b[i] and ans_a[i] != "-":
            same = same + 1
            if ans_a[i] != key[i]:          # both gave the same answer and it is wrong
                same_wrong = same_wrong + 1
    similarity = same / len(key) * 100
    return similarity, same_wrong


def find_similarity_features(df, q_cols, key):
    """Compare every pair of students of the same centre."""
    answers = df[q_cols].values.tolist()      # list of answer lists, one per student
    ids = df["student_id"].tolist()
    n = len(df)

    max_similarity = [0] * n
    max_same_wrong = [0] * n
    n_partners = [0] * n
    pairs = []

    for center in df["center_id"].unique():
        members = df.index[df["center_id"] == center].tolist()   # row numbers of this centre

        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a = members[x]
                b = members[y]
                sim, wrong = compare_students(answers[a], answers[b], key)

                # keep the highest values for both students
                for s in (a, b):
                    if sim > max_similarity[s]:
                        max_similarity[s] = sim
                    if wrong > max_same_wrong[s]:
                        max_same_wrong[s] = wrong

                # suspicious pair?
                if sim >= MIN_SIMILARITY and wrong >= MIN_SAME_WRONG:
                    n_partners[a] = n_partners[a] + 1
                    n_partners[b] = n_partners[b] + 1
                    pairs.append((ids[a], ids[b], center, round(sim, 1), wrong))

    pairs = pd.DataFrame(pairs, columns=["student_a", "student_b", "center_id", "similarity", "same_wrong"])
    return max_similarity, max_same_wrong, n_partners, pairs


# ---------- Step 4 and 5: Isolation Forest -> risk score 0-100 ----------
def calculate_risk(features):
    model = IsolationForest(n_estimators=300, random_state=42)
    model.fit(features.values)
    raw = -model.score_samples(features.values)     # higher = more unusual
    risk = 100 * (raw - raw.min()) / (raw.max() - raw.min() + 0.000001)   # scale to 0-100
    return risk


def get_level(risk):
    if risk >= HIGH_LIMIT:
        return "High"
    if risk >= MEDIUM_LIMIT:
        return "Medium"
    return "Low"


# ---------- Main function ----------
def run_analysis(df, key=None):
    df, q_cols = clean_data(df)
    if key is None:
        key = find_answer_key(df, q_cols)
    key = [str(k).strip().upper() for k in key][:len(q_cols)]

    # score of each student = % of correct answers
    scores = []
    for row in df[q_cols].values.tolist():
        correct = 0
        for i in range(len(key)):
            if row[i] == key[i]:
                correct = correct + 1
        scores.append(correct / len(key) * 100)

    max_sim, max_wrong, partners, pairs = find_similarity_features(df, q_cols, key)

    students = pd.DataFrame({
        "student_id": df["student_id"], "center_id": df["center_id"], "score": scores,
        "max_similarity": max_sim, "max_same_wrong": max_wrong, "n_partners": partners})

    # Step 3: how far is each score above the centre average / overall average
    center_mean = students.groupby("center_id")["score"].transform("mean")
    center_std = students.groupby("center_id")["score"].transform("std").fillna(1).clip(lower=1)
    students["center_z"] = (students["score"] - center_mean) / center_std
    overall_z = (students["score"] - students["score"].mean()) / students["score"].std()

    features = pd.DataFrame({
        "center_z": students["center_z"].clip(lower=0),      # only "too high" is suspicious
        "overall_z": overall_z.clip(lower=0),
        "max_similarity": students["max_similarity"],
        "max_same_wrong": students["max_same_wrong"],
        "n_partners": students["n_partners"]})

    students["risk_score"] = calculate_risk(features).round(1)
    students["risk_level"] = students["risk_score"].apply(get_level)

    # reason text for each student
    reasons = []
    for i in range(len(students)):
        r = students.iloc[i]
        parts = []
        if r["n_partners"] > 0:
            parts.append(f"Answers match {int(r['n_partners'])} other student(s); up to {int(r['max_same_wrong'])} identical wrong answers")
        if r["center_z"] >= 2:
            parts.append(f"Score {r['center_z']:.1f} SD above centre average")
        reasons.append("; ".join(parts) if parts else "No specific indicator")
    students["reasons"] = reasons

    # ----- Centre level analysis -----
    students["flagged"] = (students["n_partners"] > 0).astype(int) * 100   # 100 if flagged, else 0
    centers = students.groupby("center_id").agg(
        n_students=("student_id", "count"),
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        score_std=("score", "std"),
        pct_flagged=("flagged", "mean"),            # = % of students flagged
        mean_similarity=("max_similarity", "mean"),
        mean_wrong=("max_same_wrong", "mean")).reset_index()
    centers["score_std"] = centers["score_std"].fillna(0)

    avg_z = (centers["avg_score"] - centers["avg_score"].mean()) / centers["avg_score"].std()
    center_features = pd.DataFrame({
        "avg_z": avg_z.clip(lower=0), "pct_flagged": centers["pct_flagged"],
        "mean_wrong": centers["mean_wrong"], "mean_similarity": centers["mean_similarity"]})

    centers["risk_score"] = calculate_risk(center_features).round(1)
    centers["risk_level"] = centers["risk_score"].apply(get_level)
    center_reasons = []
    for i in range(len(centers)):
        c = centers.iloc[i]
        center_reasons.append(f"{c['pct_flagged']:.0f}% students have a suspicious match; average score {c['avg_score']:.0f}%")
    centers["reasons"] = center_reasons

    students = students.drop(columns=["flagged"]).round(2)
    centers = centers.drop(columns=["mean_wrong"]).round(2)

    return {"students": students.sort_values("risk_score", ascending=False).reset_index(drop=True),
            "centers": centers.sort_values("risk_score", ascending=False).reset_index(drop=True),
            "pairs": pairs.sort_values(["same_wrong", "similarity"], ascending=False).reset_index(drop=True),
            "n_questions": len(q_cols)}
