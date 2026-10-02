"""Measures detection quality against the synthetic ground truth: python evaluate.py"""
import pandas as pd
import analysis

df = pd.read_csv("data/exam_data.csv"); key = pd.read_csv("data/answer_key.csv").iloc[0].tolist()
gt = pd.read_csv("data/ground_truth.csv")
res = analysis.run_analysis(df, key)
S = res["students"].merge(gt, on="student_id")
for name, mask in [("High only", S.risk_level == "High"), ("Medium+High", S.risk_level != "Low")]:
    tp = ((mask) & (S.suspicious == 1)).sum(); fp = ((mask) & (S.suspicious == 0)).sum(); fn = ((~mask) & (S.suspicious == 1)).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"Students [{name}]: precision={p:.2f} recall={r:.2f} F1={2*p*r/max(p+r,1e-9):.2f} (TP={tp} FP={fp} FN={fn})")
cg = gt.groupby("center_id").center_suspicious.max().reset_index()
C = res["centers"].merge(cg, on="center_id")
print("Truly suspicious centres:", C[C.center_suspicious == 1].center_id.tolist())
print("Top 5 flagged centres  :", C.head(5)[["center_id", "risk_score", "risk_level"]].values.tolist())
