"""
Empirical Evaluation & Benchmarking Module for Research Validation.
Compares the AI Forensic Anomaly Detection Model against injected Ground Truth.

Computes:
1. Precision, Recall, F1-Score, Specificity, Accuracy across risk thresholds
2. Confusion Matrix (TP, FP, TN, FN)
3. Multi-Algorithm Benchmarking: Proposed IF vs Standard IF vs One-Class SVM vs LOF vs Baselines
4. AUC-ROC & Average Precision (PR-AUC) metrics with Plotly coordinate arrays
5. Feature Ablation Study (impact of IIR, Z-score, Aberrance, Collusion Partners)
6. Center-Level Question Leak Detection Accuracy
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import analysis
import config


def compute_metrics(y_true, y_pred):
    """Calculates classification metrics from binary arrays."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "specificity": round(float(specificity), 4),
        "accuracy": round(float(accuracy), 4),
    }


def _subsample_curve(x_arr, y_arr, max_pts=60):
    """Subsamples coordinate arrays for smooth and lightweight client-side Plotly rendering."""
    n = len(x_arr)
    if n <= max_pts:
        return [round(float(v), 4) for v in x_arr], [round(float(v), 4) for v in y_arr]
    indices = np.linspace(0, n - 1, max_pts, dtype=int)
    return [round(float(x_arr[i]), 4) for i in indices], [round(float(y_arr[i]), 4) for i in indices]


def evaluate_dataset(exam_csv=config.SAMPLE_EXAM_CSV, key_csv=config.SAMPLE_KEY_CSV, gt_csv=config.GROUND_TRUTH_CSV):
    """
    Runs full scientific benchmark comparing model output against ground truth.
    Returns structured results for terminal display, paper reporting, and web UI rendering.
    """
    if not (os.path.exists(exam_csv) and os.path.exists(gt_csv)):
        raise FileNotFoundError(
            f"Evaluation requires '{exam_csv}' and '{gt_csv}'. Run generate_data.py first."
        )

    exam_df = pd.read_csv(exam_csv)
    key = None
    if os.path.exists(key_csv):
        key_df = pd.read_csv(key_csv)
        key = key_df.iloc[0].tolist()

    gt_df = pd.read_csv(gt_csv)
    result = analysis.run_analysis(exam_df, key)
    students = result["students"].merge(gt_df[["student_id", "suspicious"]], on="student_id", how="left")
    students["suspicious"] = students["suspicious"].fillna(0).astype(int)

    y_true = students["suspicious"].values
    n_q = result["n_questions"]

    # 1. Standard Threshold Metrics
    pred_high = (students["risk_level"] == "High").astype(int).values
    metrics_high = compute_metrics(y_true, pred_high)

    pred_med_high = (students["risk_level"] != "Low").astype(int).values
    metrics_med_high = compute_metrics(y_true, pred_med_high)

    pred_sim_base = (students["max_similarity"] >= 85.0).astype(int).values
    metrics_sim = compute_metrics(y_true, pred_sim_base)

    pred_z_base = (students["center_z"] >= 2.0).astype(int).values
    metrics_z = compute_metrics(y_true, pred_z_base)

    # 2. Multi-Model Comparative Benchmarking
    pos_c_z = students["center_z"].clip(lower=0.0).values
    pos_o_z = students["overall_z"].clip(lower=0.0).values
    sim_norm = (students["max_similarity"] / 100.0).values
    wrong_norm = (students["max_same_wrong"] / max(n_q, 1)).values
    part_norm = (students["n_partners"] / 5.0).clip(upper=2.0).values
    aberr_norm = (students.get("aberrance_score", pd.Series(0, index=students.index)) / 100.0).values

    X_full = np.column_stack([pos_c_z, pos_o_z, sim_norm, wrong_norm, part_norm, aberr_norm])

    # Model 1: Proposed Dual-Stage Isolation Forest (Calibrated)
    proposed_scores = students["risk_score"].values / 100.0

    # Model 2: Standard Isolation Forest (Uncalibrated)
    if_std = IsolationForest(n_estimators=config.IF_ESTIMATORS, contamination="auto", random_state=config.IF_RANDOM_STATE)
    if_std.fit(X_full)
    raw_if = -if_std.score_samples(X_full)
    std_if_scores = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min() + 1e-9)

    # Model 3: One-Class SVM (RBF Kernel)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_full)
    raw_ocsvm = -ocsvm.decision_function(X_full)
    ocsvm_scores = (raw_ocsvm - raw_ocsvm.min()) / (raw_ocsvm.max() - raw_ocsvm.min() + 1e-9)

    # Model 4: Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=25, novelty=True)
    lof.fit(X_full)
    raw_lof = -lof.decision_function(X_full)
    lof_scores = (raw_lof - raw_lof.min()) / (raw_lof.max() - raw_lof.min() + 1e-9)

    # Baselines
    sim_scores = students["max_similarity"].values / 100.0
    z_scores = np.clip(students["center_z"].values / 3.0, 0.0, 1.0)

    model_dict = {
        "Proposed Dual-Stage IF": proposed_scores,
        "Standard Isolation Forest": std_if_scores,
        "One-Class SVM (RBF)": ocsvm_scores,
        "Local Outlier Factor (LOF)": lof_scores,
        "Baseline: Response Similarity": sim_scores,
        "Baseline: Center Z-Score": z_scores,
    }

    benchmark_table = []
    roc_curves_data = {}
    pr_curves_data = {}

    for name, s_arr in model_dict.items():
        fpr, tpr, _ = roc_curve(y_true, s_arr)
        roc_auc_val = float(auc(fpr, tpr))
        ap_val = float(average_precision_score(y_true, s_arr))

        p_c, r_c, ths = precision_recall_curve(y_true, s_arr)
        f1_c = (2 * p_c * r_c) / np.maximum(p_c + r_c, 1e-9)
        best_idx = int(np.argmax(f1_c))

        benchmark_table.append({
            "model_name": name,
            "auc_roc": round(roc_auc_val, 4),
            "avg_precision": round(ap_val, 4),
            "max_f1": round(float(f1_c[best_idx]), 4),
            "precision_at_f1": round(float(p_c[best_idx]), 4),
            "recall_at_f1": round(float(r_c[best_idx]), 4),
        })

        sub_fpr, sub_tpr = _subsample_curve(fpr, tpr, max_pts=50)
        sub_rec, sub_prec = _subsample_curve(r_c, p_c, max_pts=50)

        roc_curves_data[name] = {"fpr": sub_fpr, "tpr": sub_tpr, "auc": round(roc_auc_val, 3)}
        pr_curves_data[name] = {"recall": sub_rec, "precision": sub_prec, "ap": round(ap_val, 3)}

    # 3. Systematic Feature Ablation Study
    ablation_sets = {
        "Full Feature Set (Proposed)": X_full,
        "w/o Identical Wrong (No IIR)": np.column_stack([pos_c_z, pos_o_z, sim_norm, part_norm, aberr_norm]),
        "w/o Center Z-Score": np.column_stack([pos_o_z, sim_norm, wrong_norm, part_norm, aberr_norm]),
        "w/o Collusion Partners": np.column_stack([pos_c_z, pos_o_z, sim_norm, wrong_norm, aberr_norm]),
        "w/o Item Aberrance Index": np.column_stack([pos_c_z, pos_o_z, sim_norm, wrong_norm, part_norm]),
        "Pure Similarity Only": sim_norm.reshape(-1, 1),
    }

    ablation_table = []
    for ab_name, feats in ablation_sets.items():
        m_ab = IsolationForest(n_estimators=config.IF_ESTIMATORS, contamination="auto", random_state=config.IF_RANDOM_STATE)
        m_ab.fit(feats)
        r_ab = -m_ab.score_samples(feats)
        s_ab = (r_ab - r_ab.min()) / (r_ab.max() - r_ab.min() + 1e-9)

        fpr_ab, tpr_ab, _ = roc_curve(y_true, s_ab)
        auc_ab = float(auc(fpr_ab, tpr_ab))
        ap_ab = float(average_precision_score(y_true, s_ab))

        p_ab, r_ab_c, _ = precision_recall_curve(y_true, s_ab)
        f1_ab = (2 * p_ab * r_ab_c) / np.maximum(p_ab + r_ab_c, 1e-9)
        max_f1_ab = float(np.max(f1_ab))

        ablation_table.append({
            "feature_set": ab_name,
            "auc_roc": round(auc_ab, 4),
            "avg_precision": round(ap_ab, 4),
            "max_f1": round(max_f1_ab, 4),
        })

    # 4. Center-Level Leak Detection
    centers = result["centers"]
    center_metrics = {}
    if "center_suspicious" in gt_df.columns:
        center_gt = gt_df.groupby("center_id")["center_suspicious"].max().reset_index()
        c_merged = centers.merge(center_gt, on="center_id", how="left").fillna(0)
        c_true = c_merged["center_suspicious"].values
        c_pred = (c_merged["risk_level"] != "Low").astype(int).values
        c_met = compute_metrics(c_true, c_pred)
        leak_centers = c_merged[c_merged["center_suspicious"] == 1]["center_id"].tolist()
        flagged_centers = c_merged[c_merged["risk_level"] != "Low"]["center_id"].tolist()
        center_metrics = {
            "metrics": c_met,
            "leak_centers": leak_centers,
            "detected_centers": [c for c in leak_centers if c in flagged_centers],
            "total_leak_centers": len(leak_centers),
        }

    return {
        "total_examinees": len(students),
        "injected_cheaters": int(np.sum(y_true)),
        "metrics_high": metrics_high,
        "metrics_med_high": metrics_med_high,
        "baseline_similarity": metrics_sim,
        "baseline_zscore": metrics_z,
        "center_analysis": center_metrics,
        "benchmark_table": benchmark_table,
        "ablation_table": ablation_table,
        "roc_curves_data": roc_curves_data,
        "pr_curves_data": pr_curves_data,
    }


def main():
    print("==================================================================")
    print("   AI-Based Forensic Examination Analysis: Research Benchmark     ")
    print("==================================================================")
    results = evaluate_dataset()

    print(f"\nExaminees Evaluated: {results['total_examinees']:,}")
    print(f"Ground-Truth Irregular Examinees Injected: {results['injected_cheaters']}")

    print("\n[Evaluation Metric Comparison across Detection Thresholds]")
    print("------------------------------------------------------------------")
    print(f"{'Method / Threshold':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Accuracy':<10}")
    print("------------------------------------------------------------------")

    mh = results["metrics_high"]
    print(f"{'Proposed Model (High Risk)':<30} | {mh['precision']:<10.2f} | {mh['recall']:<10.2f} | {mh['f1']:<10.2f} | {mh['accuracy']:<10.2f}")

    mm = results["metrics_med_high"]
    print(f"{'Proposed Model (Med+High)':<30} | {mm['precision']:<10.2f} | {mm['recall']:<10.2f} | {mm['f1']:<10.2f} | {mm['accuracy']:<10.2f}")

    bs = results["baseline_similarity"]
    print(f"{'Baseline 1: Similarity Only':<30} | {bs['precision']:<10.2f} | {bs['recall']:<10.2f} | {bs['f1']:<10.2f} | {bs['accuracy']:<10.2f}")

    bz = results["baseline_zscore"]
    print(f"{'Baseline 2: Score Outlier Only':<30} | {bz['precision']:<10.2f} | {bz['recall']:<10.2f} | {bz['f1']:<10.2f} | {bz['accuracy']:<10.2f}")
    print("------------------------------------------------------------------")

    print("\n[Multi-Model Benchmark Comparison (AUC-ROC & Average Precision)]")
    print("------------------------------------------------------------------")
    print(f"{'Architecture':<32} | {'AUC-ROC':<8} | {'PR-AUC':<8} | {'Max F1':<8}")
    print("------------------------------------------------------------------")
    for row in results["benchmark_table"]:
        print(f"{row['model_name']:<32} | {row['auc_roc']:<8.4f} | {row['avg_precision']:<8.4f} | {row['max_f1']:<8.4f}")
    print("------------------------------------------------------------------")

    print("\n[Feature Ablation Study]")
    print("------------------------------------------------------------------")
    print(f"{'Ablation Configuration':<32} | {'AUC-ROC':<8} | {'PR-AUC':<8} | {'Max F1':<8}")
    print("------------------------------------------------------------------")
    for row in results["ablation_table"]:
        print(f"{row['feature_set']:<32} | {row['auc_roc']:<8.4f} | {row['avg_precision']:<8.4f} | {row['max_f1']:<8.4f}")
    print("------------------------------------------------------------------")

    ca = results["center_analysis"]
    if ca:
        print("\n[Examination Center Malpractice Analysis]")
        print(f"Actual Question-Leak Centers: {ca['leak_centers']}")
        print(f"Detected Collusion Centers:   {ca['detected_centers']}")
        cm = ca["metrics"]
        print(f"Center Detection: Precision={cm['precision']:.2f}, Recall={cm['recall']:.2f}, F1={cm['f1']:.2f}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
