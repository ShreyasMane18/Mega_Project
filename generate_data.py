"""Creates a synthetic exam dataset with KNOWN cheating injected, so the system can be evaluated.
Outputs (in data/): exam_data.csv, answer_key.csv, ground_truth.csv
Usage: python generate_data.py [--centers 30] [--per-center 60] [--questions 60] [--seed 42]
"""
import argparse, os
import numpy as np, pandas as pd

def main(n_centers=30, per=60, Q=60, seed=42, out="data"):
    rng = np.random.default_rng(seed)
    opts = np.array(list("ABCD"))
    key = rng.choice(opts, Q)
    difficulty = rng.normal(0, 0.10, Q)
    popular = np.array([rng.choice([o for o in opts if o != k]) for k in key])  # popular wrong option per question

    def solve(ability):
        p = np.clip(ability + difficulty, 0.05, 0.98)
        ans = key.copy()
        for q in range(Q):
            if rng.random() > p[q]:
                wrong = [o for o in opts if o != key[q]]
                ans[q] = popular[q] if rng.random() < 0.4 else rng.choice(wrong)
        return ans

    rows, truth_s = [], {}
    centers = [f"C{str(i+1).zfill(2)}" for i in range(n_centers)]
    leak_centers = set(rng.choice(centers, 2, replace=False))
    copy_centers = list(rng.choice([c for c in centers if c not in leak_centers], 8, replace=False))

    for c in centers:
        ability = np.clip(rng.normal(0.55, 0.16, per), 0.15, 0.92)
        center_shift = rng.normal(0, 0.02)
        answers = [solve(a + center_shift) for a in ability]
        ids = [f"{c}-S{str(i+1).zfill(3)}" for i in range(per)]
        flag = {i: 0 for i in ids}
        if c in leak_centers:  # organised malpractice: many students follow one shared (leaked) sheet
            sheet = key.copy()
            for q in rng.choice(Q, 14, replace=False):
                sheet[q] = rng.choice([o for o in opts if o != key[q]])
            for k in range(per):
                if rng.random() < 0.6:
                    a = answers[k].copy()
                    m = rng.random(Q) < 0.9
                    a[m] = sheet[m]; answers[k] = a; flag[ids[k]] = 1
        if c in copy_centers:  # individual copying: neighbour copies ~85% of source's answers
            for _ in range(2):
                src, dst = rng.choice(per, 2, replace=False)
                a = answers[dst].copy(); m = rng.random(Q) < 0.85
                a[m] = answers[src][m]; answers[dst] = a
                flag[ids[dst]] = 1; flag[ids[src]] = 1
        for k, sid in enumerate(ids):
            rows.append([sid, c, k + 1, *answers[k]]); truth_s[sid] = flag[sid]

    cols = ["student_id", "center_id", "seat_no"] + [f"q{i+1}" for i in range(Q)]
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(f"{out}/exam_data.csv", index=False)
    pd.DataFrame([key], columns=cols[3:]).to_csv(f"{out}/answer_key.csv", index=False)
    gt = pd.DataFrame({"student_id": list(truth_s), "suspicious": list(truth_s.values())})
    gt["center_id"] = gt.student_id.str.split("-").str[0]
    gt["center_suspicious"] = gt.center_id.isin(leak_centers).astype(int)
    gt.to_csv(f"{out}/ground_truth.csv", index=False)
    print(f"{len(rows)} students, {n_centers} centers, {Q} questions | injected suspicious students: {gt.suspicious.sum()} | leak centers: {sorted(leak_centers)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", type=int, default=30); ap.add_argument("--per-center", type=int, default=60)
    ap.add_argument("--questions", type=int, default=60); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); main(a.centers, a.per_center, a.questions, a.seed)
