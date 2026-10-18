"""Create a small MCQ exam CSV with a few planted copying cases (for demo + paper)."""

import os
import random

random.seed(42)
os.makedirs("data", exist_ok=True)

N_QUESTIONS = 20
OPTIONS = "ABCD"
CENTERS = [f"C{i}" for i in range(1, 9)]
N = 160


def random_answers():
    return "".join(random.choice(OPTIONS) for _ in range(N_QUESTIONS))


def score_of(answers, key):
    return sum(1 for i in range(N_QUESTIONS) if answers[i] == key[i]) * 5  # out of 100


key = random_answers()
rows = ["student_id,center_id,score,answers"]

students = []
for i in range(1, N + 1):
    sid = f"S{i:03d}"
    center = random.choice(CENTERS)
    ans = random_answers()
    students.append([sid, center, ans])

# Copying pair in same center (almost identical answers)
students[0][1] = "C1"
students[1][1] = "C1"
students[1][2] = students[0][2][:18] + random_answers()[18:]

# Copying ring of 3 in C2
base = random_answers()
for idx in (10, 11, 12):
    students[idx][1] = "C2"
    students[idx][2] = base[:17] + random_answers()[17:]

# High scorer with odd pattern (possible preknowledge): matches key, unlike peers
students[20][1] = "C3"
students[20][2] = key

for sid, center, ans in students:
    sc = score_of(ans, key)
    rows.append(f"{sid},{center},{sc},{ans}")

path = "data/sample_exam.csv"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
print("Wrote", path, "students", N)
