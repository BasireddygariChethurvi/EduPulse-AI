"""Generate a reproducible synthetic student performance dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "student_performance.csv"
OUT.parent.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n = 1200
study_hours = np.clip(rng.gamma(shape=2.2, scale=1.25, size=n), 0, 10)
attendance = np.clip(rng.normal(78, 13, n), 25, 100)
previous = np.clip(rng.normal(63, 17, n), 5, 100)
assignments = np.clip(rng.normal(76, 19, n), 0, 100)
sleep = np.clip(rng.normal(7.1, 1.15, n), 3, 11)
internet = rng.choice(["Yes", "No"], size=n, p=[0.86, 0.14])
extra = rng.choice(["Yes", "No"], size=n, p=[0.58, 0.42])

internet_effect = np.where(internet == "Yes", 2.5, -2.5)
extra_effect = np.where(extra == "Yes", 1.2, 0)
noise = rng.normal(0, 8, n)
score = (
    10 + study_hours * 3.3 + attendance * 0.19 + previous * 0.31
    + assignments * 0.12 + (sleep - 7) * 1.2 + internet_effect + extra_effect + noise
)
score = np.clip(score, 0, 100).round(1)
status = np.where(score >= 50, "Pass", "Needs Support")

df = pd.DataFrame({
    "study_hours": study_hours.round(1),
    "attendance_pct": attendance.round(1),
    "previous_score": previous.round(1),
    "assignments_completed": assignments.round(1),
    "sleep_hours": sleep.round(1),
    "internet_access": internet,
    "extracurricular": extra,
    "final_score": score,
    "pass_status": status,
})
df.to_csv(OUT, index=False)
print(f"Created {OUT} with {len(df)} records.")
