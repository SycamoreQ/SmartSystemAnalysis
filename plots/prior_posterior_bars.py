# plots/prior_posterior_bars.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

FD="FD001"; OUT=Path("./outputs")
df = pd.read_csv(OUT/f"{FD}_dbn_reliability_weibull.csv")
# Approximate posterior fail prob as 1 - reliability
post_fail = 1.0 - df.groupby("cycle")["reliability"].mean()
prior_fail = 1.0 - (1.0/5.0)  # if prior is uniform over 5 states with one fail state
plt.figure(figsize=(6,4))
plt.bar(["Prior (avg)","Posterior (avg)"], [prior_fail, post_fail.mean()])
plt.ylabel("Failure probability"); plt.title("Prior vs posterior failure probability (aggregated)")
plt.tight_layout(); plt.show()
