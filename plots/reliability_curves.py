#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FD = "FD001"
OUT = Path("./outputs")
wb  = pd.read_csv(OUT / f"{FD}_dbn_reliability_weibull.csv")

g = wb.groupby("cycle")["reliability"]
mean = g.mean(); p10 = g.quantile(0.10); p90 = g.quantile(0.90)

plt.figure(figsize=(8,5))
plt.plot(mean.index, mean.values, label="Weibull CPD")
plt.fill_between(mean.index, p10.values, p90.values, alpha=0.2)
plt.xlabel("Cycle"); plt.ylabel("Reliability R(t)")
plt.title(f"{FD} DBN Reliability: Weibull only")
plt.legend(); plt.tight_layout(); plt.show()
