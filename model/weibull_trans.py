import numpy as np
import pandas as pd
from pathlib import Path
from reliability.Fitters import Fit_Weibull_2P

PROC_DIR = Path("./processed")
FD = "FD001"

#Load train labels for fitting Weibull on RUL
train = pd.read_parquet(PROC_DIR / f"{FD}_train_preprocessed.parquet") if (PROC_DIR / f"{FD}_train_preprocessed.parquet").exists() \
        else pd.read_csv(PROC_DIR / f"{FD}_train_preprocessed.csv")
rul_vals = train["RUL"].values.astype(float)
fit = Fit_Weibull_2P(failures=rul_vals, show_plot=False, print_results=False)
alpha = float(fit.alpha)  # shape
beta = float(fit.beta)    # scale

def weibull_transition_matrix(n_states=5, dt=1.0, alpha=alpha, beta=beta, k=10.0, stay_frac=0.85):
    """
    Build P(C_{t+1}|C_t) using Weibull interval hazards evaluated at a proxy age t = s*k.
    - n_states-1 is absorbing fail state.
    - For states 0..n-2, column sums to 1.
    """
    P = np.zeros((n_states, n_states))
    for s in range(n_states):
        if s == n_states - 1:
            P[s, s] = 1.0
            continue
        t = s * k
        S_t = np.exp(-((t / beta) ** alpha))
        S_tdt = np.exp(-(((t + dt) / beta) ** alpha))
        p_fail = 1.0 - (S_tdt / S_t)
        p_fail = float(np.clip(p_fail, 0.0, 1.0))
        p_stay = stay_frac * (1 - p_fail)
        p_next = 1.0 - p_fail - p_stay
        p_next = float(max(0.0, p_next))
        # assign
        P[s, s] = p_stay
        P[s + 1, s] = p_next
        P[n_states - 1, s] += max(0.0, 1.0 - (p_stay + p_next))  # ensure column sums to 1
    # normalize columns exactly
    col_sum = P.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    P = P / col_sum
    return P

P_weibull = weibull_transition_matrix()
pd.DataFrame(P_weibull).to_csv("./models/FD001_P_C_next_given_C_weibull.csv", index=False, header=False)
print("Saved ./models/FD001_P_C_next_given_C_weibull.csv")
