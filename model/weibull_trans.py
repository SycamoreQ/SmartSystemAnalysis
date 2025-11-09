#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from reliability.Fitters import Fit_Weibull_2P  # pip install reliability

FD = os.environ.get("FD", "FD001")
# ROOT = Path(__file__).resolve().parents[1] # <-- Original
ROOT = Path(".").resolve()                  # <-- FIX
PROC_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def read_any(pq: Path, csv: Path) -> pd.DataFrame:
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)

# 1) Determine K (state count) from the monitoring CPT
pyc_path = MODELS_DIR / f"{FD}_P_y_given_c.csv"
if not pyc_path.exists():
    raise FileNotFoundError(f"Missing monitoring CPT: {pyc_path}. Run NASA.py first.")
P_y_given_c = pd.read_csv(pyc_path, header=None).values
K = int(P_y_given_c.shape[0])
print(f"[weibull_trans] K={K} states read from {pyc_path.name}")

# 2) Load training RULs and fit Weibull on positive values.
train = read_any(PROC_DIR / f"{FD}_train_preprocessed.parquet",
                 PROC_DIR / f"{FD}_train_preprocessed.csv")
rul = train["RUL"].astype(float).values
rul_pos = rul[rul > 0]
if len(rul_pos) < 3:
    raise ValueError("Not enough positive RUL values to fit Weibull.")
fit = Fit_Weibull_2P(failures=rul_pos, show_probability_plot=False, print_results=False)
alpha = float(fit.alpha); beta = float(fit.beta)
print(f"[weibull_trans] Fitted Weibull alpha={alpha:.4f} beta={beta:.4f}")

# 3) Build a KxK column-stochastic transition matrix P(C_{t+1}|C_t).
def weibull_transition_matrix(n_states, dt, alpha, beta, age_step, stay_frac):
    P = np.zeros((n_states, n_states))
    EPS = 1e-6 # Add smoothing
    
    for s in range(n_states):
        if s == n_states - 1: # Last state is absorbing (failure)
            P[s, s] = 1.0
            continue
            
        t = s * age_step # Map state 's' to an equivalent "age" t
        
        # Calculate discrete hazard rate (prob of failure in [t, t+dt])
        S_t   = np.exp(-((t      / beta) ** alpha))
        S_tdt = np.exp(-(((t+dt) / beta) ** alpha))
        p_fail = 1.0 - (S_tdt / (S_t + 1e-15)) # Add epsilon for stability
        p_fail = float(np.clip(p_fail, 0.0, 1.0))
        
        p_stay = stay_frac * (1.0 - p_fail)
        p_next = max(0.0, 1.0 - p_fail - p_stay)
        
        P[s, s] = p_stay
        P[min(s+1, n_states-1), s] = p_next # Move to next state
        P[n_states-1, s] += max(0.0, 1.0 - (p_stay + p_next)) # Remainder goes to failure
    
    # Epsilon smoothing + column renorm
    P += EPS
    P /= (P.sum(axis=0, keepdims=True) + 1e-12)
    return P

DT       = float(os.environ.get("DBN_DT", "1.0"))
AGE_STEP = float(os.environ.get("DBN_AGE_STEP", "10.0"))
STAY_FR  = float(os.environ.get("DBN_STAY_FRAC", "0.85"))

P_trans = weibull_transition_matrix(K, DT, alpha, beta, AGE_STEP, STAY_FR)

# 4) Save CPD + metadata
cpd_path  = MODELS_DIR / f"{FD}_P_C_next_given_C_weibull.csv"
meta_path = MODELS_DIR / f"{FD}_weibull_meta.json"
pd.DataFrame(P_trans).to_csv(cpd_path, index=False, header=None)
pd.Series({
    "alpha": alpha, "beta": beta, "K": K,
    "dt": DT, "age_step": AGE_STEP, "stay_frac": STAY_FR
}).to_json(meta_path, indent=2)
print(f"[weibull_trans] Saved {cpd_path} (shape {P_trans.shape})")