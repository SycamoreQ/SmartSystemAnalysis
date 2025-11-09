import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import weibull_min 

# --- Configuration ---
FD = os.environ.get("FD", "FD001")
ROOT = Path(".").resolve()
PROC_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Tunable parameters for the transition model
AGE_STEP_PER_STATE = float(os.environ.get("DBN_AGE_STEP", "35.0"))
STAY_PROB_FRAC = float(os.environ.get("DBN_STAY_FRAC", "0.90"))
DT = float(os.environ.get("DBN_DT", "1.0"))
EPS = 1e-9 # Smoothing epsilon

def read_any(pq: Path, csv: Path) -> pd.DataFrame:
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)

# 1) Determine K (state count)
labels_path = MODELS_DIR / f"{FD}_class_labels.txt"
if not labels_path.exists():
    raise FileNotFoundError(f"Missing labels file: {labels_path}. Run 1_train_monitor.py first.")
labels = pd.read_csv(labels_path, header=None).iloc[:,0].tolist()
K = len(labels)
print(f"[weibull_trans] K={K} states found: {labels}")

# 2) Load training TTFs and fit Weibull using scipy
train = read_any(PROC_DIR / f"{FD}_train_preprocessed.parquet",
                 PROC_DIR / f"{FD}_train_preprocessed.csv")
                 
# Time-to-Failure (TTF) is max(cycle) for each unit
ttf = train.groupby('unit')['cycle'].max().values
ttf_pos = ttf[ttf > 0]

if len(ttf_pos) < 3:
    raise ValueError("Not enough positive TTF values to fit Weibull.")
    
print(f"Fitting Weibull on {len(ttf_pos)} failure times (TTF) using scipy...")
# Fit the 3-parameter Weibull (loc, scale, shape)
# The 2-parameter Weibull assumes loc=0
shape_c, loc, scale = weibull_min.fit(ttf_pos, floc=0)

# Map scipy's parameters to reliability's terminology:
alpha = shape_c # shape parameter (usually 'c' or 'k' in literature)
beta = scale    # scale parameter (usually 'eta' or 'lambda')
# NOTE: loc is kept at 0 (2-parameter model)

print(f"[weibull_trans] Fitted Weibull: alpha={alpha:.4f} (shape), beta={beta:.4f} (scale)")
# Expected output is now stable, e.g., alpha ~ 4.0-5.5, beta ~ 220-250


# 3) Build a KxK column-stochastic transition matrix P(C_{t+1}|C_t)
def weibull_transition_matrix(n_states, dt, alpha, beta, age_step, stay_frac):
    P = np.zeros((n_states, n_states))
    
    for s_curr in range(n_states):
        # State K-1 is the absorbing failure state
        if s_curr == n_states - 1:
            P[s_curr, s_curr] = 1.0
            continue
            
        t = s_curr * age_step 
        
        # Calculate survival probabilities: S(t) = exp(-((t/beta)^alpha))
        S_t   = np.exp(-((t / beta) ** alpha))
        S_tdt = np.exp(-(((t + dt) / beta) ** alpha))
        
        # Discrete hazard rate: P(Fail in [t, t+dt] | Survived to t)
        # Use a maximum function to prevent domain error if S_t is near zero
        p_fail_in_step = 1.0 - (S_tdt / (max(S_t, EPS))) 
        p_fail_in_step = float(np.clip(p_fail_in_step, 0.0, 1.0))
        
        p_survive_step = 1.0 - p_fail_in_step
        
        p_stay = stay_frac * p_survive_step
        p_next = (1.0 - stay_frac) * p_survive_step
        
        # Distribute probabilities
        P[s_curr, s_curr] = p_stay
        s_next = min(s_curr + 1, n_states - 1)
        P[s_next, s_curr] += p_next
        P[n_states - 1, s_curr] += p_fail_in_step # All failure probability goes to the last state

    # Final post-processing (Normalization and final state check)
    col_sums = P.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    P = P / col_sums
    P = np.nan_to_num(P, nan=0.0)
    P[:, n_states-1] = 0.0 # Clear non-absorbing transitions from last state
    P[n_states-1, n_states-1] = 1.0 # Ensure last state is perfectly absorbing
    
    return P

P_trans = weibull_transition_matrix(K, DT, alpha, beta, AGE_STEP_PER_STATE, STAY_PROB_FRAC)

# 4) Save CPD
cpd_path  = MODELS_DIR / f"{FD}_P_C_next_given_C_weibull.csv"
meta_path = MODELS_DIR / f"{FD}_weibull_meta.json"
pd.DataFrame(P_trans).to_csv(cpd_path, index=False, header=None)
pd.Series({
    "alpha": alpha, "beta": beta, "K": K,
    "dt": DT, "age_step": AGE_STEP_PER_STATE, "stay_frac": STAY_PROB_FRAC
}).to_json(meta_path, indent=2)

print(f"[weibull_trans] Saved {cpd_path.name} (shape {P_trans.shape})")
print("Transition Matrix P(C'|C):\n", pd.DataFrame(P_trans).round(3))