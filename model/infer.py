#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import DiscreteFactor # <-- This import is correct

FD = os.environ.get("FD", "FD001")
# ROOT = Path(__file__).resolve().parents[1] # <-- Original
ROOT = Path(".").resolve()                  # <-- FIX
PROC_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_any(pq: Path, csv: Path):
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)

# Features (prefer trainer export, else preprocessing list)
f_models = MODELS_DIR / f"{FD}_features.txt"
f_proc   = PROC_DIR  / f"{FD}_feature_list.txt"
if f_models.exists():
    feature_cols = pd.read_csv(f_models, header=None).iloc[:,0].tolist()
    print(f"[infer] Using feature list: {f_models.name}")
elif f_proc.exists():
    feature_cols = pd.read_csv(f_proc, header=None).iloc[:,0].tolist()
    print(f"[infer] Using feature list from preprocessing: {f_proc.name}")
else:
    raise FileNotFoundError(f"Missing feature list: {f_models} and {f_proc}")

# Monitor model (calibrated)
calibrated_path = MODELS_DIR / f"{FD}_monitor_calibrated.joblib"
if not calibrated_path.exists():
    raise FileNotFoundError(f"Missing calibrated monitor: {calibrated_path}")
monitor = load(calibrated_path)
print(f"[infer] Loaded calibrated monitor: {calibrated_path.name}")

# Data and CPDs
test = read_any(PROC_DIR / f"{FD}_test_preprocessed.parquet",
                PROC_DIR / f"{FD}_test_preprocessed.csv")

# Label mapping consistent with CPT rows
labels_file = MODELS_DIR / f"{FD}_class_labels.txt"
if labels_file.exists():
    labels = pd.read_csv(labels_file, header=None).iloc[:,0].astype(int).tolist()
elif hasattr(monitor, "classes_"):
    labels = list(monitor.classes_)
else:
    labels = list(range(P_y_given_c.shape[0]))
K = len(labels)
print(f"[infer] Found K={K} states: {labels}")

def fix_and_normalize_cpd(M: np.ndarray, name: str, K: int, absorbing_last: bool):
    M = np.array(M, dtype=float)
    if M.shape != (K, K):
         raise ValueError(f"{name} shape {M.shape} != ({K}, {K})")
         
    # If rows look normalized more than columns, transpose so columns are conditional dists
    col_dev = float(np.max(np.abs(M.sum(axis=0) - 1.0)))
    row_dev = float(np.max(np.abs(M.sum(axis=1) - 1.0)))
    if row_dev < (col_dev - 1e-5): # Be robust to floating point
        print(f"{name}: Transposing matrix, row_dev={row_dev:.2e} < col_dev={col_dev:.2e}")
        M = M.T
        
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.clip(M, 0.0, 1.0)
    
    # Repair degenerate evidence columns
    colsum = M.sum(axis=0, keepdims=True)
    bad_cols = np.where((colsum <= 1e-15) | ~np.isfinite(colsum))[1]
    for j in bad_cols:
        M[:, j] = 0.0
        if absorbing_last and j == K - 1:
            M[K - 1, j] = 1.0
        else:
            M[min(j, K - 2), j] = 1.0 # Fallback: stay in same state
            
    # Normalize columns
    colsum = M.sum(axis=0, keepdims=True)
    colsum[colsum == 0.0] = 1.0
    M = M / colsum
    dev = float(np.max(np.abs(M.sum(axis=0) - 1.0)))
    print(f"{name}: shape={M.shape}, max_col_deviation={dev:.2e}")
    return M

# Load and validate CPDs
P_y_given_c = pd.read_csv(MODELS_DIR / f"{FD}_P_y_given_c.csv", header=None).values
P_trans = pd.read_csv(MODELS_DIR / f"{FD}_P_C_next_given_C_weibull.csv", header=None).values
P_y_given_c = fix_and_normalize_cpd(P_y_given_c, "P_y_given_c", K, absorbing_last=False)
P_trans     = fix_and_normalize_cpd(P_trans,     "P_trans",     K, absorbing_last=True) 

# DBN
prior = np.zeros(K); prior[0] = 1.0 # Start in healthy state
model = DBN()
model.add_edges_from([
    (('C', 0), ('Y', 0)),
    (('C', 0), ('C', 1)),
])

cpd_C0 = TabularCPD(('C', 0), K, prior.reshape(-1,1))
cpd_C1 = TabularCPD(('C', 1), K, P_trans, evidence=[('C', 0)], evidence_card=[K])
cpd_Y0 = TabularCPD(('Y', 0), K, P_y_given_c, evidence=[('C', 0)], evidence_card=[K])
model.add_cpds(cpd_C0, cpd_C1, cpy_Y0) # Typo in your script, was cpd_Y0
model.initialize_initial_state()
model.check_model()
print("[infer] DBN model constructed and checked.")

def run_unit(g: pd.DataFrame):
    X_u = g[feature_cols].values
    proba = monitor.predict_proba(X_u) # GBDT output, shape (T, K)
    reliab = []
    inf = DBNInference(model)
    
    for t in range(len(proba)):
        # This is correct: use GBDT output as soft evidence for the
        # *current* time slice in the inference object.
        ev_lik = DiscreteFactor(variables=[('Y', t)], cardinality=[K], values=proba[t])
        post = inf.forward_inference([('C', t)], virtual_evidence=[ev_lik])[('C', t)].values
        p_fail = post[-1] # Probability of being in the last (failure) state
        reliab.append(1.0 - p_fail)
    return np.array(reliab), np.argmax(proba, axis=1)


outs = []
for u, g in test.groupby("unit", sort=True):
    g = g.sort_values("cycle")
    r, y_seq = run_unit(g)
    outs.append(pd.DataFrame({"unit":u,"cycle":g["cycle"].values,"reliability":r,"y_obs":y_seq,"RUL":g["RUL"].values}))
res = pd.concat(outs, ignore_index=True)
out_path = OUT_DIR / f"{FD}_dbn_reliability_weibull.csv"
res.to_csv(out_path, index=False)
print(f"[infer] Saved results to: {out_path}")