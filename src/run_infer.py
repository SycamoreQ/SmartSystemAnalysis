import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import DBNInference

# --- Configuration ---
FD = os.environ.get("FD", "FD001")
ROOT = Path(".").resolve()
PROC_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_any(pq: Path, csv: Path):
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)

# --- Load Artifacts ---
print("[infer] Loading all artifacts...")

# 1. Data & Features
test = read_any(PROC_DIR / f"{FD}_test_preprocessed.parquet",
                PROC_DIR / f"{FD}_test_preprocessed.csv")
feature_cols = pd.read_csv(MODELS_DIR / f"{FD}_features.txt", header=None).iloc[:,0].tolist()

# 2. Labels (Keep as integers 0, 1, 2, 3, 4 for indexing)
labels = pd.read_csv(MODELS_DIR / f"{FD}_class_labels.txt", header=None).iloc[:,0].tolist()
K = len(labels)

# CRITICAL FIX: Define the required string state names for soft evidence factors
STRING_STATE_NAMES = [str(l) for l in labels] # ['0', '1', '2', '3', '4']
print(f"[infer] K={K} states: {labels}")

# 3. Monitor
calibrated_path = MODELS_DIR / f"{FD}_monitor_calibrated.joblib"
monitor = load(calibrated_path)
print(f"[infer] Loaded calibrated monitor: {calibrated_path.name}")

# 4. CPTs
P_y_given_c = pd.read_csv(MODELS_DIR / f"{FD}_P_y_given_c.csv", header=None).values
P_trans = pd.read_csv(MODELS_DIR / f"{FD}_P_C_next_given_C_weibull.csv", header=None).values

# --- CPD Validation & Normalization ---
def fix_and_normalize_cpd(M: np.ndarray, name: str, shape: tuple):
    if M.shape != shape:
        raise ValueError(f"{name} shape is {M.shape}, expected {shape}")
    
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.clip(M, 1e-9, 1.0) 
    
    col_sums = M.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    M = M / col_sums
    
    dev = float(np.max(np.abs(M.sum(axis=0) - 1.0)))
    print(f"[infer] {name}: shape={M.shape}, max_col_deviation={dev:.2e}")
    return M

P_y_given_c = fix_and_normalize_cpd(P_y_given_c, "P_y_given_c", (K, K))
P_trans     = fix_and_normalize_cpd(P_trans,     "P_trans",     (K, K))

# --- Build DBN Model ---
print("[infer] Building DBN...")
model = DBN()
# C: Hidden State, Y: Observed State
model.add_edges_from([
    (('C', 0), ('Y', 0)),  # Emission: C_t -> Y_t
    (('C', 0), ('C', 1)),  # Transition: C_t -> C_t+1
])

# Define Prior: P(C_0). Start in the healthiest state (State 0)
prior = np.zeros(K); prior[0] = 1.0

# Define CPDs: Use default integer state names for stability
cpd_C0 = TabularCPD(('C', 0), K, prior.reshape(-1, 1))
cpd_C1 = TabularCPD(('C', 1), K, P_trans, 
                    evidence=[('C', 0)], evidence_card=[K])
cpd_Y0 = TabularCPD(('Y', 0), K, P_y_given_c, 
                    evidence=[('C', 0)], evidence_card=[K])

model.add_cpds(cpd_C0, cpd_C1, cpd_Y0)
model.initialize_initial_state()
model.check_model()
print("[infer] DBN model constructed and checked.")

# --- Inference Function ---
def run_inference_on_unit(g: pd.DataFrame):
    X_u = g[feature_cols].values
    proba_y = monitor.predict_proba(X_u)
    y_hard = np.argmax(proba_y, axis=1).astype(int)

    reliab = []
    inf = DBNInference(model)
    for t, y in enumerate(y_hard):
        post_c = inf.forward_inference([('C', t)], evidence={(('Y', t)): int(y)})[('C', t)].values
        reliab.append(1.0 - post_c[-1])  # last index = fail
    return np.array(reliab), y_hard


# --- Run Inference on All Test Units ---
print("[infer] Running inference on all test units...")
outs = []
for u, g in test.groupby("unit", sort=True):
    g = g.sort_values("cycle")
    # r = reliability sequence, y_seq = GBDT observed state sequence
    r, y_seq = run_inference_on_unit(g)
    
    unit_df = pd.DataFrame({
        "unit": u,
        "cycle": g["cycle"].values,
        "reliability": r,
        "y_obs": y_seq,
        "RUL_true": g["RUL"].values,
        "state_true": g["state"].values
    })
    outs.append(unit_df)
    if u % 10 == 0:
        print(f"  ... completed unit {u}")

res = pd.concat(outs, ignore_index=True)
out_path = OUT_DIR / f"{FD}_dbn_reliability.csv"
res.to_csv(out_path, index=False)
print(f"\n[infer] Success! Saved results to: {out_path}")