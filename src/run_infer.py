import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import DBNInference

# Configuration 
FD = os.environ.get("FD", "FD001")
ROOT = Path(".").resolve()
PROC_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define prediction horizon 
MAX_PRED_CYCLES = 250 

def read_any(pq: Path, csv: Path):
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)

print("[infer] Loading all artifacts...")
test = read_any(PROC_DIR / f"{FD}_test_preprocessed.parquet",
                PROC_DIR / f"{FD}_test_preprocessed.csv")
feature_cols = pd.read_csv(MODELS_DIR / f"{FD}_features.txt", header=None).iloc[:,0].tolist() 

labels = pd.read_csv(MODELS_DIR / f"{FD}_class_labels.txt", header=None).iloc[:,0].tolist()
K = len(labels)

STRING_STATE_NAMES = [str(l) for l in labels] 
print(f"[infer] K={K} states: {labels}")

# Load GBDT monitor
calibrated_path = MODELS_DIR / f"{FD}_monitor_calibrated.joblib"
monitor = load(calibrated_path)
print(f"[infer] Monitor loaded: {calibrated_path.name}")

# Load CPTs
P_y_given_c = pd.read_csv(MODELS_DIR / f"{FD}_P_y_given_c.csv", header=None).values
P_trans     = pd.read_csv(MODELS_DIR / f"{FD}_P_C_next_given_C_weibull.csv", header=None).values

# DBN Construction
model = DBN([(('C', 0), ('C', 1)), (('C', 0), ('Y', 0))])
prior = np.array([1.0] + [0.0]*(K-1)) # Initial state: 100% in State 0

assert P_y_given_c.shape == (K, K), f"P(Y|C) must be {K}x{K}, got {P_y_given_c.shape}"
assert P_trans.shape     == (K, K), f"P(C'|C) must be {K}x{K}, got {P_trans.shape}"

cpd_C0 = TabularCPD(('C', 0), K, prior.reshape(-1,1))
cpd_C1 = TabularCPD(('C', 1), K, P_trans, evidence=[('C', 0)], evidence_card=[K])
cpd_Y0 = TabularCPD(('Y', 0), K, P_y_given_c, evidence=[('C', 0)], evidence_card=[K])

model.add_cpds(cpd_C0, cpd_C1, cpd_Y0)
model.initialize_initial_state()
model.check_model()
print("[infer] DBN model constructed and checked.")


def run_inference_and_prediction_on_unit(g: pd.DataFrame):
    """
    Runs DBN inference in two phases: Filtering (data available) and Prediction (extrapolation).
    """
    unit_id = g['unit'].iloc[0]
    X_u = g[feature_cols].values
    T_max = len(g)
    
    # GBDT soft evidence P(Y|X) and hard evidence (Y_obs)
    proba_obs = monitor.predict_proba(X_u)
    y_hard = np.argmax(proba_obs, axis=1)
    
    #Phase 1 : FILTERING (t <= T_max)
    posteriors = [] # To store P(C_t | Y_1:t)
    inf = DBNInference(model) 
    current_post = np.array(prior)

    # Filtering loop using hard evidence
    for t in range(T_max):
        post = inf.forward_inference([('C', t)], evidence={(('Y', t)): int(y_hard[t])})[('C', t)].values
        current_post = post.flatten()
        posteriors.append(current_post)
        
    last_post = current_post if posteriors else prior
    
    #Phase 2: PREDICTION (t > T_max) ---
    # We extrapolate the blue line using only the Transition CPT (P_trans).
    
    predicted_posteriors = []
    current_pred_post = last_post

    for t in range(T_max, MAX_PRED_CYCLES):
        # P(C_next) = P(C_next | C_current) * P(C_current)
        P_next = P_trans @ current_pred_post  
        
        predicted_posteriors.append(P_next)
        current_pred_post = P_next
        
        if (1.0 - P_next[K-1]) < 0.001:
            break
            
    all_posteriors = np.array(posteriors + predicted_posteriors)
    total_cycles = len(all_posteriors)
    
    # DBN Reliability (Blue Line)
    reliability = 1.0 - all_posteriors[:, K-1]
    
    # --- Prepare Output DataFrame ---
    
    gbdt_state_padded = np.pad(y_hard.astype(float), (0, total_cycles - T_max), constant_values=np.nan)
    
    rul_true_padded = np.pad(g['RUL'].values.astype(float), (0, total_cycles - T_max), constant_values=np.nan)
    
    state_true_padded = np.pad(g['state'].values.astype(float), (0, total_cycles - T_max), constant_values=np.nan)


    results = pd.DataFrame({
        'unit': unit_id,
        'cycle': np.arange(1, total_cycles + 1),
        'DBN_reliability': reliability,
        'DBN_state': np.argmax(all_posteriors, axis=1),
        'GBDT_state': gbdt_state_padded, 
        'RUL_true': rul_true_padded, 
        'state_true': state_true_padded, 
        'is_prediction': [False]*T_max + [True]*(total_cycles - T_max)
    })
    
    return results

print("[infer] Running inference and prediction on all test units...")
all_results = []
for u, g in test.groupby("unit", sort=True):
    g = g.sort_values("cycle")
    results = run_inference_and_prediction_on_unit(g)
    all_results.append(results)
    
df_out = pd.concat(all_results, ignore_index=True)

out_path = OUT_DIR / f"{FD}_dbn_reliability.csv"
df_out.to_csv(out_path, index=False)
print(f"[infer] Results saved to {out_path.name}")