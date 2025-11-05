import numpy as np
import pandas as pd
from pathlib import Path
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
from joblib import load
from xgboost import XGBClassifier

FD = "FD001"
PROC_DIR = Path("./processed")
MODELS_DIR = Path("./models")
OUT_DIR = Path("./outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_any(pq: Path, csv: Path):
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)

test = read_any(PROC_DIR / f"{FD}_test_preprocessed.parquet", PROC_DIR / f"{FD}_test_preprocessed.csv")
feature_cols = pd.read_csv(MODELS_DIR / f"{FD}_features.txt", header=None).iloc[:,0].tolist()
P_y_given_c = pd.read_csv(MODELS_DIR / f"{FD}_P_y_given_c.csv", header=False).values
P_trans = pd.read_csv(MODELS_DIR / f"{FD}_P_C_next_given_C_weibull.csv", header=False).values

cal_path = MODELS_DIR / f"{FD}_xgb_monitor_calibrated.joblib"
json_path = MODELS_DIR / f"{FD}_xgb_monitor.json"
if cal_path.exists():
    monitor = load(cal_path)
else:
    xgb = XGBClassifier(); xgb.load_model(str(json_path)); monitor = xgb

N_STATES = P_y_given_c.shape[0]
prior = np.ones(N_STATES) / N_STATES

model = DBN()
model.add_edges_from([
    (('C', 0), ('Y', 0)),
    (('C', 0), ('C', 1)),
])
cpd_C0 = TabularCPD(('C', 0), N_STATES, prior.reshape(-1,1))
cpd_C1 = TabularCPD(('C', 1), N_STATES, P_trans, evidence=[('C', 0)], evidence_card=[N_STATES])
cpd_Y0 = TabularCPD(('Y', 0), N_STATES, P_y_given_c, evidence=[('C', 0)], evidence_card=[N_STATES])
model.add_cpds(cpd_C0, cpd_C1, cpd_Y0)
model.initialize_initial_state()
 
def run_unit(unit_df):
    X_u = unit_df[feature_cols].values
    y_seq = monitor.predict(X_u)
    reliab = []
    inf = DBNInference(model)
    for y in y_seq:
        q = inf.query(variables=[('C',0)], evidence={(('Y',0)): int(y)})
        p_fail = q[('C',0)].values[-1]
        reliab.append(1.0 - p_fail)
        inf._forward_inference()
    return np.array(reliab), y_seq

outs = []
for u, g in test.groupby("unit"):
    g = g.sort_values("cycle")
    r, y_seq = run_unit(g)
    outs.append(pd.DataFrame({"unit": u, "cycle": g["cycle"].values, "reliability": r, "y_obs": y_seq, "RUL": g["RUL"].values}))
res = pd.concat(outs, ignore_index=True)
res.to_csv(OUT_DIR / f"{FD}_dbn_reliability_weibull.csv", index=False)
print("Saved:", OUT_DIR / f"{FD}_dbn_reliability_weibull.csv")
