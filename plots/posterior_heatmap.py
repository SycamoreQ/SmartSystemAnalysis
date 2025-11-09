# plots/posterior_heatmap.py
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from xgboost import XGBClassifier
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

FD="FD001"; PROC=Path("./processed"); MODELS=Path("./models")
test = pd.read_parquet(PROC/f"{FD}_test_preprocessed.parquet") if (PROC/f"{FD}_test_preprocessed.parquet").exists() else pd.read_csv(PROC/f"{FD}_test_preprocessed.csv")
features = pd.read_csv(MODELS/f"{FD}_features.txt", header=None).iloc[:,0].tolist()
P_y_given_c = pd.read_csv(MODELS/f"{FD}_P_y_given_c.csv", header=False).values
P_trans = pd.read_csv(MODELS/f"{FD}_P_C_next_given_C_weibull.csv", header=False).values
cal_path = MODELS / f"{FD}_xgb_monitor_calibrated.joblib"
if cal_path.exists(): monitor = load(cal_path)
else: 
    xgb = XGBClassifier(); xgb.load_model(str(MODELS/f"{FD}_xgb_monitor.json")); monitor = xgb

N = P_y_given_c.shape[0]
prior = np.ones(N)/N
m = DBN(); m.add_edges_from([(('C',0),('Y',0)), (('C',0),('C',1))])
cpd_C0 = TabularCPD(('C',0), N, prior.reshape(-1,1))
cpd_C1 = TabularCPD(('C',1), N, P_trans, evidence=[('C',0)], evidence_card=[N])
cpd_Y0 = TabularCPD(('Y',0), N, P_y_given_c, evidence=[('C',0)], evidence_card=[N])
m.add_cpds(cpd_C0, cpd_C1, cpd_Y0); m.initialize_initial_state()

unit = test["unit"].iloc[0]
g = test[test["unit"]==unit].sort_values("cycle")
X_u = g[features].values
y_seq = monitor.predict(X_u)

inf = DBNInference(m)
posts = []
for y in y_seq:
    q = inf.query(variables=[('C',0)], evidence={(('Y',0)): int(y)})
    posts.append(q[('C',0)].values)
    inf._forward_inference()
P = np.stack(posts, axis=0)  # T x N

plt.figure(figsize=(8,4))
sns.heatmap(P.T, cmap="viridis", cbar=True)
plt.xlabel("Time (cycle)"); plt.ylabel("State s")
plt.title(f"Posterior P(C_t=s) heatmap (unit {unit})")
plt.show()
