import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
from joblib import load
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


FD = os.environ.get("FD", "FD001")
PROC_DIR = Path("./processed")
MODELS_DIR = Path("./models")
OUT_DIR = Path("./outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_any(pq: Path, csv: Path):
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {pq} and {csv}")

test = read_any(PROC_DIR / f"{FD}_test_preprocessed.parquet", PROC_DIR / f"{FD}_test_preprocessed.csv")
feature_cols = pd.read_csv(MODELS_DIR / f"{FD}_feature_list.txt", header=None).iloc[:,0].tolist()

#Load P(Y|C) CPT from trainer
P_y_given_c = pd.read_csv(MODELS_DIR / f"{FD}_P_y_given_c.csv", header=False).values  


cal_path = MODELS_DIR / f"{FD}_xgb_monitor_calibrated.joblib"
json_path = MODELS_DIR / f"{FD}_xgb_monitor.json"
monitor = None
if cal_path.exists():
    monitor = load(cal_path) 
elif json_path.exists():
    xgb = XGBClassifier()
    xgb.load_model(str(json_path))
    monitor = xgb
else:
    raise FileNotFoundError("No trained monitor model found")



