import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

FD = "FD001"
PROC_DIR = Path("./processed")
def read_any(pq: Path, csv: Path):
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)

train = read_any(PROC_DIR / f"{FD}_train_preprocessed.parquet",
                 PROC_DIR / f"{FD}_train_preprocessed.csv")

X = train[[c for c in train.columns if c not in {"unit","cycle","RUL","state"}]].values
y = train["state"].astype(int).values
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

classes = np.unique(ytr)
w = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
wmap = {c:wi for c,wi in zip(classes, w)}
sw = np.array([wmap[yi] for yi in ytr])

gb = HistGradientBoostingClassifier(
    loss="log_loss",       
    max_depth=6,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)
gb.fit(Xtr, ytr, sample_weight=sw)

cal = CalibratedClassifierCV(gb, cv="prefit", method="isotonic")
cal.fit(Xval, yval)

y_pred = cal.predict(Xval)
print(classification_report(yval, y_pred, digits=4))

cm = confusion_matrix(yval, y_pred, labels=sorted(np.unique(y)))
row_sums = cm.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums==0] = 1.0
P_y_given_c = (cm / row_sums).T
pd.DataFrame(P_y_given_c).to_csv("./models/FD001_P_y_given_c.csv", header=False, index=False)
