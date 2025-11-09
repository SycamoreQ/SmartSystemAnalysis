import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

FD = os.environ.get("FD", "FD001")
ROOT = Path(".").resolve()
PROC_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading preprocessed training data...")
train = pd.read_parquet(PROC_DIR / f"{FD}_train_preprocessed.parquet")
feature_cols = pd.read_csv(PROC_DIR / f"{FD}_feature_list.txt", header=None).iloc[:,0].tolist()
labels = pd.read_csv(PROC_DIR / f"{FD}_class_labels.txt", header=None).iloc[:,0].tolist()
K = len(labels)

X = train[feature_cols].values
y = train["state"].astype(int).values

Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training on {len(ytr)} samples, validating on {len(yval)} samples.")

classes = np.unique(ytr)
w = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
wmap = {c: wi for c, wi in zip(classes, w)}
sw = np.array([wmap[yi] for yi in ytr], dtype=float)
 
print("Training GBDT monitor...")
gb = HistGradientBoostingClassifier(
    loss="log_loss",
    max_depth=12,        
    learning_rate=0.05,
    max_leaf_nodes=63,
    l2_regularization=0.5,
    random_state=42,
    validation_fraction=None, 
    class_weight=None  
)
gb.fit(Xtr, ytr)

print("Calibrating model (isotonic)...")
cal = CalibratedClassifierCV(gb, cv="prefit", method="isotonic")
cal.fit(Xval, yval)

y_pred = cal.predict(Xval)
print("\n--- GBDT Monitor Classification Report ---")
target_names = [f"State {l}" for l in labels]
print(classification_report(yval, y_pred, digits=4, target_names=target_names))
print("-------------------------------------------\n")


print("Generating P(Y|C) CPT from confusion matrix...")
cm = confusion_matrix(yval, y_pred, labels=labels)
row_sums = cm.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums == 0] = 1.0 # Avoid divide-by-zero
P_predicted_given_true = cm / row_sums

P_y_given_c = P_predicted_given_true.T
print(f"P(Y|C) shape: {P_y_given_c.shape}")

pd.DataFrame(P_y_given_c).to_csv(MODELS_DIR / f"{FD}_P_y_given_c.csv", header=False, index=False)
pd.Series(feature_cols).to_csv(MODELS_DIR / f"{FD}_features.txt", index=False, header=False)
pd.Series(labels).to_csv(MODELS_DIR / f"{FD}_class_labels.txt", index=False, header=False)
dump(cal, MODELS_DIR / f"{FD}_monitor_calibrated.joblib")

print("Saved artifacts to:", MODELS_DIR)