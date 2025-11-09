import pandas as pd
import numpy as np
from pathlib import Path
import os 

FD = "FD001"
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "archive" 
PROC_DIR = ROOT / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

op_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensors = [f's_{i}' for i in range(1, 22)]
cols = ['unit', 'cycle'] + op_settings + sensors

features = [
    's_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
    's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21'
]
features_cols = op_settings + features
print(f"Using {len(features_cols)} features.")

bins = [-np.inf, 20, 60, 120, 200, np.inf]
labels = [4, 3, 2, 1, 0]

def load_and_process_train():
    print(f"Processing {FD}_train.txt...")
    df = pd.read_csv(DATA_DIR / f"train_{FD}.txt", sep=' ', header=None, names=cols, index_col=False)
    df = df.dropna(axis=1, how='all')
    
    max_cycles = df.groupby('unit')['cycle'].max().to_dict()
    df['RUL'] = df['unit'].map(max_cycles) - df['cycle']


    df['state'] = pd.cut(df['RUL'], bins=bins, labels=labels, right=True).astype(int)
    
    print(f"Train data shape: {df.shape}")
    print("State distribution (Train):\n", df['state'].value_counts().sort_index())
    
    out_path = PROC_DIR / f"{FD}_train_preprocessed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return df

def load_and_process_test():
    print(f"Processing {FD}_test.txt and RUL_{FD}.txt...")
    df = pd.read_csv(DATA_DIR / f"test_{FD}.txt", sep=' ', header=None, names=cols, index_col=False)
    df = df.dropna(axis=1, how='all')
    
    true_rul = pd.read_csv(DATA_DIR / f"RUL_{FD}.txt", sep=' ', header=None, names=['RUL'], index_col=False)
    true_rul = true_rul.dropna(axis=1, how='all')
    true_rul['unit'] = true_rul.index + 1
    
    max_cycles = df.groupby('unit')['cycle'].max().to_dict()
    
    df = df.merge(true_rul, on='unit', how='left')
    
    df['RUL'] = (df['unit'].map(max_cycles) - df['cycle']) + df['RUL']
    
    df['state'] = pd.cut(df['RUL'], bins=bins, labels=labels, right=True).astype(int)

    print(f"Test data shape: {df.shape}")
    print("State distribution (Test):\n", df['state'].value_counts().sort_index())

    out_path = PROC_DIR / f"{FD}_test_preprocessed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return df



if __name__ == "__main__":
    load_and_process_train()
    load_and_process_test()
    
    sorted_labels = sorted(labels)
    pd.Series(features_cols).to_csv(PROC_DIR / f"{FD}_feature_list.txt", index=False, header=False)
    pd.Series(sorted_labels).to_csv(PROC_DIR / f"{FD}_class_labels.txt", index=False, header=False)
    print("Preprocessing complete.")
