"""
C-MAPSS FD001 preprocessing:
 - Load raw files, compute RUL (train/test), cap RUL, discretize 5 health states.
 - Drop low-information channels, generate rolling features per unit.
 - Standardize features using training stats; save Parquet or CSV fallback.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler



FD = "FD001"                
DATA_DIR = Path("./archive")    #
OUT_DIR = Path("./processed")

ROLL_WIN = 20                # Rolling window size (cycles)
# Commonly dropped noisy/low-informative channels (adjust per analysis)
DROP_COLS = ["op3", "s1", "s5", "s10", "s16", "s19"]
RUL_CAP = 130                # Cap RUL to stabilize learning

# -------------------------
# Loaders and labeling
# -------------------------

def load_fd(fd: str, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # FD subsets have 26 columns: unit, cycle, op1-op3, sensors s1..s21
    colnames = (["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)])
    train = pd.read_csv(data_dir / f"train_{fd}.txt", sep=r"\s+", header=None, names=colnames)
    test  = pd.read_csv(data_dir / f"test_{fd}.txt",  sep=r"\s+", header=None, names=colnames)
    rul   = pd.read_csv(data_dir / f"RUL_{fd}.txt",   sep=r"\s+", header=None, names=["RUL"])
    for c in ["unit", "cycle"]:
        train[c] = train[c].astype(int)
        test[c]  = test[c].astype(int)
    return train, test, rul  # NASA docs confirm shape and columns per FD subset [web:11]

def add_rul_train(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df  # RUL from last cycle per unit is standard for C-MAPSS train [web:11]

def add_rul_test(df: pd.DataFrame, rul_df: pd.DataFrame, cap: int) -> pd.DataFrame:
    # The test RUL file provides remaining cycles after last observed cycle per unit in test
    df = df.copy()
    last_cycles = df.groupby("unit")["cycle"].transform("max")
    unit_ids = sorted(df["unit"].unique())
    assert len(unit_ids) == len(rul_df), "RUL file must have one row per test unit"
    final_rul_map = {u: int(rul_df.iloc[i, 0]) for i, u in enumerate(unit_ids)}
    df["RUL_final"] = df["unit"].map(final_rul_map)
    df["RUL"] = (df["RUL_final"] + (last_cycles - df["cycle"])).clip(upper=cap)
    df.drop(columns=["RUL_final"], inplace=True)
    return df  # This reconstructs per-row test RUL from the provided per-unit final RUL [web:11]

def drop_low_value_columns(df: pd.DataFrame, drops: List[str]) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in drops]
    return df[keep].copy()  # Dropping known low-informative channels is common in open baselines [web:11]

# -------------------------
# Health state discretization
# -------------------------

def discretize_states_from_rul(df: pd.DataFrame) -> pd.DataFrame:
    # 5 states from RUL with typical thresholds aligned to the cap
    df = df.copy()
    rul = df["RUL"].values
    states = np.select(
        [
            rul > 130,                 # S0 healthiest
            (rul <= 130) & (rul > 80), # S1
            (rul <= 80)  & (rul > 30), # S2
            (rul <= 30)  & (rul > 10), # S3
            rul <= 10,                 # S4 critical
        ],
        [0, 1, 2, 3, 4]
    ).astype(int)
    df["state"] = states
    return df  # Discrete health states support the GBDT monitoring classifier [web:11]

# -------------------------
# Rolling features per unit
# -------------------------

def add_rolling_features(df: pd.DataFrame,
                         sensors: List[str],
                         ops: List[str],
                         win: int) -> pd.DataFrame:
    df = df.sort_values(["unit", "cycle"]).copy()
    # Keep group key so it can be restored after apply
    grp = df.groupby("unit", group_keys=True)  # group_keys=True retains key in index [web:79]

    def _roll_g(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        for col in sensors + ops:
            r = g[col].rolling(win, min_periods=1)
            out[f"{col}_mean{win}"] = r.mean().values
            out[f"{col}_std{win}"]  = r.std().fillna(0.0).values
            out[f"{col}_min{win}"]  = r.min().values
            out[f"{col}_max{win}"]  = r.max().values
            out[f"{col}_d1"]        = g[col].diff().fillna(0.0).values
        return out  # Rolling windows per unit capture local trends and variability [web:11]

    # Forward-compatible semantics: exclude grouping columns during apply; restore group key
    res = grp.apply(_roll_g, include_groups=False)  # avoids future deprecation and matches pandas ≥2.2 [web:76][web:75]
    res = res.reset_index(level=0)  # bring 'unit' back as a column after grouped apply [web:79]
    return res.reset_index(drop=True)  # Final tidy frame for merging labels [web:79]

# -------------------------
# Standardization
# -------------------------

def fit_transform_standardize(train: pd.DataFrame,
                              test: pd.DataFrame,
                              feature_cols: List[str]):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[feature_cols])
    Xte = scaler.transform(test[feature_cols])
    train_std = train.copy()
    test_std  = test.copy()
    train_std[feature_cols] = Xtr
    test_std[feature_cols]  = Xte
    return train_std, test_std, scaler  # Standardization uses train stats for proper generalization [web:11]

# -------------------------
# Safe save (Parquet with fallback)
# -------------------------

def safe_save_frames(out: Dict[str, pd.DataFrame], fd: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = out["train"]
    test_df  = out["test"]
    try:
        train_df.to_parquet(out_dir / f"{fd}_train_preprocessed.parquet", index=False)
        test_df.to_parquet(out_dir / f"{fd}_test_preprocessed.parquet", index=False)
        print("Saved Parquet with available engine.")
    except ImportError:
        print("Parquet engine missing; falling back to CSV.")
        train_df.to_csv(out_dir / f"{fd}_train_preprocessed.csv", index=False)
        test_df.to_csv(out_dir / f"{fd}_test_preprocessed.csv", index=False)



def preprocess_fd(fd: str = FD,
                  data_dir: Path = DATA_DIR,
                  drop_cols: List[str] = DROP_COLS,
                  roll_win: int = ROLL_WIN,
                  rul_cap: int = RUL_CAP) -> Dict[str, pd.DataFrame]:
    train, test, rul = load_fd(fd, data_dir)

    # Remove low-value columns early
    train = drop_low_value_columns(train, drop_cols)
    test  = drop_low_value_columns(test,  drop_cols)

    # Labels
    train = add_rul_train(train, cap=rul_cap)
    test  = add_rul_test(test, rul_df=rul, cap=rul_cap)
    train = discretize_states_from_rul(train)
    test  = discretize_states_from_rul(test)

    # Strict sensor detection: s1..s21 only (avoid 'state')
    sensor_cols = [c for c in train.columns if re.fullmatch(r"s\d+", c)]
    op_cols     = [c for c in ["op1", "op2"] if c in train.columns]
    id_cols     = ["unit", "cycle"]
    label_cols  = ["RUL", "state"]

    # Feature engineering per unit
    base_cols = id_cols + op_cols + sensor_cols
    train_fe = add_rolling_features(train[base_cols], sensor_cols, op_cols, win=roll_win)
    test_fe  = add_rolling_features(test[base_cols],  sensor_cols, op_cols, win=roll_win)

    # Ensure id columns present before merging labels
    assert set(id_cols).issubset(train_fe.columns), "ID columns missing after feature engineering"
    assert set(id_cols).issubset(test_fe.columns),  "ID columns missing after feature engineering"

    # Re-attach labels
    train_fe = train_fe.merge(train[id_cols + label_cols], on=id_cols, how="left")
    test_fe  = test_fe.merge(test[id_cols + label_cols],   on=id_cols, how="left")

    # Final feature list: base + generated rolling/deltas, excluding ids and labels
    base_feats = op_cols + sensor_cols
    gen_prefixes = tuple([f"{c}_" for c in base_feats])
    feature_cols = [c for c in train_fe.columns
                    if (c in base_feats) or c.startswith(gen_prefixes) or c.endswith("_d1")]
    feature_cols = sorted(set(feature_cols) - set(id_cols) - set(label_cols))

    # Safety: no label leakage
    assert "state" not in feature_cols and "RUL" not in feature_cols, "Labels leaked into features"

    # Standardize
    train_std, test_std, scaler = fit_transform_standardize(train_fe, test_fe, feature_cols)

    # Assemble outputs
    cols_out = id_cols + feature_cols + label_cols
    train_out = train_std[cols_out].sort_values(["unit", "cycle"]).reset_index(drop=True)
    test_out  = test_std[cols_out].sort_values(["unit", "cycle"]).reset_index(drop=True)

    return {
        "train": train_out,
        "test":  test_out,
        "features": feature_cols,
        "scaler_mean": pd.Series(scaler.mean_, index=feature_cols, name="mean").to_frame(),
        "scaler_scale": pd.Series(scaler.scale_, index=feature_cols, name="scale").to_frame(),
    }  

if __name__ == "__main__":
    out = preprocess_fd()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_save_frames(out, FD, OUT_DIR)
    pd.Series(out["features"]).to_csv(OUT_DIR / f"{FD}_feature_list.txt", index=False, header=False)
    out["scaler_mean"].to_csv(OUT_DIR / f"{FD}_scaler_mean.csv")
    out["scaler_scale"].to_csv(OUT_DIR / f"{FD}_scaler_scale.csv")
    print(f"Saved outputs to {OUT_DIR.resolve()}")
