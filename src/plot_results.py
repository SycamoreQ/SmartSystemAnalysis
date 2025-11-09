import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

FD = os.environ.get("FD", "FD001")
ROOT = Path(".").resolve()
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

results_path = OUT_DIR / f"{FD}_dbn_reliability.csv"
if not results_path.exists():
    print(f"Results file not found: {results_path}")
    print("Please run 3_run_inference.py first.")
    exit()

df = pd.read_csv(results_path)
print(f"Loaded results from {results_path.name}")

unit_ids_to_plot = [5, 10, 15, 25, 30]
K = df['state_true'].max() + 1
colors = plt.cm.viridis(np.linspace(0, 1, K))

print(f"Generating plots for units: {unit_ids_to_plot}")

for unit_id in unit_ids_to_plot:
    g = df[df['unit'] == unit_id].sort_values('cycle')
    if g.empty:
        print(f"Unit {unit_id} not found in results.")
        continue
        
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('DBN Reliability', color=color)
    ax1.plot(g['cycle'], g['reliability'], color=color, linewidth=2.5, label='DBN Reliability (P(C_t != Fail))')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    
    ax1.scatter(g['cycle'], g['y_obs'] / (K-1), color='orange', alpha=0.5, s=15, label=f'GBDT State (Y_obs, normalized)')
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('True RUL', color=color)
    ax2.plot(g['cycle'], g['RUL_true'], color=color, linestyle='--', linewidth=2, label='True RUL')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=-5)
    
    fig.suptitle(f'DBN Reliability Assessment - Unit {unit_id}')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    
    plot_path = OUT_DIR / f"{FD}_unit_{unit_id}_reliability_plot.png"
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path.name}")

print("Plotting complete.")