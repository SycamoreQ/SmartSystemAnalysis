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
    print("Please run run_infer.py first.")
    exit()

df = pd.read_csv(results_path)
print(f"Loaded results from {results_path.name}")

# K is the number of states (0 to K-1). Use the max value from the true state column.
# Use 5 as a fallback if all true RUL is NaN (e.g., in the prediction phase)
K = int(df['state_true'].max() + 1) if df['state_true'].max() + 1 > 0 else 5 
colors = plt.cm.viridis(np.linspace(0, 1, K))

unit_ids_to_plot = [5, 10, 15, 25, 30] # Common units for plotting

print(f"Generating plots for units: {unit_ids_to_plot}")

for unit_id in unit_ids_to_plot:
    g = df[df['unit'] == unit_id].sort_values('cycle')
    if g.empty:
        print(f"Unit {unit_id} not found in results.")
        continue
        
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # ----------------------------------------------------
    # AXIS 1: DBN Reliability (Blue Line) and GBDT State (Orange Dots)
    # ----------------------------------------------------
    color = 'tab:blue'
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('DBN Reliability', color=color)
    
    # Plot DBN Reliability (Filtered + Predicted)
    ax1.plot(g['cycle'], g['DBN_reliability'], color=color, linewidth=2.5, label='DBN Reliability (Filtered & Predicted)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot GBDT State (Only in the Filtering Phase - where it is not NaN)
    # Normalize GBDT state to the range [0, 1] for visual overlay
    valid_gbdt = g[~g['GBDT_state'].isna()]
    ax1.scatter(valid_gbdt['cycle'], valid_gbdt['GBDT_state'] / (K-1), 
                color='orange', alpha=0.5, s=15, 
                label=f'GBDT State (Y_obs, normalized)')
    
    # Prediction Region Marker
    T_max_data = g[~g['is_prediction']]['cycle'].max()
    if not pd.isna(T_max_data):
        ax1.axvline(x=T_max_data, color='gray', linestyle='--', linewidth=1, label='Prediction Start')
        # Shade the prediction region lightly
        ax1.axvspan(T_max_data, g['cycle'].max(), color='gray', alpha=0.1, label='Prediction Region')

    # ----------------------------------------------------
    # AXIS 2: True RUL (Green Dashed Line)
    # ----------------------------------------------------
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('True RUL', color=color)
    
    # Plot True RUL (Only in the Filtering Phase - where it is not NaN)
    valid_rul = g[~g['RUL_true'].isna()]
    ax2.plot(valid_rul['cycle'], valid_rul['RUL_true'], color=color, linestyle='--', linewidth=2.0, label='True RUL')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Set title and legend
    fig.suptitle(f'Unit {unit_id} Reliability Assessment and Prediction (K={K})', fontsize=16)
    
    # Manually collect handles and labels to combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = OUT_DIR / f"{FD}_unit_{unit_id}_reliability_prediction_plot.png"
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved plot for Unit {unit_id} to {plot_filename.name}")