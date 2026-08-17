import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

# Academic Formatting Dictionary
STYLES = [
    {"color": "black", "linestyle": ":", "marker": "o"},    # Solid line, Circle
    {"color": "red", "linestyle": "-", "marker": "s"},      # Dashed line, Square
    {"color": "gray", "linestyle": "--", "marker": "^"},    # Dash-dot line, Triangle
    {"color": "blue", "linestyle": "-", "marker": "D"},     # Dotted line, Diamond
    {"color": "orange", "linestyle": "--", "marker": "|"},  # Dotted line, Pipe
    {"color": "green", "linestyle": ":", "marker": "X"},    # Dotted line, X
]

def smooth_data(data, alpha=0.01):
    """Applies Exponential Moving Average smoothing."""
    return data.ewm(alpha=alpha, adjust=False).mean()

def parse_label(filepath):
    """Parses filename to get the standard label."""
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    method = parts[2][:-4] if len(parts) > 1 else "Unknown"
    algo = parts[1] if len(parts) > 1 else "Unknown"
    
    # Capitalize cleanly for the legend
    if method == "gat": method = "transformer" # assuming this mapping based on your previous code
    
    # Replace the confusing "Default" / "BLine" with "Padding" as discussed
    if method.lower() in ["bline", "default"]: method = "Padding"
    if method.lower() == "deep sets": method = "Deep Sets"
    if method.lower() == "transformer": method = "Transformer"
    
    return f"{algo} + {method}"

def sort_key(label):
    """Sorts legend labels logically."""
    rank = 0
    if "Transformer" in label: rank += 1
    elif "Deep Sets" in label: rank += 2
    else: rank += 3
        
    algo_rank = 10 if "PPO" in label else 20
    return rank + algo_rank

def plot_academic_csvs():
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    
    # Create 2 subplots vertically stacked
    # Width=4, Height=6 is perfect for a single column in an IEEE double-column paper
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
    
    # ==========================================
    # TOP PLOT: STATIC TRAINING
    # ==========================================
    train_files = glob.glob("CybORG/.playground/results/training/*")
    print(f"Found {len(train_files)} files for STATIC training") 
    
    for idx, file in enumerate(train_files):
        df = pd.read_csv(file)
        label = parse_label(file)
        
        #smoothed_vals = smooth_data(df['Value'])
        vals = df['Value']
        style = STYLES[idx]
        
        ax1.plot(df['Step'], vals, label=label, 
                color=style["color"], linestyle=style["linestyle"], 
                marker=style["marker"], markevery=max(3, len(df) // 10), 
                markersize=3, linewidth=0.6)

    # ax1.set_title("(a) Static Network Training", fontsize=11)
    
    ax1.text(0.5, -0.25, "(a) Static Network Training",
         transform=ax1.transAxes,
         ha='center', va='top', fontsize=11)
    
    ax1.set_ylabel("Average Reward")
    # ax1.set_xlabel("Timesteps") # Leave off for top plot to save space
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ==========================================
    # BOTTOM PLOT: DYNAMIC TUNING
    # ==========================================
    tune_files = glob.glob("CybORG/.playground/results/tuning/*")
    print(f"Found {len(tune_files)} files for DYNAMIC tuning") 
    
    for idx, file in enumerate(tune_files):
        df = pd.read_csv(file)
        label = parse_label(file)
        
        #smoothed_vals = smooth_data(df['Value'])
        vals = df['Value']
        if label == "PPO + Deep Sets":
            vals -= 0 # move if needed to compare where they match
        style = STYLES[idx]
            
        style = STYLES[idx % len(STYLES)]
        
        ax2.plot(df['Step'], vals, label=label, 
                color=style["color"], linestyle=style["linestyle"], 
                marker=style["marker"], markevery=max(3, len(df) // 10), 
                markersize=3, linewidth=0.6)

    # ax2.set_title("(b) Variable-Size Network Tuning", fontsize=11)
    
    ax2.text(0.5, -0.25, "(b) Variable-Size Network Tuning",
         transform=ax2.transAxes,
         ha='center', va='top', fontsize=11)
    
    ax2.set_ylabel("Average Reward")
    ax2.set_xlabel("Timesteps")
    ax2.grid(True, linestyle='--', alpha=0.6)

    # ==========================================
    # SHARED LEGEND (PLACED AT TOP)
    # ==========================================
    # Grab handles from ONE of the axes (since they are the same)
    handles, labels = ax1.get_legend_handles_labels()
    
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs)

    # For ax1 (static)
    ax1.xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, pos: f"{int(x/1000)}k" if x > 0 else 0)
    )

    # For ax2 (dynamic)
    ax2.xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, pos: f"{int(x/1000)}k" if x > 0 else 0)
    )

    # Make tick labels smaller on both axes
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    # Place a single legend spanning across the very top of the whole figure
    fig.legend(
        sorted_handles, 
        sorted_labels, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.05), # Pushes it just above the top plot
        ncol=2,                     # 2 columns to keep it flat
        framealpha=0.9, 
        edgecolor='black',
        fontsize=9
    )
    
    # Adjust layout so the legend doesn't overlap the title of ax1
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    
    
    plt.savefig("combined_learning_curves.png", dpi=1600, bbox_inches='tight')
    print("Saved plot to combined_learning_curves.png")
    # plt.show()

if __name__ == "__main__":
    plot_academic_csvs()