import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import argparse
from pathlib import Path

PLAYGROUND_DIR = Path(__file__).resolve().parent

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
    parts = Path(filepath).stem.split("_")
    algo = next((part for part in parts if part in ("DQN", "PPO")), "Unknown")
    method = next(
        (part.lower() for part in parts if part.lower() in ("transformer", "padding", "deep sets")),
        "Unknown",
    )
    if method == "deep sets":
        method = "Deep Sets"
    else:
        method = method.title()
    return f"{algo} + {method}"

def sort_key(label):
    """Sorts legend labels logically."""
    rank = 0
    if "Transformer" in label: rank += 1
    elif "Deep Sets" in label: rank += 2
    else: rank += 3
        
    algo_rank = 10 if "PPO" in label else 20
    return rank + algo_rank


def parse_optional_float(value):
    if value.lower() in ("none", "auto"):
        return None
    return float(value)


def plot_academic_csvs(phase="all", output_path=None, csv_files=None):
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    phase_specs = {
        "training": (
            PLAYGROUND_DIR / "results" / "training",
            "(a) Static Network Training",
        ),
        "tuning": (
            PLAYGROUND_DIR / "results" / "tuning",
            "(b) Variable-Size Network Tuning",
        ),
    }
    phases = tuple(phase_specs) if phase == "all" else (phase,)
    fig, axes = plt.subplots(
        len(phases),
        1,
        figsize=(4, 6 if len(phases) > 1 else 3.4),
        squeeze=False,
    )
    axes = axes[:, 0]
    all_handles = {}

    for ax, current_phase in zip(axes, phases):
        results_dir, caption = phase_specs[current_phase]
        files = (
            [str(path) for path in csv_files]
            if csv_files is not None
            else sorted(glob.glob(str(results_dir / "*.csv")))
        )
        print(f"Found {len(files)} files for {current_phase}")

        for idx, file in enumerate(files):
            df = pd.read_csv(file)
            if df.empty:
                continue
            label = parse_label(file)
            style = STYLES[idx % len(STYLES)]
            line, = ax.plot(
                df["Step"],
                df["Value"],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(3, len(df) // 10),
                markersize=3,
                linewidth=0.6,
            )
            all_handles.setdefault(label, line)

        ax.text(0.5, -0.25, caption, transform=ax.transAxes, ha="center", va="top", fontsize=11)
        ax.set_ylabel("Average Reward")
        ax.set_xlabel("Timesteps")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: f"{int(x / 1000)}k" if x > 0 else 0)
        )
        ax.tick_params(axis="both", which="major", labelsize=8)

    if all_handles:
        sorted_pairs = sorted(all_handles.items(), key=lambda item: sort_key(item[0]))
        labels = [label for label, _ in sorted_pairs]
        handles = [handle for _, handle in sorted_pairs]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            framealpha=0.9,
            edgecolor="black",
            fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if output_path is None:
        output_path = PLAYGROUND_DIR / "combined_learning_curves.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=1600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation reward CSV files.")
    parser.add_argument("--phase", choices=("training", "tuning", "all"), default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path, help="Plot only this run's reward CSV.")
    parser.add_argument("--algo", choices=("DQN", "PPO"), default="PPO")
    parser.add_argument("--mode", choices=("transformer", "ds", "padding"), default="transformer")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--focus_mode", choices=("stateful", "salience"), default="stateful")
    parser.add_argument("--dist_lambda", type=float, default=2.0)
    parser.add_argument("--cls_alpha", type=parse_optional_float, default=None)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=64)
    args = parser.parse_args()

    selected_csv = args.csv
    if selected_csv is None and args.phase != "all":
        cls_alpha = "auto" if args.cls_alpha is None else format(args.cls_alpha, ".12g")
        suffix = (
            f"topk{args.top_k}_focus-{args.focus_mode}_layers{args.n_layers}"
            f"_heads{args.n_heads}_embed{args.embedding_dim}"
            f"_clsalpha{cls_alpha}_distlambda{format(args.dist_lambda, '.12g')}"
        )
        stage = {"training": 0, "tuning": 1}[args.phase]
        selected_csv = (
            PLAYGROUND_DIR
            / "results"
            / args.phase
            / f"{stage}_{args.algo}_{args.mode}_{suffix}.csv"
        )

    if selected_csv is not None and not selected_csv.is_file():
        parser.error(f"Reward CSV does not exist: {selected_csv}")

    output_path = args.output
    if output_path is None and selected_csv is not None:
        output_path = PLAYGROUND_DIR / "results" / "plots" / f"{selected_csv.stem}.png"
    plot_academic_csvs(
        phase=args.phase,
        output_path=output_path,
        csv_files=[selected_csv] if selected_csv is not None else None,
    )