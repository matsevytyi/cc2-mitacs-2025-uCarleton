import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import argparse
import re
import shutil
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
    {"color": "purple", "linestyle": "-.", "marker": "v"},
    {"color": "brown", "linestyle": ":", "marker": "P"},
]

def smooth_data(data, alpha=0.01):
    """Applies Exponential Moving Average smoothing."""
    return data.ewm(alpha=alpha, adjust=False).mean()

def parse_label(filepath):
    """Parses filename to get the standard label."""
    path = Path(filepath)
    stem = path.stem
    if stem in ("2_DQN_transformer", "4_PPO_transformer"):
        algorithm = "DQN" if "DQN" in stem else "PPO"
        return f"{algorithm} Transformer (reference)"

    parts = stem.split("_")
    algorithm = next((part for part in parts if part in ("DQN", "PPO")), "Unknown")
    mode = next(
        (part for part in parts if part in ("transformer", "padding", "deep")),
        "Unknown",
    )
    if mode == "transformer":
        label = f"{algorithm} Transformer"
        top_k = re.search(r"topk(\d+)", stem)
        focus = re.search(r"focus-([^_]+)", stem)
        layers = re.search(r"layers(\d+)", stem)
        heads = re.search(r"heads(\d+)", stem)
        dimension = re.search(r"embed(\d+)", stem)
        alpha = re.search(r"clsalpha([^_]+)", stem)
        dist_lambda = re.search(r"distlambda([^_]+)", stem)
        settings = [
            f"k={top_k.group(1)}" if top_k else None,
            focus.group(1) if focus else None,
            f"L={layers.group(1)}" if layers else None,
            f"H={heads.group(1)}" if heads else None,
            f"D={dimension.group(1)}" if dimension else None,
            f"alpha={alpha.group(1)}" if alpha else None,
            f"lambda={dist_lambda.group(1)}" if dist_lambda else None,
        ]
        return f"{label} ({', '.join(item for item in settings if item)})"
    return f"{algorithm} {mode.title()}"

def sort_key(label):
    """Sorts legend labels logically."""
    rank = 0
    if "Transformer" in label: rank += 1
    elif "Deep Sets" in label: rank += 2
    else: rank += 3
        
    algo_rank = 10 if "PPO" in label else 20
    return rank + algo_rank


def plot_academic_csvs(stage="stage1", phase="all", output_path=None):
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    phase_specs = {
        "training": (
            "training",
            "(a) Static Network Training",
        ),
        "tuning": (
            "tuning",
            "(b) Variable-Size Network Tuning",
        ),
    }
    phases = tuple(phase_specs) if phase == "all" else (phase,)
    fig, axes = plt.subplots(
        len(phases),
        1,
        figsize=(9, 7 if len(phases) > 1 else 4.5),
        squeeze=False,
    )
    axes = axes[:, 0]
    all_handles = {}

    for ax, current_phase in zip(axes, phases):
        phase_dir, caption = phase_specs[current_phase]
        results_dir = PLAYGROUND_DIR / "sweeps" / stage / "results" / phase_dir
        files = sorted(results_dir.glob("*.csv"))
        if current_phase == "training":
            legacy_training_dir = results_dir / "training"
            files.extend(sorted(legacy_training_dir.glob("*.csv")))
        source_reference_dir = PLAYGROUND_DIR / "results" / phase_dir
        reference_dir = (
            PLAYGROUND_DIR / "sweeps" / stage / "results" / "references" / phase_dir
        )
        reference_dir.mkdir(parents=True, exist_ok=True)
        references = []
        for filename in ("2_DQN_transformer.csv", "4_PPO_transformer.csv"):
            source = source_reference_dir / filename
            destination = reference_dir / filename
            if source.is_file() and (
                not destination.exists()
                or source.stat().st_mtime > destination.stat().st_mtime
            ):
                shutil.copy2(source, destination)
            if destination.is_file():
                references.append(destination)
        files.extend(references)
        print(f"Found {len(files)} curves for {stage}/{current_phase}")

        for idx, file in enumerate(files):
            df = pd.read_csv(file)
            if df.empty:
                continue
            label = parse_label(file)
            is_reference = file.parent == reference_dir
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
                linewidth=0.9,
                zorder=3 if is_reference else 2,
            )
            if label not in all_handles:
                all_handles[label] = line

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
        output_path = (
            PLAYGROUND_DIR / "sweeps" / stage / "plots" / f"{phase}_learning_curves.png"
        )
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=1600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep-stage reward CSV files.")
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2a", "stage2b", "stage3"),
        required=True,
    )
    parser.add_argument("--phase", choices=("training", "tuning", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_academic_csvs(stage=args.stage, phase=args.phase, output_path=args.output)