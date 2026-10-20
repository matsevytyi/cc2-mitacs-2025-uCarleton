import os, sys, inspect
import argparse
from pathlib import Path

from torch.cuda import device_of

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import DeepSetsPermInvWrapper, TransformerWrapper, ChallengeWrapper, PaddingWrapper
from scenario_shuffler import copy_scenario_for_run

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor


def parse_optional_float(value):
    if value.lower() in ("none", "auto"):
        return None
    return float(value)


def add_pipeline_arguments(parser):
    parser.add_argument("--algo", choices=("DQN", "PPO"), default="PPO")
    parser.add_argument("--mode", choices=("transformer", "ds", "padding"), default="transformer")
    parser.add_argument(
        "--action_space_mode",
        choices=("default", "topk", "pad", "cutoff"),
        default="topk",
    )
    parser.add_argument("--top_k", type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument("--focus_mode", choices=("stateful", "salience"), default="stateful")
    parser.add_argument("--dist_lambda", type=float, default=2.0)
    parser.add_argument("--cls_alpha", type=parse_optional_float, default=None)
    parser.add_argument("--n_layers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, choices=(32, 64, 128, 256), default=64)
    parser.add_argument(
        "--sweep_stage",
        choices=("stage1", "stage2a", "stage2b", "stage3"),
        default="stage1",
    )


def hyperparameter_suffix(args):
    cls_alpha = "auto" if args.cls_alpha is None else format(args.cls_alpha, ".12g")
    return (
        f"topk{args.top_k}_focus-{args.focus_mode}_layers{args.n_layers}"
        f"_heads{args.n_heads}_embed{args.embedding_dim}"
        f"_clsalpha{cls_alpha}_distlambda{format(args.dist_lambda, '.12g')}"
    )


def make_training_run_id(args):
    action_tag = (
        f"_{args.action_space_mode}_k{args.top_k}_{args.focus_mode}"
        if args.mode == "transformer" and args.action_space_mode in ("default", "topk")
        else f"_{args.action_space_mode}" if args.mode == "transformer" else ""
    )
    return (
        f"TRAIN_{args.mode}{action_tag}_{hyperparameter_suffix(args)}"
        f"_{args.algo}_vs_RedMeander"
    )


def reward_csv_path(results_dir, phase, args):
    stage = {"training": 0, "tuning": 1}[phase]
    return results_dir / (
        f"{stage}_{args.algo}_{args.mode}_{hyperparameter_suffix(args)}.csv"
    )


def export_reward_csv(log_root, csv_path):
    import csv
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = []
    for root, _, files in os.walk(log_root):
        event_files.extend(
            Path(root) / name
            for name in files
            if name.startswith("events.out.tfevents")
        )

    if not event_files:
        raise RuntimeError(f"No TensorBoard event files found under {log_root}")

    latest_event_file = max(event_files, key=lambda path: path.stat().st_mtime)
    accumulator = EventAccumulator(
        str(latest_event_file.parent),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    if "rollout/ep_rew_mean" not in accumulator.Tags().get("scalars", []):
        raise RuntimeError(
            "No rollout/ep_rew_mean scalars found in the latest TensorBoard run "
            f"{latest_event_file.parent}"
        )

    rewards = {
        event.step: event.value
        for event in accumulator.Scalars("rollout/ep_rew_mean")
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("Step", "Value"))
        writer.writerows(sorted(rewards.items()))
    print(f"Reward CSV saved to: {csv_path}")

# ========== CONFIGURATION ==========
PLAYGROUND_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Train the transformer ablation configuration.")
add_pipeline_arguments(parser)
args = parser.parse_args()
if args.n_heads <= 0 or (2 * args.embedding_dim) % args.n_heads:
    parser.error("--n_heads must be positive and evenly divide 2 * --embedding_dim")

STAGE_DIR = PLAYGROUND_DIR / "sweeps" / args.sweep_stage
LOG_DIR = STAGE_DIR / "logs" / "training"
RESULTS_DIR = STAGE_DIR / "results" / "training"
MODEL_DIR = STAGE_DIR / "models" / "training"
SCENARIO_DIR = STAGE_DIR / "scenarios"

ALGORITHM = {"DQN": DQN, "PPO": PPO}[args.algo]
mode = args.mode
action_space_mode = args.action_space_mode
top_k = args.top_k
focus_mode = args.focus_mode

transformer = (mode == "transformer")
deepset = (mode == "ds")
pad = (mode == "padding")

RUN_ID = make_training_run_id(args)
extended = True

TOTAL_TIMESTEPS = 500_000

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters per algorithm
HYPERPARAMS = {
    "DQN": {
        "policy": "MlpPolicy",
        "verbose": 2,
        "tensorboard_log": str(LOG_DIR / RUN_ID),
        "device": device,
        "buffer_size": 500_000,
        "learning_starts": 10_000,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 5_000,
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.025,
    },
    "PPO": {
        "policy": "MlpPolicy",
        "verbose": 2,
        "tensorboard_log": str(LOG_DIR / RUN_ID),
        "device": device,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}

# ========== ENVIRONMENT SETUP ==========
if extended:
    base_scenario_path = PLAYGROUND_DIR / "scenarios" / "Scenario2.yaml"
    path = str(copy_scenario_for_run(
        base_scenario_path,
        SCENARIO_DIR / f"{RUN_ID}.yaml",
    ))
else:
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
cyborg.reset()

if transformer:
    gym_env = TransformerWrapper(
        raw_cyborg=cyborg,
        agent_name='Blue',
        max_steps=100,
        device=device,
        action_space_mode=action_space_mode,
        max_actions=None if action_space_mode == "topk" else 240,
        top_k=top_k,
        embedding_dim=args.embedding_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        focus_mode=focus_mode,
        cls_alpha=args.cls_alpha,
        dist_lambda=args.dist_lambda,
    )
elif pad:
    gym_env = PaddingWrapper(env=cyborg, agent_name='Blue', max_devices=100, max_steps=100)
elif deepset:
    gym_env = DeepSetsPermInvWrapper(raw_cyborg=cyborg, agent_name='Blue', max_actions=240, max_steps=100)
else:
    gym_env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=100)
    from gymnasium.wrappers import EnvCompatibility
    gym_env = EnvCompatibility(gym_env)

gym_env.reset()
#print(cyborg)

# ========== MODEL INITIALIZATION ==========
algorithm_name = ALGORITHM.__name__
hyperparams = HYPERPARAMS[algorithm_name]

model = ALGORITHM(env=gym_env, **hyperparams)

# ========== TRAINING ==========
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name=f"{RUN_ID}",
    log_interval=1 if algorithm_name == "PPO" else 10
)

# PPO logs by rollout, so short runs may finish before its usual interval.
episode_returns = [
    float(episode["r"])
    for episode in model.ep_info_buffer
    if "r" in episode
]
if episode_returns:
    model.logger.record(
        "rollout/ep_rew_mean",
        sum(episode_returns) / len(episode_returns),
    )
    model.logger.dump(step=model.num_timesteps)

# ========== SAVE MODEL ==========
wrapper_type = "transformer" if transformer else "padding" if pad else "challenge"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / RUN_ID
model.save(str(model_path))
print(f"Model saved to: {model_path}.zip")

if transformer or deepset:
    encoder = gym_env.encoder  # or gym_env.transformer_encoder or similar
    encoder_path = Path(f"{model_path}.encoder.pth")
    encoder.save_weights(str(encoder_path))
    print(f"Encoder weights saved to: {encoder_path}")

csv_path = reward_csv_path(RESULTS_DIR, "training", args)
export_reward_csv(LOG_DIR / RUN_ID, csv_path)