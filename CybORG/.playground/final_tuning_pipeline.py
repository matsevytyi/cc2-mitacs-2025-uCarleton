import os, sys, inspect
import argparse
from pathlib import Path
import numpy as np
import torch
import random
from collections import deque


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import TransformerWrapper, PaddingWrapper, DeepSetsPermInvWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure


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
LOG_DIR = PLAYGROUND_DIR / "logs"
RESULTS_DIR = PLAYGROUND_DIR / "results"

parser = argparse.ArgumentParser(description="Tune the transformer ablation configuration.")
add_pipeline_arguments(parser)
args = parser.parse_args()
if args.n_heads <= 0 or (2 * args.embedding_dim) % args.n_heads:
    parser.error("--n_heads must be positive and evenly divide 2 * --embedding_dim")

ALGORITHM = {"DQN": DQN, "PPO": PPO}[args.algo]
mode = args.mode
action_space_mode = args.action_space_mode
top_k = args.top_k
focus_mode = args.focus_mode

transformer = (mode == "transformer")
deepset = (mode == "ds")
pad = (mode == "padding")

OLD_RUN_ID = make_training_run_id(args)

USE_PRETRAINED = True
TOTAL_TIMESTEPS = 25_000
action_tag = (
    f"_{action_space_mode}_k{top_k}_{focus_mode}"
    if transformer and action_space_mode in ("default", "topk")
    else f"_{action_space_mode}" if transformer else ""
)
model_label = f"Transformer{action_tag}" if transformer else "Padding" if pad else "DeepSets"
RUN_ID = f"{ALGORITHM.__name__}_{model_label}_{hyperparameter_suffix(args)}_tuning_x150_extended_dynamic_topology"
TENSORBOARD_LOG = LOG_DIR / "tuning" / RUN_ID

device = 'cpu'

base_dir = PLAYGROUND_DIR.parent
scenario_name = f"Scenario2_{'Transformer' if transformer else 'Padding' if pad else 'DeepSets'}_{ALGORITHM.__name__}"
SCENARIO_PATH = str(PLAYGROUND_DIR / "scenarios" / f"{scenario_name}.yaml")

HYPERPARAMS = {
    "DQN": {
        "policy": "MlpPolicy",
        "verbose": 2,
        "tensorboard_log": str(TENSORBOARD_LOG),
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
        "tensorboard_log": str(TENSORBOARD_LOG),
        "device": device,
        "learning_rate": 5e-5,
        "n_steps": 4096,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}

def create_cyborg_env(yaml_path: str):
    cyborg = CybORG(yaml_path, 'sim', agents={'Red': RedMeanderAgent})
    cyborg.reset()
    return cyborg

# ========== LOGGING ===========

import sys

outname = str(PLAYGROUND_DIR / f"{RUN_ID}_console_log.txt")
sys.stdout = open(outname, 'w', buffering=1)  # line buffering for real-time write

# Optionally, also keep output in console:
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.__stdout__, open(outname, 'a', buffering=1))

# ========== TRAINING PIPELINE ==========

initial_raw_cyborg = create_cyborg_env(SCENARIO_PATH)

if transformer:
    gym_env = TransformerWrapper(
        agent_name='Blue',
        raw_cyborg=initial_raw_cyborg,
        max_steps=100,
        action_space_mode=action_space_mode,
        knowledge_update_mode="tune",
        env_creator=create_cyborg_env,
        yaml_path=SCENARIO_PATH,
        max_actions=None if action_space_mode == "topk" else 240,
        weights_path=str(PLAYGROUND_DIR / f"{OLD_RUN_ID}.encoder.pth"),
        top_k=top_k,
        embedding_dim=args.embedding_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        focus_mode=focus_mode,
        cls_alpha=args.cls_alpha,
        dist_lambda=args.dist_lambda,
    )
elif deepset:
    gym_env = DeepSetsPermInvWrapper(
        agent_name='Blue',
        raw_cyborg=initial_raw_cyborg,
        max_steps=100,
        knowledge_update_mode="tune",
        env_creator=create_cyborg_env,
        yaml_path=SCENARIO_PATH,
        max_actions=240,
        weights_path=f"{OLD_RUN_ID}.encoder.pth"
    )
else:
    gym_env = PaddingWrapper(
        agent_name='Blue',
        env=initial_raw_cyborg,
        max_steps=100,
        knowledge_update_mode="tune",
        env_creator=create_cyborg_env,
        yaml_path=SCENARIO_PATH,
        )

if USE_PRETRAINED:
    gym_env._reload_environment()
gym_env.reset()

# Create model
if USE_PRETRAINED:
    model = ALGORITHM.load(
        str(PLAYGROUND_DIR / OLD_RUN_ID),
        env=gym_env,
        tensorboard_log=str(TENSORBOARD_LOG),
    )
else:
    hyperparams = HYPERPARAMS[ALGORITHM.__name__]
    model = ALGORITHM(env=gym_env, **hyperparams)

# ========== MANUAL LOGGING FOR CONVERGENCE ANALYSIS ==========
# Configure TensorBoard logger
logger = configure(folder=str(TENSORBOARD_LOG), format_strings=["tensorboard", "stdout"])
model.set_logger(logger)

# Track metrics for convergence analysis
episode_rewards = deque(maxlen=100)
episode_lengths = deque(maxlen=100)
recon_losses = deque(maxlen=100) if transformer else None

total_steps = 0
episode_count = 0

# NEW: Force initial logging at step 0
model.logger.record("rollout/ep_rew_mean", 0.0)
model.logger.record("rollout/ep_len_mean", 0.0)
if transformer:
    init_recon = gym_env.encoder.recon_loss
    model.logger.record("train/recon_loss", init_recon.item() if isinstance(init_recon, torch.Tensor) else float(init_recon))
model.logger.dump(step=0)

print(f"Starting training with DYNAMIC topology")

# Training loop with manual logging
LOG_INTERVAL = 1000  # Log every 1k steps for smoother graphs

while total_steps < TOTAL_TIMESTEPS:
    # Run until next log point
    steps_to_run = min(LOG_INTERVAL, TOTAL_TIMESTEPS - total_steps)
    
    model.learn(
        total_timesteps=steps_to_run,
        tb_log_name=f"{RUN_ID}",
        reset_num_timesteps=False,
        log_interval=999999  # Disable internal logging
    )
    
    total_steps += steps_to_run
    
    # Manual logging
    if hasattr(gym_env, 'get_episode_rewards') and hasattr(gym_env, 'get_episode_lengths'):
        ep_rewards = gym_env.get_episode_rewards()
        ep_lengths = gym_env.get_episode_lengths()
    else:
        # Fallback: use monitor wrapper data if available
        ep_rewards = getattr(gym_env, 'episode_rewards_list', np.mean)
        ep_lengths = getattr(gym_env, 'episode_lengths_list', np.mean)
    
    episode_rewards.extend(ep_rewards)
    mean_reward = np.mean(episode_rewards)
    model.logger.record("rollout/ep_rew_mean", mean_reward)
    
    episode_lengths.extend(ep_lengths)
    mean_length = np.mean(episode_lengths)
    model.logger.record("rollout/ep_len_mean", mean_length)
    
    # NEW: Log reconstruction loss for Transformer (convergence metric)
    if transformer:
        recon_loss = gym_env.encoder.recon_loss
        if isinstance(recon_loss, torch.Tensor):
            recon_loss = recon_loss.item()
        recon_losses.append(recon_loss)
        model.logger.record("train/recon_loss", np.mean(recon_losses))
    
    # NEW: Convergence speed metric (reward improvement rate)
    if len(episode_rewards) >= 10:
        recent_10 = list(episode_rewards)[-10:]
        early_10 = list(episode_rewards)[:10]
        improvement = np.mean(recent_10) - np.mean(early_10)
        model.logger.record("convergence/reward_improvement", improvement)
    
    model.logger.dump(step=total_steps)
    
    print(f"[{total_steps}/{TOTAL_TIMESTEPS}] Reward: {mean_reward:.3f}, Length: {mean_length:.3f}" +
          (f", Recon Loss: {np.mean(recon_losses):.4f}" if transformer else ""))

# NEW: Force final logging at last step
model.logger.record("rollout/ep_rew_mean", np.mean(episode_rewards))
model.logger.record("rollout/ep_len_mean", np.mean(episode_lengths))
if transformer:
    model.logger.record("train/recon_loss", np.mean(recon_losses))
model.logger.dump(step=TOTAL_TIMESTEPS)

# Save model
out_name = PLAYGROUND_DIR / RUN_ID
model.save(str(out_name))
print(f"Saved model to {out_name}.zip")

if transformer or deepset:
    encoder = gym_env.encoder
    encoder_path = Path(f"{out_name}.encoder.pth")
    encoder.save_weights(str(encoder_path))
    print(f"Encoder weights saved to: {encoder_path}")

# ========== CONVERGENCE ANALYSIS SUMMARY ==========
print("\n" + "="*70)
print("CONVERGENCE ANALYSIS SUMMARY")
print("="*70)

csv_path = reward_csv_path(RESULTS_DIR / "tuning", "tuning", args)
export_reward_csv(TENSORBOARD_LOG, csv_path)

# Calculate convergence metrics
rewards_arr = np.array(episode_rewards)
if len(rewards_arr) > 20:
    early_phase = rewards_arr[:len(rewards_arr)//4]  # First 25%
    late_phase = rewards_arr[-len(rewards_arr)//4:]  # Last 25%
    
    improvement = np.mean(late_phase) - np.mean(early_phase)
    improvement_pct = (improvement / abs(np.mean(early_phase))) * 100 if np.mean(early_phase) != 0 else 0
    
    print(f"Early reward (first 25%): {np.mean(early_phase):.2f} ± {np.std(early_phase):.2f}")
    print(f"Late reward (last 25%):  {np.mean(late_phase):.2f} ± {np.std(late_phase):.2f}")
    print(f"Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
    print(f"Convergence speed: {'FAST' if improvement > 5 else 'SLOW' if improvement > 0 else 'NO CONVERGENCE'}")

if transformer and recon_losses:
    recon_arr = np.array(recon_losses)
    print(f"\nReconstruction loss: {np.mean(recon_arr):.4f} ± {np.std(recon_arr):.4f}")
    print(f"Loss stability: {'STABLE' if np.std(recon_arr) < 0.1 else 'UNSTABLE'}")

print("="*70)