# ========== TRAINING PIPELINE (updated) ==========

import os, sys, inspect
import numpy as np
import torch
import random
from collections import deque

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import TransformerWrapper, PaddingWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure

# ========== CONFIGURATION ==========

transformer = True

ALGORITHM = DQN
algorithm_name = ALGORITHM.__name__
USE_PRETRAINED = True
MODEL_PATH = f"{ALGORITHM.__name__}_{'Transformer' if transformer else 'Padding'}_training_for_dynamic"
TOTAL_TIMESTEPS = 450_000
RUN_ID = f"{ALGORITHM.__name__}_{'Transformer' if transformer else 'Padding'}_tuning_x150_extended_dynamic_topology"
TENSORBOARD_LOG = f"./logs/{RUN_ID}"

device = 'cpu'

base_dir = os.path.dirname(os.path.dirname(__file__))
scenario_name = f"Scenario2_{'Transformer' if transformer else 'Padding'}_{algorithm_name}"
SCENARIO_PATH = os.path.join(base_dir, f".playground/scenarios/{scenario_name}.yaml")

HYPERPARAMS = {
    "DQN": {
        "policy": "MlpPolicy",
        "verbose": 2,
        "tensorboard_log": f"./logs/{RUN_ID}",
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
        "tensorboard_log": f"./logs/{RUN_ID}",
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

def create_cyborg_env(yaml_path: str):
    cyborg = CybORG(yaml_path, 'sim', agents={'Red': RedMeanderAgent})
    cyborg.reset()
    return cyborg

# ========== LOGGING ===========

import sys

outname = f"{RUN_ID}_console_log.txt"
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

sys.stdout = Tee(sys.__stdout__, open(outname, 'w', buffering=1))

# ========== TRAINING PIPELINE ==========

initial_raw_cyborg = create_cyborg_env(SCENARIO_PATH)

if transformer:
    gym_env = TransformerWrapper(
        agent_name='Blue',
        raw_cyborg=initial_raw_cyborg,
        max_steps=100,
        knowledge_update_mode="tune",
        env_creator=create_cyborg_env,
        yaml_path=SCENARIO_PATH,
        max_actions=240,
        weights_path=f"{MODEL_PATH}.encoder.pth"
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
    model = ALGORITHM.load(MODEL_PATH, env=gym_env, tensorboard_log=TENSORBOARD_LOG)
else:
    hyperparams = HYPERPARAMS[algorithm_name]
    model = ALGORITHM(env=gym_env, **hyperparams)

# ========== MANUAL LOGGING FOR CONVERGENCE ANALYSIS ==========
# Configure TensorBoard logger
logger = configure(folder=TENSORBOARD_LOG, format_strings=["tensorboard", "stdout"])
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
    model.logger.record("train/recon_loss", gym_env.transformer_encoder.recon_loss.item())
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
        recon_loss = gym_env.transformer_encoder.recon_loss
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
out_name = f"{RUN_ID}"
model.save(out_name)
print(f"Saved model to {out_name}.zip")

if transformer:
    encoder = gym_env.transformer_encoder
    encoder.save_weights(out_name + ".encoder.pth")
    print(f"Encoder weights saved to: {out_name}.encoder.pth")

# ========== CONVERGENCE ANALYSIS SUMMARY ==========
print("\n" + "="*70)
print("CONVERGENCE ANALYSIS SUMMARY")
print("="*70)

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
