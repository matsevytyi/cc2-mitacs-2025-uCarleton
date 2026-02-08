# ========== TRAINING PIPELINE (updated) ==========

import os, sys, inspect
import numpy as np
import torch
import random

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import TransformerWrapper, PaddingWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure

from stable_baselines3.common.evaluation import evaluate_policy

# ========== CONFIGURATION ==========

transformer = True

ALGORITHM = PPO
algorithm_name = ALGORITHM.__name__
TOTAL_TIMESTEPS = 500_000 if transformer else 100_000

WEIGHTS_PATH = f"{algorithm_name}_Transformer_tuning_x150_extended_dynamic_topology.encoder.pth"

TENSORBOARD_LOG = "./logs/"

RUN_ID = "CLS[1,2*64] perhost + SPLIT BACKPROP + save weights" if transformer else ""
TOTAL_TIMESTEPS = 500_000

device = 'cpu'

base_dir = os.path.dirname(os.path.dirname(__file__))
scenario_name = f"Scenario2_{'Transformer' if transformer else 'Padding'}_{algorithm_name}"
SCENARIO_PATH = os.path.join(base_dir, f".playground/scenarios/{scenario_name}.yaml")

HYPERPARAMS = {
    "DQN": {
        "policy": "MlpPolicy",
        "verbose": 2,
        "tensorboard_log": "./logs/",
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
        "tensorboard_log": "./logs/",
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

def evaluate_model(algorithm_class, filename, gym_env, env_name):
    """Evaluate a model on a given environment"""
    try:
        model = algorithm_class.load(filename, env=gym_env)
        mean_reward, std_reward = evaluate_policy(
            model, gym_env, n_eval_episodes=10, 
            deterministic=True, return_episode_rewards=True
        )
        
        print(f"\n{algorithm_class.__name__} on {env_name}:")
        print(f"  Mean reward: {0.5 * np.max(mean_reward) + 0.5 * np.min(mean_reward):.2f}")
        print(f"  Max reward:  {np.max(mean_reward):.2f}")
        print(f"  Min reward:  {np.min(mean_reward):.2f}")
        print(f"  Std reward:  {np.mean(std_reward):.2f}")
        
        return {
            'algorithm': algorithm_class.__name__,
            'env': env_name,
            'mean': 0.5 * np.max(mean_reward) + 0.5 * np.min(mean_reward),
            'max': np.max(mean_reward),
            'min': np.min(mean_reward),
            'std': np.mean(std_reward)
        }
    except Exception as e:
        print(f"Error evaluating {algorithm_class.__name__} on {env_name}: {e}")
        return None

# ========== ENVIRONMENT CREATION FUNCTION ==========
# This is the KEY: Define as a separate function
def create_cyborg_env(yaml_path: str):
    """
    Create fresh CybORG environment from YAML
    This will be called every reset() to reload topology
    """
    cyborg = CybORG(yaml_path, 'sim', agents={'Red': B_lineAgent})
    cyborg.reset()
    return cyborg


# ========== TRAINING PIPELINE ==========

# 1) Create initial wrapper with env_creator function passed in

initial_raw_cyborg = create_cyborg_env(SCENARIO_PATH)

if transformer:
    gym_env = TransformerWrapper(
        agent_name='Blue',
        raw_cyborg=initial_raw_cyborg,
        max_steps=100,
        knowledge_update_mode = "tune",
        env_creator=create_cyborg_env,  # NEW: Pass function reference
        yaml_path=SCENARIO_PATH,  
        max_actions=240,       # NEW: Pass YAML path
        weights_path=WEIGHTS_PATH
    )
else:
    gym_env = PaddingWrapper(
        agent_name='Blue',
        env=initial_raw_cyborg,
        knowledge_update_mode = "tune",
        max_steps=100,
        env_creator=create_cyborg_env,
        yaml_path=SCENARIO_PATH,  
    )

gym_env.reset()

# 2) Create and train model

# result = evaluate_model(PPO, 'PPO_Padding_dynamic_topology.zip', gym_env, "Modifying Env")
# print(result)
# result = evaluate_model(DQN, 'DQN_Padding_dynamic_topology.zip', gym_env, "Modifying Env")
# print(result)

result = evaluate_model(PPO, 'PPO_Transformer_tuning_x150_extended_dynamic_topology.zip', gym_env, "Modifying Env")
print(result)
result = evaluate_model(DQN, 'DQN_Transformer_tuning_x150_extended_dynamic_topology.zip', gym_env, "Modifying Env")
print(result)