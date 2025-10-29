# ========== TRAINING PIPELINE (updated) ==========

import os, sys, inspect
import numpy as np
import torch
import random

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import TransformerWrapper, PaddingWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure

# ========== CONFIGURATION ==========

transformer = True

ALGORITHM = DQN
algorithm_name = ALGORITHM.__name__
USE_PRETRAINED = False
MODEL_PATH = "DQN_transformer_model.zip"
TOTAL_TIMESTEPS = 500_000
RUN_ID = f"{ALGORITHM.__name__}_{'Transformer' if transformer else 'Padding'}_dynamic_topology"
TENSORBOARD_LOG = "./logs/"

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


# ========== ENVIRONMENT CREATION FUNCTION ==========
# This is the KEY: Define as a separate function
def create_cyborg_env(yaml_path: str):
    """
    Create fresh CybORG environment from YAML
    This will be called every reset() to reload topology
    """
    cyborg = CybORG(yaml_path, 'sim', agents={'Red': RedMeanderAgent})
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
        env_creator=create_cyborg_env,  # NEW: Pass function reference
        yaml_path=SCENARIO_PATH,  
        max_actions=240,       # NEW: Pass YAML path
    )
else:
    gym_env = PaddingWrapper(
        agent_name='Blue',
        env=initial_raw_cyborg,
        max_steps=100,
        env_creator=create_cyborg_env,
        yaml_path=SCENARIO_PATH,  
    )

gym_env.reset()

# 2) Create and train model

if USE_PRETRAINED:
    model = ALGORITHM.load(MODEL_PATH, env=gym_env, tensorboard_log=TENSORBOARD_LOG)
else:
    hyperparams = HYPERPARAMS[algorithm_name]

    model = ALGORITHM(env=gym_env, **hyperparams)


# 4) Train
print(f"Starting training with {'DYNAMIC' if True else 'STATIC'} topology")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name=f"{RUN_ID}",
    log_interval=10
)

# 5) Save model
out_name = f"{RUN_ID}"
model.save(out_name)
print(f"Saved model to {out_name}.zip")

if transformer:
    encoder = gym_env.transformer_encoder  # or gym_env.transformer_encoder or similar
    encoder.save_weights(out_name + ".encoder.pth")
    print(f"Encoder weights saved to: {out_name}.encoder.pth")