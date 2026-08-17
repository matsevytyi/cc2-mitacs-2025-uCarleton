import os, sys, inspect

from torch.cuda import device_of

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import DeepSetsPermInvWrapper, TransformerWrapper, ChallengeWrapper, PaddingWrapper

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# ========== CONFIGURATION ==========
ALGORITHM = PPO  # Change this to DQN, PPO, etc.

mode = "transformer" # "transformer", "ds", or "padding"

transformer = (mode == "transformer")
deepset = (mode == "ds")
pad = (mode == "padding")

method = "TRAIN"

RUN_ID = f"{method}_{mode}_{ALGORITHM.__name__}_vs_RedMeander"
extended = True

TOTAL_TIMESTEPS = 500_000

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters per algorithm
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

# ========== ENVIRONMENT SETUP ==========
if extended:
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".playground/scenarios/Scenario2.yaml"
    )
else:
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
cyborg.reset()

if transformer:
    gym_env = TransformerWrapper(raw_cyborg=cyborg, agent_name='Blue', max_steps=100, device=device, max_actions=240)
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
    log_interval=10
)

# ========== SAVE MODEL ==========
wrapper_type = "transformer" if transformer else "padding" if pad else "challenge"
model.save(RUN_ID)
print(f"Model saved to: {RUN_ID}.zip")

if transformer or deepset:
    encoder = gym_env.encoder  # or gym_env.transformer_encoder or similar
    encoder.save_weights(RUN_ID + ".encoder.pth")
    print(f"Encoder weights saved to: {RUN_ID}.encoder.pth")

