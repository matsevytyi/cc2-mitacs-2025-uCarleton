import os, sys, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import TransformerWrapper, ChallengeWrapper, PaddingWrapper

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# ========== CONFIGURATION ==========
ALGORITHM = DQN  # Change this to DQN, PPO, etc.
extended = True
transformer = True
pad = False

RUN_ID = "CLS[1,2*64] perhost + SPLIT BACKPROP + save weights"
TOTAL_TIMESTEPS = 500_000

# Hyperparameters per algorithm
HYPERPARAMS = {
    "DQN": {
        "policy": "MlpPolicy",
        "verbose": 2,
        "tensorboard_log": "./logs/",
        "device": "cpu",
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
        "device": "cpu",
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

cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
cyborg.reset()

if transformer:
    gym_env = TransformerWrapper(raw_cyborg=cyborg, agent_name='Blue', max_steps=50)
elif pad:
    gym_env = PaddingWrapper(env=cyborg, agent_name='Blue', max_devices=100, max_steps=50)
else:
    gym_env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=50)
    from gymnasium.wrappers import EnvCompatibility
    gym_env = EnvCompatibility(gym_env)

gym_env.reset()
print(cyborg)

# ========== MODEL INITIALIZATION ==========
algorithm_name = ALGORITHM.__name__
hyperparams = HYPERPARAMS[algorithm_name]

model = ALGORITHM(env=gym_env, **hyperparams)

# ========== TRAINING ==========
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name=f"{algorithm_name}_run_{RUN_ID}",
    log_interval=10
)

# ========== SAVE MODEL ==========
wrapper_type = "transformer" if transformer else "padding" if pad else "challenge"
filename = f"{algorithm_name}_{wrapper_type}_model_{TOTAL_TIMESTEPS}_{RUN_ID}"
model.save(filename)
print(f"Model saved to: {filename}.zip")

if transformer:
    encoder = gym_env.transformer_encoder  # or gym_env.transformer_encoder or similar
    encoder.save_weights(ALGORITHM + "." + RUN_ID + ".encoder.pth")
    print(f"Encoder weights saved to: {filename}.encoder.pth")
