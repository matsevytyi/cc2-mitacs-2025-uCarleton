import os, sys, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from CybORG.Agents.Wrappers import TransformerWrapper

from stable_baselines3 import DQN

from stable_baselines3.common.logger import configure
from RewardLoggingCallback import RewardLoggingCallback


# Load cfg
extended = True

# Locate Scenario2.yaml path
if extended:

    path = "Scenario2.yaml"
    # path = "Scenario2_Linear.yaml"
    # path = "Scenario2_Extended.yaml"

    path = os.path.dirname(os.path.dirname(__file__)) + "/.playground/scenarios/" + path
else:

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
cyborg.reset()

gym_env = TransformerWrapper(env=cyborg, agent_name='Blue')
gym_env.reset()

print(cyborg)

# ppo/c51 - try

# Build model
model = DQN(
    policy='MlpPolicy',
    env=gym_env,
    verbose=2,
    tensorboard_log="./logs/",
    device='cpu',
    buffer_size=100_000
)

callback = RewardLoggingCallback(verbose=1)

log_path = "./logs/DQN_17/"
new_logger = configure(log_path, ["stdout", "tensorboard"])
model.set_logger(new_logger)

model.learn(
    total_timesteps=1000,
    tb_log_name="DQN_17",
    log_interval=10,
    callback=callback  
    )
