import os, sys, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from CybORG.Agents.Wrappers import TransformerWrapper, ChallengeWrapper

from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from gymnasium.wrappers.compatibility import LegacyEnv
from gymnasium.wrappers import EnvCompatibility


from stable_baselines3 import DQN, PPO
#from sb3_contrib import C51


from stable_baselines3.common.logger import configure
from RewardLoggingCallback import RewardLoggingCallback
from MetricLoggingCallback import MetricLoggingCallback
from stable_baselines3.common.monitor import Monitor


# Load cfg
extended = True
transformer = False

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


if transformer:
    gym_env = TransformerWrapper(env=cyborg, agent_name='Blue', max_steps=100)
else:
    gym_env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=100)
    gym_env = EnvCompatibility(gym_env)
    #gym_env = StepAPICompatibility(gym_env, output_truncation_bool=True)
    
gym_env = StepAPICompatibility(gym_env, output_truncation_bool=True)
    
gym_env.reset()

print(cyborg)

# ppo/c51 - try

model1 = DQN(
    policy='MlpPolicy',
    env=gym_env,
    verbose=2,
    tensorboard_log="./logs/",
    device='cpu',
    buffer_size=100_000
)

callback = RewardLoggingCallback(verbose=1)
# callback = MetricLoggingCallback(verbose=1)

log_path = "./logs/"
new_logger = configure(log_path, ["stdout", "tensorboard"])
model1.set_logger(new_logger)

model1.learn(
    total_timesteps=100000,
    tb_log_name="DQN",
    log_interval=10,
    callback=callback  
    )

filename = "dqn_transformer_model_100000" if transformer else "dqn_default_model_100000"
model1.save(filename)

#print("Metrics:", callback.get_metrics())