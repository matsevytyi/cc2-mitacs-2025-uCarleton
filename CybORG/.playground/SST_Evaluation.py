import os, sys, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from CybORG.Agents.Wrappers import TransformerWrapper, ChallengeWrapper

from stable_baselines3 import DQN, PPO
from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from gymnasium.wrappers import EnvCompatibility
#from sb3_contrib import C51


from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from RewardLoggingCallback import RewardLoggingCallback
from MetricLoggingCallback import MetricLoggingCallback


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
    
gym_env = StepAPICompatibility(gym_env, output_truncation_bool=True)
    
gym_env = Monitor(gym_env)
    
gym_env.reset()

print(cyborg)

# ppo/c51 - try

filename = "dqn_transformer_model" if transformer else "dqn_default_model"
model = DQN.load(filename, env=gym_env)

mean_reward, std_reward = evaluate_policy(model, gym_env, n_eval_episodes=10, deterministic=True, return_episode_rewards=True)
print(f"Mean reward: {mean_reward} ± {std_reward}")
