import os, sys, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from CybORG.Agents.Wrappers import PaddingWrapper, TransformerWrapper, ChallengeWrapper

from stable_baselines3 import DQN, PPO
from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from gymnasium.wrappers import EnvCompatibility
#from sb3_contrib import C51


from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import numpy as np

#from RewardLoggingCallback import RewardLoggingCallback
#from MetricLoggingCallback import MetricLoggingCallback


# Load cfg
extended = True
transformer = False

# Locate Scenario2.yaml path
if extended:

    path_1 = "Scenario2.yaml"
    path_2 = "Scenario2_more_hosts.yaml"
    path_3 = "Scenario2_4th_subnet.yaml"
    # path = "Scenario2_Linear.yaml"
    # path = "Scenario2_Extended.yaml"

    path_1 = os.path.dirname(os.path.dirname(__file__)) + "/.playground/scenarios/" + path_1
    path_2 = os.path.dirname(os.path.dirname(__file__)) + "/.playground/scenarios/" + path_2
    path_3 = os.path.dirname(os.path.dirname(__file__)) + "/.playground/scenarios/" + path_3
else:

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

cyborg_1 = CybORG(path_1, 'sim', agents={'Red': B_lineAgent})
cyborg_1.reset()

cyborg_2 = CybORG(path_2, 'sim', agents={'Red': B_lineAgent})
cyborg_2.reset()

#cyborg_3 = CybORG(path_3, 'sim', agents={'Red': B_lineAgent})
#cyborg_3.reset()

if transformer:
    gym_env_1 = TransformerWrapper(raw_cyborg=cyborg_1, agent_name='Blue', max_steps=50)
    gym_env_2 = TransformerWrapper(raw_cyborg=cyborg_2, agent_name='Blue', max_steps=50)
    #gym_env_3 = TransformerWrapper(raw_cyborg=cyborg_3, agent_name='Blue', max_steps=50)
else:
    gym_env_1 = PaddingWrapper(env=cyborg_1, agent_name='Blue', max_devices=100, max_steps=50)
    gym_env_2 = PaddingWrapper(env=cyborg_2, agent_name='Blue', max_devices=100, max_steps=50)
    #gym_env_3 = PaddingWrapper(env=cyborg_3, agent_name='Blue', max_devices=100, max_steps=50)

    
#gym_env = StepAPICompatibility(gym_env, output_truncation_bool=True)
    
#gym_env = Monitor(gym_env)
    
gym_env_1.reset()
gym_env_2.reset()
#gym_env_3.reset()

# ppo/c51 - try

filename = "dqn_transformer_model" if transformer else "dqn_padding_model_100000"

model = DQN.load(filename, env=gym_env_1)
mean_reward, std_reward = evaluate_policy(model, gym_env_1, n_eval_episodes=10, deterministic=True, return_episode_rewards=True)
print(f"Mean/max/min reward for training env: {np.mean(mean_reward)}/{np.max(mean_reward)}/{np.min(mean_reward)}")

# data, params = DQN.load(filename, env=None, device="cpu", print_system_info=True, _load_data=True)

# # # force action space to current env
# data["action_space"] = gym_env_2.action_space

# reload using patched data
model = DQN.load(
    filename,
    env=gym_env_2,
    custom_objects={"action_space": gym_env_2.action_space},
    device="cpu"
)

# Fix the output layers to match new env
with torch.no_grad():
    old_weight = model.policy.q_net.q_net[-1].weight.data
    old_bias = model.policy.q_net.q_net[-1].bias.data

    model.q_net.q_net[-1].weight.data[:old_weight.shape[0]] = old_weight
    model.q_net.q_net[-1].bias.data[:old_bias.shape[0]] = old_bias

mean_reward, std_reward = evaluate_policy(model, gym_env_2, n_eval_episodes=10, deterministic=True, return_episode_rewards=True)
print(f"Mean/max/min reward for env with more hosts: {np.mean(mean_reward)}/{np.max(mean_reward)}/{np.min(mean_reward)}")

# model = DQN.load(filename, env=gym_env_3)
# mean_reward, std_reward = evaluate_policy(model, gym_env_3, n_eval_episodes=10, deterministic=True, return_episode_rewards=True)
# print(f"Mean/max/min reward for env with more subnets: {np.mean(mean_reward)}/{np.max(mean_reward)}/{np.min(mean_reward)}")