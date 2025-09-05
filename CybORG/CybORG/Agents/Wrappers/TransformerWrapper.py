from gymnasium import Env, spaces
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Dict, Discrete, MultiBinary, Box
import gym

from CybORG.Agents.Wrappers.TransformerStateEncoder import TransformerStateEncoder

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Note: basically it is modified challengewrapper to use encoder
class TransformerWrapper(Env,BaseWrapper):
    def __init__(self, agent_name: str, raw_cyborg, agent=None,
            reward_threshold=None, max_steps = None, device='cpu'):
        super().__init__(raw_cyborg, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        env = table_wrapper(raw_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self.env = env
        self.action_space = self.env.action_space

        self.device = device

        embedding_dim = 64 # transformer embedding dimension
        observation_shape_size = embedding_dim * 2 # or *1 depending on the architecture

        self.transformer_encoder = TransformerStateEncoder(
            observation_space=None,  # Not needed inside encoder since I don't use it, may add that later (may be explicitely passed too)
            embedding_dim=embedding_dim,
            cyborg_env=raw_cyborg,
            agent_name=agent_name
        ).to(self.device)
        

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_shape_size, ),  # embedding_dim from transformer encoder
            dtype=np.float32
        )

        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

    def step(self,action=None, debug=False):
        obs, reward, terminated, info = self.env.step(action=action)
    
        self.step_counter += 1
        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True
        
        #print(f"Step {self.step_counter}: action={action}, reward={reward}, done={terminated or truncated}")
            
        if debug:    
            if "episode_reward" not in info:
                info["episode_reward"] = 0.0
                info["episode_length"] = 0

            info["episode_reward"] += reward
            info["episode_length"] += 1

            if terminated:
                info["episode"] = {
                    "r": info["episode_reward"],
                    "l": info["episode_length"]
                }
            
            #print(info)

        with torch.no_grad():
            encoded_obs = self.transformer_encoder(None)

        return encoded_obs.cpu().numpy(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_counter = 0
        _ = self.env.reset(**kwargs)

        with torch.no_grad():
            encoded_obs = self.transformer_encoder(None)  # env is fetched inside encoder
        return encoded_obs.cpu().numpy(), {}

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)