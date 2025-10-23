from gymnasium import Env, spaces
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stable_baselines3 import DQN

from CybORG.Agents.Wrappers.TransformerStateEncoder import TransformerStateEncoder

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Note: modified challengewrapper to use encoder
class TransformerWrapper(Env,BaseWrapper):
    def __init__(self, agent_name: str, raw_cyborg, agent=None,
            reward_threshold=None, max_steps = None, max_actions=None, action_space_mode="pad", device='cpu', version="ip_local"):
        super().__init__(raw_cyborg, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')
        
        self.version = version
        self.raw_cyborg = raw_cyborg
        
        self.host_order = tuple(self.raw_cyborg.environment_controller.state.hosts.keys()) # to freeze the order

        # upscale the wrapper similarly to challenge wrapper
        env = table_wrapper(raw_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)
        
        self.table_env = table_wrapper(raw_cyborg, output_mode='table')

        self.env = env
        self.action_space = self.env.action_space

        self.device = device

        embedding_dim = 64 # transformer embedding dimension
        observation_shape_size = embedding_dim*2 # or *1 depending on the architecture

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_shape_size, ),  # embedding_dim from transformer encoder
            dtype=np.float32
        )

        self.transformer_encoder = TransformerStateEncoder(
            observation_space=self.observation_space,
            embedding_dim=embedding_dim,
            initial_host_count=len(self.host_order)
        ).to(self.device)
        

        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None
        
        # fixed action space padding/cutoff
        self.max_actions = max_actions
        self.action_space_mode = action_space_mode  # "pad" or "cutoff"
        if self.max_actions is not None:
            self.action_space = spaces.Discrete(int(self.max_actions))

    def step(self,action=None, debug=False, verbose=False):
        
        # Map out-of-range actions to a valid one based on the selected mode
        if action is not None:
            try:
                n_valid = int(self.env.action_space.n)
            except Exception:
                n_valid = None
            if self.max_actions is not None and n_valid is not None and n_valid > 0:
                if action >= n_valid:
                    if self.action_space_mode == "cutoff":
                        action = n_valid - 1
                    else:
                        action = self.env.action_space.sample()
                        
        obs, reward, terminated, info = self.env.step(action=action)
        
        # self.env.env.env.env.env is same as active self.raw_cyborg

        # enrich obs with other contextual information
        obs = self.extract_host_state(raw_cyborg=self.env.env.env.env.env, obs=obs)
        if verbose:
            print('step obs', obs)
    
        self.step_counter += 1
        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True
        
        # with torch.no_grad():
        #     encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)
        encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)

        return encoded_obs.detach().cpu().numpy(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_counter = 0
        obs = self.env.reset(**kwargs)
        
        # enrich obs with other contextual information
        obs = self.extract_host_state(raw_cyborg=self.env.env.env.env.env, obs=obs)

        # with torch.no_grad():
        #     encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)
        encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)
            
        return encoded_obs.detach().cpu().numpy(), {}

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

    def extract_host_state(self, raw_cyborg, obs):
        hosts_dict = {}
        for i, hname in enumerate(self.host_order):
            hstate = raw_cyborg.environment_controller.state.hosts[hname]
            
            obs_chunk = obs[4 * i : 4 * (i + 1)] # original signal
            
            ips = [
                iface.ip_address.__str__()
                for iface in hstate.interfaces
                if iface.name.startswith("eth")
            ]
            
            subnets = [
                iface.subnet.__str__().split("/")[0]
                for iface in hstate.interfaces
                if iface.name.startswith("eth")
            ]
            
            ports = np.array([
                conn.get("local_port")
                for proc in hstate.processes
                for conn in proc.connections
            ])
            
            processes = np.array([proc.name for proc in hstate.processes])
            
            hosts_dict[hname] = {
                "obs": obs_chunk,
                "ips": ips,
                "subnets": subnets,
                "ports": ports,
                "processes": processes,
            }
            
        return hosts_dict