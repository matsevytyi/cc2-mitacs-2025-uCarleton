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
            reward_threshold=None, max_steps = None, device='cpu', version="ip_local"):
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

        self.transformer_encoder = TransformerStateEncoder(
            observation_space=None,
            embedding_dim=embedding_dim
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

    def step(self,action=None, debug=False, verbose=False):
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
        
        with torch.no_grad():
            encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)

        return encoded_obs.cpu().numpy(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_counter = 0
        obs = self.env.reset(**kwargs)
        
        # enrich obs with other contextual information
        obs = self.extract_host_state(raw_cyborg=self.env.env.env.env.env, obs=obs)

        with torch.no_grad():
            encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)
            
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

# gt_hosts = self.raw_cyborg.environment_controller.state.hosts

        # for hname, hstate in gt_hosts.items():
            # Interfaces / IPs
            # for iface in hstate.interfaces:
            #     print(f"[{hname}] IP: {iface.get_state()}")

            # # Processes and open ports
            # for proc in hstate.processes:
            #     if proc.connections:
            #         pass
            #         for conn in proc.connections:
            #             print(f"[{hname}] PID {proc} listening on :{conn.get('local_address')}:{conn.get('local_port')} -> connected to :{conn.get('local_address')}:{conn.get('local_port')}")
            #     else:
            #         print(f"[{hname}] PID {proc}, User: {proc}")
            
            # Interfaces (eth only)
        
        # # IPs
        # ips = [
        #     iface.ip_address
        #     for hstate in self.raw_cyborg.environment_controller.state.hosts.values()
        #     for iface in hstate.interfaces
        #     if iface.name.startswith("eth")
        # ]

        # # Ports
        # ports = [
        #     conn.get('local_port')
        #     for hstate in self.raw_cyborg.environment_controller.state.hosts.values()
        #     for proc in hstate.processes
        #     for conn in proc.connections
        # ]

        # # Processes
        # processes = [
        #     proc.name
        #     for hstate in self.raw_cyborg.environment_controller.state.hosts.values()
        #     for proc in hstate.processes
        # ]
        
        # if verbose:
        #     print("IPs:", ips)
        #     print("Ports:", ports)
        #     print("Processes:", processes)