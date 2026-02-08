from gymnasium import Env, spaces
import numpy as np
import csv
from datetime import datetime
import random

from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper, RedTableWrapper, EnumActionWrapper
from scenario_shuffler import churn_hosts

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


class PaddingWrapper(Env, BaseWrapper):
    
    def __init__(self, agent_name: str, env, max_devices=100, agent=None,
            reward_threshold=None, max_steps=None, 
            knowledge_update_mode = "train",
            env_creator=None, yaml_path=None):
        super().__init__(env, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        self.host_order = tuple(env.environment_controller.state.hosts.keys())
        
        self.max_devices = max_devices

        self.knowledge_update_mode = knowledge_update_mode
        
        # Store env reload params
        if env_creator and yaml_path:
            self.env_creator = env_creator
            self.yaml_path = yaml_path
        
        env = table_wrapper(env, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self.env = env

        # following cc2 formula, 2 general actions + 11 actions per potential device
        self.total_actions = 2 + self.max_devices * 11
        self.action_space = spaces.Discrete(self.total_actions)
        
        # Track actions
        self.action_history = []
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_devices * 4,),
            dtype=np.float32
        )
        
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps

        self.episode = 0
        self.step_counter = 0
        self.total_env_step_counter = 0

        self.episode_reward = 0
        self.episode_rewards_list = []
        self.episode_lengths_list = []

    def step(self, action=None):
        if action is not None:
            self.action_history.append(int(action))
        
        if action >= self.env.action_space.n:
            action = self.env.action_space.sample()
        
        obs, reward, terminated, info = self.env.step(action=action)
        obs = self.pad_observation(obs, self.max_devices)

        self.episode_reward += reward
        
        self.step_counter += 1
        self.total_env_step_counter += 1
        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):

        if self.knowledge_update_mode == "tune":
            if self.total_env_step_counter > 150_000:
                self._reload_environment()
                self.total_env_step_counter = 0
        
        self.episode_rewards_list.append(self.episode_reward)
        self.episode_lengths_list.append(self.step_counter)
        self.episode_reward = 0
        self.step_counter = 0

        obs = self.env.reset(**kwargs)
        obs = self.pad_observation(obs, self.max_devices)
        
        # Dump actions to CSV
        if self.action_history:
            csv_dir = "action_logs"
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"actions_{self.agent_name}_HOTRELOAD_x150_extended_Tuning_Padding_DQN.csv")
            
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if os.stat(csv_path).st_size == 0:
                    writer.writerow(['timestamp', 'episode', 'step', 'action_type', 'host_count'])
                timestamp = datetime.now().isoformat()
                for step, action in enumerate(self.action_history):
                    writer.writerow([timestamp, self.episode, step, self.decode_action(action)[1], len(self.host_order)])
        
        self.action_history = []
        self.episode += 1
        
        return obs, {}

    def _reload_environment(self):
        """Reload CybORG environment with dynamic topology"""
        try:
            if self.env_creator is None or self.yaml_path is None:
                return False
        except:
            return False
        
        # Apply churn
        churn_hosts(self.yaml_path)
        
        # Create fresh environment
        fresh_cyborg = self.env_creator(self.yaml_path)

        self.host_order = tuple(fresh_cyborg.environment_controller.state.hosts.keys())
        
        # Recreate wrapper stack
        if self.agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            table_wrapper = RedTableWrapper
        
        env = table_wrapper(fresh_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=self.agent_name, env=env)
        
        self.env = env
        
        return True

    def get_episode_rewards(self):
        """Return list of episode rewards since last call, then clear"""
        rewards = self.episode_rewards_list.copy()
        self.episode_rewards_list.clear()
        return rewards

    def get_episode_lengths(self):
        """Return list of episode lengths since last call, then clear"""
        lengths = self.episode_lengths_list.copy()
        self.episode_lengths_list.clear()
        return lengths

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)
    
    def pad_observation(self, obs, max_devices):
        missing = max_devices * 4 - len(obs)
        if missing > 0:
            left = missing // 2
            right = missing - left
            obs = np.pad(obs, (left, right), constant_values=0)
        return obs
    
     # Total actions = 2 + (N * 11)
# where N is the number of hosts in the network

    # Device-agnostic actions (always indices 0-1)
    SLEEP = 0
    MONITOR = 1

    # Device-specific actions for host i (where i ranges from 0 to N-1)
    def get_action_index(self, host_id, action_type):
        """
        host_id: int from 0 to N-1
        action_type: int from 0 to 10 representing one of the 11 actions
        
        Action types:
        0: Analyze
        1: Remove  
        2: Restore
        3-10: Eight different Decoy services
        """
        return 2 + (host_id * 11) + action_type

    # Inverse mapping: from flat action index to (device, action)
    def decode_action(self, action_idx):

        """
        Convert flat action index to (device_id, action_type) tuple
        
        Returns:
            (device_id, action_type) where:
            - device_id is None for device-agnostic actions, or 0 to num_hosts-1
            - action_type: 0=Sleep, 1=Monitor, or 0-10 for device actions
        """
        if action_idx == 0:
            return (None, 'Sleep')
        elif action_idx == 1:
            return (None, 'Monitor')
        else:

            action_names = ['Analyze', 'Remove', 'Restore', 
                        'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMTP',
                        'DecoySmss', 'DecoySSHD', 'DecoySvchost', 
                        'DecoyTomcat', 'DecoyVsftpd']

            # Device-specific action
            adjusted_idx = action_idx - 2
            device_id = adjusted_idx // 11
            action_type = adjusted_idx % 11
            
            return (device_id, action_names[action_type])
