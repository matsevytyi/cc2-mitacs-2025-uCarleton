from types import NoneType
from gymnasium import Env, spaces
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper, RedTableWrapper, EnumActionWrapper
from scenario_shuffler import churn_hosts

import numpy as np

import csv
from datetime import datetime

from CybORG.Agents.Wrappers.TransformerStateEncoder import TransformerStateEncoder
from CybORG.Agents.Wrappers.TransformerStateEncoder import TransformerStateEncoderV2

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# Note: modified challengewrapper to use encoder
class TransformerWrapper(Env, BaseWrapper):
    def __init__(self, agent_name: str, raw_cyborg, agent=None,
            reward_threshold=None, max_steps=None, max_actions=None,
            action_space_mode="default", knowledge_update_mode="train",
            env_creator=None, yaml_path=None,
            device='cpu', version="ip_local", weights_path=None, top_k=8):
        super().__init__(raw_cyborg, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        self.knowledge_update_mode = knowledge_update_mode
        self.version = version
        self.raw_cyborg = raw_cyborg
        self.device = device
        self.top_k = top_k
        self.max_actions = max_actions
        self.action_space_mode = action_space_mode  # "default"/"topk", "pad", or "cutoff"
        self.use_topk = (action_space_mode in ("default", "topk")) and bool(top_k)

        self.host_order = tuple(self.raw_cyborg.environment_controller.state.hosts.keys())  # to freeze the order

        # upscale the wrapper similarly to challenge wrapper
        env = table_wrapper(raw_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self.table_env = table_wrapper(raw_cyborg, output_mode='table')
        self.env = env

        self.action_history = []
        self.recon_loss_history = []

        if env_creator and yaml_path:
            self.env_creator = env_creator
            self.yaml_path = yaml_path
            
        embedding_dim = 64  # transformer embedding dimension
        token_dim = embedding_dim * 2  # 64D obs + 64D IP
        
        if self.use_topk:
            observation_shape_size = (1 + self.top_k) * token_dim
            
            # 2 global actions (Sleep, Monitor) + 11 host actions per candidate slot
            self.action_space = spaces.Discrete(2 + self.top_k * 11)
        elif self.max_actions is not None:
            self.action_space = spaces.Discrete(int(self.max_actions))
            observation_shape_size = token_dim
        else:
            self.action_space = self.env.action_space
            observation_shape_size = token_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_shape_size, ),  # embedding_dim from transformer encoder
            dtype=np.float32
        )

        self.encoder = TransformerStateEncoderV2(
            observation_space=self.observation_space,
            embedding_dim=embedding_dim,
            initial_host_count=len(self.host_order),
            mode=self.knowledge_update_mode,
            top_k=self.top_k,
            use_candidate_retrieval=self.use_topk
        ).to(self.device)

        self.reward_threshold = reward_threshold
        self.max_steps = max_steps

        self.episode = 0
        self.step_counter = 0
        self.total_env_step_counter = 0

        self.episode_reward = 0
        self.episode_rewards_list = []
        self.episode_lengths_list = []

        if weights_path:
            self.encoder.load_weights(weights_path)

    def _resolve_action(self, action):
        """Translate wrapper action (Top-K or padded/cutoff) into a valid CybORG action index."""
        if action is None:
            return None
        action = int(action)

        if self.use_topk:
            if action < 2 or not self.encoder.current_candidate_hosts:
                return action  # 0: Sleep, 1: Monitor
            slot_idx = min((action - 2) // 11, len(self.encoder.current_candidate_hosts) - 1)
            action_type = (action - 2) % 11
            target_host = self.encoder.current_candidate_hosts[slot_idx]
            if self.encoder.focus_mode == "stateful":
                self.encoder.focus_host = target_host
            host_id = self.host_order.index(target_host)
            return self.get_action_index(host_id, action_type)

        n_valid = self.env.action_space.n
        if action >= n_valid:
            return (n_valid - 1) if self.action_space_mode == "cutoff" else self.env.action_space.sample()
        return action

    def step(self, action=None, debug=False, verbose=False):
        action = self._resolve_action(action)

        self.action_history.append(self.decode_action(action)[1])
        self.recon_loss_history.append(float(self.encoder.recon_loss))

        obs, reward, terminated, info = self.env.step(action=action)

        self.episode_reward += reward

        # self.env.env.env.env.env is same as active self.raw_cyborg
        # enrich obs with other contextual information
        obs = self.extract_host_state(raw_cyborg=self.env.env.env.env.env, obs=obs)
        if verbose:
            print("action", action, self.decode_action(action))
            print('step obs', obs)

        self.step_counter += 1
        self.total_env_step_counter += 1
        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True

        encoded_obs, _ = self.encoder(
            obs, self.host_order, version=self.version, mode=self.knowledge_update_mode
        )

        return encoded_obs.squeeze(0).detach().cpu().numpy(), reward, terminated, truncated, info

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

        # Reset focus before encoding initial state so Step 0 populates candidate hosts
        self.encoder.reset_focus()

        # enrich obs with other contextual information
        obs = self.extract_host_state(raw_cyborg=self.env.env.env.env.env, obs=obs)

        encoded_obs, _ = self.encoder(
            obs, self.host_order, version=self.version, mode=self.knowledge_update_mode
        )

        ## update csv with actions from self.action_history and set self.action_history to []
        if self.action_history:
            csv_dir = "action_logs"
            os.makedirs(csv_dir, exist_ok=True)

            csv_path = os.path.join(
                csv_dir,
                f"actions_V2_{self.agent_name}_HOTRELOAD_x150_extended_Tuning_Transformer_RedMeander_PPO.csv"
            )

            with open(csv_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if os.stat(csv_path).st_size == 0:
                    writer.writerow(['timestamp', 'episode', 'step', 'action_type', 'host_count', 'recon_loss'])
                timestamp = datetime.now().isoformat()
                for step, action in enumerate(self.action_history):
                    writer.writerow([
                        timestamp, getattr(self, 'episode', None), step,
                        action, len(self.host_order), self.recon_loss_history[step]
                    ])

        self.episode += 1

        # Clear history after save
        self.action_history = []
        self.recon_loss_history = []

        return encoded_obs.squeeze(0).detach().cpu().numpy(), {}

    def _reload_environment(self, verbose=False):
        """
        INTERNAL: Reload CybORG environment from scratch
        Called on reset() to pick up YAML changes
        """
        try:
            if self.env_creator is None or self.yaml_path is None:
                print("Reload Ignored")
                return False
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to reload environment: {e}")
            return False

        # churn devices
        churn_hosts(self.yaml_path)

        # create fresh CybORG instance
        fresh_cyborg = self.env_creator(self.yaml_path)
        self.raw_cyborg = fresh_cyborg

        # Recreate wrapper stack
        if self.agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            table_wrapper = RedTableWrapper

        env = table_wrapper(fresh_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=self.agent_name, env=env)

        self.env = env
        self.table_env = table_wrapper(fresh_cyborg, output_mode='table')

        # Update host order (may have changed)
        self.host_order = tuple(self.raw_cyborg.environment_controller.state.hosts.keys())

        # dump dynamically old weights
        old_weights = {
            'transformer': self.encoder.transformer.state_dict(),
            'obs_embed': self.encoder.obs_embed.state_dict(),
            'ip_byte_embed': self.encoder.ip_byte_embed.state_dict(),
            'cls_token': self.encoder.cls_token.data,
            'token_head_from_cls': self.encoder.token_head_from_cls.state_dict(),
        }

        # Reinitialize encoder with new host count
        self.encoder = TransformerStateEncoderV2(
            observation_space=self.observation_space,
            embedding_dim=64,
            initial_host_count=len(self.host_order),
            mode=self.knowledge_update_mode,
            top_k=self.top_k,
            use_candidate_retrieval=self.use_topk
        ).to(self.device)

        self.encoder.load_weights_from_dict(old_weights)

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

    def extract_host_state(self, raw_cyborg, obs):
        hosts_dict = {}
        for i, hname in enumerate(self.host_order):
            hstate = raw_cyborg.environment_controller.state.hosts[hname]

            obs_chunk = obs[4 * i : 4 * (i + 1)]  # original signal

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
            action_names = [
                'Analyze', 'Remove', 'Restore',
                'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMTP',
                'DecoySmss', 'DecoySSHD', 'DecoySvchost',
                'DecoyTomcat', 'DecoyVsftpd'
            ]

            # Device-specific action
            adjusted_idx = action_idx - 2
            device_id = adjusted_idx // 11
            action_type = adjusted_idx % 11

            return (device_id, action_names[action_type])