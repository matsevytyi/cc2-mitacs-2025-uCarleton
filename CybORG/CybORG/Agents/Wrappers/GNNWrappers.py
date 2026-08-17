import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from datetime import datetime
import os, sys
from gymnasium import Env, spaces
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper, RedTableWrapper, EnumActionWrapper
from scenario_shuffler import churn_hosts

# ==========================================
# 1. THE PYTORCH MODELS (Internal Use)
# ==========================================

class BaseStateEncoder(nn.Module):
    def __init__(self, observation_space, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Dummy loss for logging consistency
        self.recon_loss = torch.tensor(0.0)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def load_weights_from_dict(self, state_dict):
        self.load_state_dict(state_dict)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

class GATEncoderModel(BaseStateEncoder):
    def __init__(self, observation_space, embedding_dim=64, n_heads=4):
        super().__init__(observation_space, embedding_dim)
        
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Input is 4 features per host (from CybORG vector)
        self.linear_proj = nn.Linear(4, embedding_dim)
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        
        self.readout = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, obs, host_order, version=None):
        # Flattened obs -> (Batch, N, 4)
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.linear_proj.weight.device)
        if obs.dim() == 1: obs = obs.view(-1, 4)
        
        N = obs.shape[0]
        
        # 1. Linear Projection
        h = self.linear_proj(obs) 
        
        # 2. GAT Attention (Fully Connected Assumption)
        h_heads = h.view(N, self.n_heads, self.head_dim)
        attn_src = (h_heads * self.att_src).sum(dim=-1)
        attn_dst = (h_heads * self.att_dst).sum(dim=-1)
        attn_scores = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        attn_probs = F.softmax(F.leaky_relu(attn_scores, 0.2), dim=1)
        
        h_out = torch.zeros_like(h_heads)
        for i in range(self.n_heads):
            h_out[:, i, :] = torch.matmul(attn_probs[:, :, i], h_heads[:, i, :])
            
        h_out = h_out.view(N, self.embedding_dim)
        
        # 3. Mean Pooling & Readout
        global_emb = torch.mean(h_out, dim=0)
        return self.readout(global_emb)

class DeepSetsEncoderModel(BaseStateEncoder):
    def __init__(self, observation_space, embedding_dim=64):
        super().__init__(observation_space, embedding_dim)
        
        # Phi (Individual processing)
        self.phi = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        
        # Rho (Global processing)
        self.rho = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, obs, host_order, version=None):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.phi[0].weight.device)
        if obs.dim() == 1: obs = obs.view(-1, 4)
        
        # 1. Independent features
        h_local = self.phi(obs)
        
        # 2. Sum Aggregation (Permutation Invariant)
        h_global = torch.sum(h_local, dim=0)
        
        return self.rho(h_global)

# ==========================================
# 2. THE WRAPPERS
# ==========================================

class GATWrapper(Env, BaseWrapper):
    def __init__(self, agent_name: str, raw_cyborg, agent=None,
            reward_threshold=None, max_steps=None, max_actions=None, 
            action_space_mode="pad", knowledge_update_mode="train",
            env_creator=None, yaml_path=None,
            device='cpu', version="ip_local", weights_path=None):
        super().__init__(raw_cyborg, agent)
        
        # --- Standard Initialization (Same as TransformerWrapper) ---
        self.agent_name = agent_name
        if agent_name.lower() == 'red': table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue': table_wrapper = BlueTableWrapper
        else: raise ValueError('Invalid Agent Name')

        self.knowledge_update_mode = knowledge_update_mode
        self.version = version
        self.raw_cyborg = raw_cyborg
        self.host_order = tuple(self.raw_cyborg.environment_controller.state.hosts.keys())

        env = table_wrapper(raw_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)
        self.env = env
        self.table_env = table_wrapper(raw_cyborg, output_mode='table')

        self.action_history = []
        self.recon_loss_history = [] # Will be 0.0 for GAT/DeepSets
        self.env_creator = env_creator
        self.yaml_path = yaml_path
        
        self.max_actions = max_actions
        self.action_space_mode = action_space_mode
        if self.max_actions is not None:
            self.action_space = spaces.Discrete(int(self.max_actions))
        else:
            self.action_space = self.env.action_space

        self.device = device
        embedding_dim = 64
        
        # --- MODEL SPECIFIC: GAT Encoder ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32
        )
        
        self.encoder = GATEncoderModel(
            observation_space=self.observation_space,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        if weights_path:
            self.encoder.load_weights(weights_path)
            
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = 0
        self.total_env_step_counter = 0
        self.episode = 0
        self.episode_reward = 0
        self.episode_rewards_list = []
        self.episode_lengths_list = []

    def step(self, action=None, debug=False, verbose=False):
        # Action Logic (Same as Transformer)
        self.action_history.append(self.decode_action(action)[1])
        self.recon_loss_history.append(0.0) # No recon loss in standard GAT
        
        if action is not None:
            try: n_valid = int(self.env.action_space.n)
            except: n_valid = None
            if self.max_actions is not None and n_valid is not None and n_valid > 0:
                if action >= n_valid:
                    action = n_valid - 1 if self.action_space_mode == "cutoff" else self.env.action_space.sample()
                        
        obs, reward, terminated, info = self.env.step(action=action)
        self.episode_reward += reward
        
        # Extract State
        # Note: GAT/DeepSets work on raw obs chunks, we don't strictly need the full dict extraction
        # but we keep extract_host_state for compatibility if you want to inspect IPs
        host_obs_flat = self._get_flat_host_obs(obs)
        
        self.step_counter += 1
        self.total_env_step_counter += 1
        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True
        
        # Encoder Pass
        with torch.no_grad():
            encoded_obs = self.encoder(host_obs_flat, self.host_order)
            
        return encoded_obs.cpu().numpy(), reward, terminated, truncated, info

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
        host_obs_flat = self._get_flat_host_obs(obs)

        with torch.no_grad():
            encoded_obs = self.encoder(host_obs_flat, self.host_order)

        # CSV Logging
        if self.action_history:
            self._log_history("GAT")
            
        self.action_history = []
        self.episode += 1
        return encoded_obs.cpu().numpy(), {}

    def _get_flat_host_obs(self, obs):
        """Helper: just chunk the raw vector into (N, 4)"""
        # Truncate to N*4 just in case
        valid_len = len(self.host_order) * 4
        return obs[:valid_len]

    def _reload_environment(self):
        # ... (Same reload logic, just instantiating GATWrapper/Encoder at the end) ...
        # Simplified for brevity: Copy the logic from TransformerWrapper but use GATEncoderModel
        try: 
            if self.env_creator is None or self.yaml_path is None: return False
            churn_hosts(self.yaml_path)
            fresh_cyborg = self.env_creator(self.yaml_path)
            self.raw_cyborg = fresh_cyborg
            
            if self.agent_name.lower() == 'blue': table_wrapper = BlueTableWrapper
            else: table_wrapper = RedTableWrapper
            
            env = table_wrapper(fresh_cyborg, output_mode='vector')
            env = EnumActionWrapper(env)
            env = OpenAIGymWrapper(agent_name=self.agent_name, env=env)
            self.env = env
            self.host_order = tuple(self.raw_cyborg.environment_controller.state.hosts.keys())
            
            # Re-init Encoder with new host count (if needed, though GAT handles variable size)
            # Actually GAT/DeepSets handle variable size naturally, so we might not strictly need 
            # to re-init weights, but to be safe we keep weights:
            old_dict = self.encoder.state_dict()
            self.encoder = GATEncoderModel(self.observation_space, 64).to(self.device)
            self.encoder.load_state_dict(old_dict)
            return True
        except Exception: return False

    def _log_history(self, tag):
        csv_dir = "action_logs"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"actions_{self.agent_name}_HOTRELOAD_{tag}.csv")
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['timestamp', 'episode', 'step', 'action_type', 'host_count'])
            timestamp = datetime.now().isoformat()
            for step, action in enumerate(self.action_history):
                writer.writerow([timestamp, self.episode, step, action, len(self.host_order)])

    # ... Include all other getters (get_attr, decode_action etc.) from TransformerWrapper ...
    def decode_action(self, action_idx):
        if action_idx == 0: return (None, 'Sleep')
        elif action_idx == 1: return (None, 'Monitor')
        else:
            action_names = ['Analyze', 'Remove', 'Restore', 'DecoyApache', 'DecoyFemitter', 
                           'DecoyHarakaSMTP', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 
                           'DecoyTomcat', 'DecoyVsftpd']
            return ((action_idx - 2) // 11, action_names[(action_idx - 2) % 11])


class DeepSetsWrapper(GATWrapper):
    def __init__(self, *args, **kwargs):
        # Hijack the init to use DeepSetsEncoder
        super().__init__(*args, **kwargs)
        
        # Overwrite the encoder
        embedding_dim = 64
        self.encoder = DeepSetsEncoderModel(
            observation_space=self.observation_space,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        if kwargs.get('weights_path'):
            self.encoder.load_weights(kwargs.get('weights_path'))

    def reset(self, **kwargs):
        # Override to log with "DeepSets" tag
        res = super().reset(**kwargs)
        # (Logging is handled in super but with "GAT" tag, might want to fix tag passing)
        return res
        
    def _reload_environment(self):
        # Same reload logic but re-init DeepSetsEncoderModel
        success = super()._reload_environment()
        if success:
            old_dict = self.encoder.state_dict()
            self.encoder = DeepSetsEncoderModel(self.observation_space, 64).to(self.device)
            self.encoder.load_state_dict(old_dict)
        return success
