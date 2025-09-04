from math import isnan
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Dict, Discrete, MultiBinary, Box
import gym
import ipaddress

from CybORG.Agents.Wrappers import BlueTableWrapper

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class TransformerStateEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=64, n_heads=4, n_layers=2, cyborg_env=None, agent_name=None):
        super().__init__(observation_space, features_dim=embedding_dim)

        self.embedding_dim = embedding_dim
        self._cyborg_env = cyborg_env
        self._has_reset = False
        self._agent_name = agent_name

        # Embeddings
        self.ip_embed = nn.Linear(8, embedding_dim)  # IP as 8-dim (4 bytes subnet, 4 bytes host)
        self.token_embed = nn.Embedding(1000, embedding_dim)  # for services, OS, vulns, etc.

        # Positional + CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.positional_embed = nn.Parameter(torch.zeros(1, 2048, embedding_dim))  # max 20 tokens

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def encode_ip(self, ip_str):
        """Convert IP to tensor of bytes"""
        ip_bytes = torch.tensor([int(x) for x in ip_str.split('.')], dtype=torch.float32)
        return ip_bytes / 255.0  # normalize

    def forward(self, _):
        assert hasattr(self, "_cyborg_env"), "CybORG env not set!"
        assert self._cyborg_env is not torch.NoneType, "CybORG env not init!"

        table_env = BlueTableWrapper(self._cyborg_env, output_mode='table')
        true_table = table_env.get_agent_state(self._agent_name)

        observations_list = self.preprocess_table(true_table)
        observations = self.lod_2_dol(observations_list)

        all_tokens = []  # Will store tokens for all hosts (flattened)
        for i in range(len(observations['device_ip'])):
            host_dict = {k: v[i] for k, v in observations.items()}

            # Encode IP (subnet + device)
            subnet_ip = self.encode_ip(host_dict['subnet'].split('/')[0])
            device_ip = self.encode_ip(host_dict['device_ip'])
            ip_tensor = torch.cat([subnet_ip, device_ip], dim=0)
            ip_embed = self.ip_embed(ip_tensor)
            host_tokens = [ip_embed]

            # Encode other tokens (categorical/textual)
            for key, value in host_dict.items():
                if key in ['subnet', 'device_ip']:
                    continue
                items = [value] if isinstance(value, str) else value
                for item in items:
                    token_id = hash(item) % 1000
                    embed = self.token_embed(torch.tensor(token_id))
                    host_tokens.append(embed)

            # Stack per-host tokens (e.g., ~20x64)
            host_tokens = torch.stack(host_tokens, dim=0)
            all_tokens.append(host_tokens)

        # Flatten all host tokens into one sequence
        all_tokens = torch.cat(all_tokens, dim=0)  # (total_tokens_across_hosts, D)

        # [CLS] token
        cls_token = self.cls_token.clone().squeeze(0)  # shape: (1, D)
        tokens_with_cls = torch.cat([cls_token, all_tokens], dim=0)  # (1 + N, D)

        # positional encoding
        seq_len = tokens_with_cls.shape[0]
        if seq_len > self.positional_embed.shape[1]:
            raise ValueError(f"Input too long: {seq_len} > max {self.positional_embed.shape[1]}")

        tokens_with_pos = tokens_with_cls + self.positional_embed[:, :seq_len, :].squeeze(0)  # (seq_len, D)
        tokens_with_pos = tokens_with_pos.unsqueeze(0)  # Add batch dim: (1, seq_len, D)

        # Pass through transformer
        encoded = self.transformer(tokens_with_pos)  # (1, seq_len, D)
        cls_output = encoded[:, 0, :]  # (1, D)

        #print("Transformer full output:", encoded.shape)
        #print("[CLS] output:", cls_output.shape)

        return cls_output.squeeze(0)  # shape: (D,)
    

    # HELPER FUNCTIONS:
    # may be optimized
    def preprocess_table(self, raw_table):
        """Convert the full deep Blue/Red table into a list of flat token dicts per host."""
        obs_list = []

        for host_name, host_data in raw_table.items():
            if not isinstance(host_data, dict):  # skip metadata like 'success'
                continue

            flat = {}

            # IP and Subnet
            interface = host_data.get('Interface', [])
            eth_iface = next((iface for iface in interface if iface.get('Interface Name') == 'eth0'), None)
            if eth_iface:
                flat['subnet'] = str(eth_iface['Subnet'].network_address)
                flat['device_ip'] = str(eth_iface['IP Address'])

            # Hostname, OS info
            sysinfo = host_data.get('System info', {})
            flat['Hostname'] = sysinfo.get('Hostname', '')
            flat['OS'] = sysinfo.get('OSType', '').name if 'OSType' in sysinfo else ''
            flat['OSVersion'] = sysinfo.get('OSVersion', '').name if 'OSVersion' in sysinfo else ''

            # Sessions
            sessions = host_data.get('Sessions', [])
            flat['Sessions'] = [
                f"{s.get('Username', '')}|{s.get('ID', '')}|{s.get('Type', '').name if 'Type' in s else ''}"
                for s in sessions
            ]

            # Processes
            processes = host_data.get('Processes', [])
            flat['Processes'] = [f"{p.get('PID', '')}|{p.get('Username', '')}" for p in processes]

            # Users
            users = host_data.get('User Info', [])
            flat['Users'] = [u.get('Username', '') for u in users if 'Username' in u]

            # Groups
            all_groups = []
            for u in users:
                groups = u.get('Groups', [])
                for g in groups:
                    gid = g.get('GID', '')
                    all_groups.append(f"GID_{gid}")
            flat['Groups'] = all_groups

            # Add per-host processed flat observation
            obs_list.append(flat)

        return obs_list

    def lod_2_dol(self, list_of_dicts):
        collated = {}
        for key in list_of_dicts[0].keys():
            collated[key] = [d[key] for d in list_of_dicts]
        return collated
    