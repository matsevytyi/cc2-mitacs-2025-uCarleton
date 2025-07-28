from math import isnan
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Dict, Discrete, MultiBinary, Box
import gym
import ipaddress

from CybORG.Agents.Wrappers import BlueTableWrapper  # safe to import here

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class TransformerStateEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=64, n_heads=4, n_layers=2, cyborg_env=None):
        super().__init__(observation_space, features_dim=embedding_dim)

        self.embedding_dim = embedding_dim
        self._cyborg_env = cyborg_env
        self._has_reset = False

        # Embeddings
        self.ip_embed = nn.Linear(8, embedding_dim)  # IP as 8-dim (4 bytes subnet, 4 bytes host)
        self.token_embed = nn.Embedding(1000, embedding_dim)  # for services, OS, vulns, etc.

        # Positional + CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.positional_embed = nn.Parameter(torch.zeros(1, 20, embedding_dim))  # max 20 tokens

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

        # Obtauin agent state from full env
        blue_table_env = BlueTableWrapper(self._cyborg_env, output_mode='table')

        true_table = blue_table_env.get_agent_state('Blue')

        # Process to dict format with keys like: device_ip, services, etc.
        observations_list = self.preprocess_table(true_table)
        observations = self.lod_2_dol(observations_list)

        batch_embeddings = []
        print("OBS")
        print(observations)
        batch_size = len(observations['device_ip'])

        for i in range(batch_size):
            tokens = []

            # Step 1: Collect per-host dict from dict of lists (to make it usable with Gym)
            host_dict = {k: v[i] for k, v in observations.items()}

            # Step 2: Encode IP
            subnet_ip = self.encode_ip(host_dict['subnet'].split('/')[0])
            device_ip = self.encode_ip(host_dict['device_ip'])
            ip_tensor = torch.cat([subnet_ip, device_ip], dim=0)
            ip_embed = self.ip_embed(ip_tensor)
            tokens.append(ip_embed)

            # Step 3: Encode categorical strings
            for key, value in host_dict.items():
                if key in ['subnet', 'device_ip']:
                    continue  # already handled
                items = [value] if isinstance(value, str) else value
                for item in items:
                    token_id = hash(item) % 1000
                    embed = self.token_embed(torch.tensor(token_id))
                    tokens.append(embed)

            # Step 4: Padding
            tokens = torch.stack(tokens, dim=0)
            cls = self.cls_token.clone().squeeze(0) # shape: (1, 1, D) ->  (1, D)
            tokens = torch.cat([cls, tokens], dim=0)

            pad_len = 20 - tokens.shape[0]
            if pad_len > 0:
                tokens = torch.cat([tokens, torch.zeros(pad_len, self.embedding_dim)], dim=0)
            else:
                tokens = tokens[:20]

            batch_embeddings.append(tokens)


        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (B, 20, D)
        tokens_with_pos = batch_embeddings + self.positional_embed[:, :20]

        # forward through transformer
        encoded = self.transformer(tokens_with_pos)

        # collect [CLS] token in the end
        cls_output = encoded[:, 0]


        print("Transformer full output:", encoded)
        print("[CLS] output:", cls_output)

        return cls_output
    
    # HELPER FUNCTIONS:
    # TODO: optimize
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