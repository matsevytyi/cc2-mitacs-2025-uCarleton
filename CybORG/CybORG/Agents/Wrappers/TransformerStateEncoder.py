from math import isnan
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Dict, Discrete, MultiBinary, Box
import gym
import ipaddress

import numpy as np

from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class TransformerStateEncoder(BaseFeaturesExtractor):
    # def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=128, n_heads=8, n_layers=3, cyborg_env=None, agent_name=None):
    #     super().__init__(observation_space, features_dim=embedding_dim)

    #     self.embedding_dim = embedding_dim
    #     self._cyborg_env = cyborg_env
    #     self._has_reset = False
    #     self._agent_name = agent_name

    #     if self._cyborg_env is not None:
    #         self.table_env = BlueTableWrapper(self._cyborg_env, output_mode='table')
    #     else:
    #         self.table_env = None

            
    #     #print(self.table_env.observation_space)

    #     # Embeddings TODO - modified
    #     #self.ip_embed = nn.Linear(8, embedding_dim)  # IP as 8-dim (4 bytes subnet, 4 bytes host)
    #     self.ip_byte_embed = nn.Embedding(256, embedding_dim // 4)

    #     self.token_embed = nn.Embedding(1000, embedding_dim)  # for services, OS, vulns, etc.

    #     # Positional + CLS token
    #     self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
    #     self.positional_embed = nn.Parameter(torch.zeros(1, 2048, embedding_dim))  # max 20 tokens

    #     # Transformer Encoder
    #     encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
    #     self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def __init__(self, observation_space: gym.spaces.Box, embedding_dim=64, n_heads=4, n_layers=2):
        super().__init__(observation_space, features_dim=embedding_dim)

        self.embedding_dim = embedding_dim*2 # 1, 2, 3, 4 depending on amount of features

        # Each host = 4 bits, embed them into embedding_dim
        self.obs_embed = nn.Linear(4, embedding_dim)
        
        self.ip_byte_embed = nn.Embedding(256, embedding_dim // 4)

        # CLS token + positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.positional_embed = nn.Parameter(torch.zeros(1, 20, self.embedding_dim))  # supports up to 20 hosts

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def embed_ip(self, ip_str: str) -> torch.Tensor:
        ip_bytes = [int(x) for x in ip_str.split('.')]  # 4 octets
        embeds = [self.ip_byte_embed(torch.tensor(b)) for b in ip_bytes]  # 4 x (D_per_byte)
        
        # weights: first octet highest, last lowest
        weights = torch.tensor([8.0, 4.0, 2.0, 1.0]).unsqueeze(-1)  # shape [4, 1]
        
        # apply weights to each embedding
        weighted_embeds = [emb * w for emb, w in zip(embeds, weights)]
        
        # concatenate to single vector
        ip_embed = torch.cat(weighted_embeds, dim=0)  # shape [4 * D_per_byte]
        
        return ip_embed

        
    def forward(self, obs: dict, host_order, version="ip_local"):
        """
        obs: dict = flattened bit vector
        """
        
        batch_size = 1 # CC2 step default
        host_tokens_list = []

        # Reshape into hosts (batch, num_hosts, 4)
        
        for name in host_order:
            if name not in obs:
                continue
            
            # obs
            obs_chunks = obs.get(name).get('obs').reshape(batch_size, -1, 4)
            # print("obs shape before ", obs_chunks.shape)
            # print("obs before ", obs_chunks)
            obs_chunks = torch.tensor(obs_chunks, dtype=torch.float32)
            obs_chunks = self.obs_embed(obs_chunks) # [1, 1, D_obs]
            # print("obs shape after ", obs_chunks.shape)
            # print("obs after ", obs_chunks)
            
            if "ip_local" in version:
            # ips
                ip_chunks = obs.get(name).get('ips')[0]
            else:
                ip_chunks = obs.get(name).get('subnets')[0]
            # print("ip shape before ", np.shape(ip_chunks))
            # print("ip before ", ip_chunks)
            ip_chunks = self.embed_ip(ip_chunks).unsqueeze(0).unsqueeze(0) # [1, 1, D_ip]
                
            # print("ip shape after ", ip_chunks.shape)
            # print("ip after ", ip_chunks)
            
            # ports
            
            # processes
            
            
            # combine together (stack or concat)
            host_token = torch.cat([obs_chunks, ip_chunks], dim=-1) # [1, 1, D_ip+obs] or [1, 1, D_total]
            host_tokens_list.append(host_token)
            
        host_tokens = torch.cat(host_tokens_list, dim=1) # [1, num_hosts, D_total]

        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [1, 1, D_total]
        tokens = torch.cat([cls_token, host_tokens], dim=1)  # [1, 1 + num_hosts, D_total]

        # Add positional encodings
        tokens = tokens + self.positional_embed[:, :tokens.size(1), :]

        # Pass through transformer
        encoded = self.transformer(tokens)  # (batch, seq_len, D)

        # Take CLS output
        cls_output = encoded[:, 0, :]  # (batch, D)
        return cls_output


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
    