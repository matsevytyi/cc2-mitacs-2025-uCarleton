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

# Just for debug purposes, to see how it works
class TransformerStateEncoderDebug(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=64, n_heads=4, n_layers=2):
        super().__init__(observation_space, features_dim=embedding_dim)

        self.embedding_dim = embedding_dim

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

    def forward(self, observations):
        batch_embeddings = []
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


        #print("Transformer full output:", encoded)
        #print("[CLS] output:", cls_output)

        return cls_output