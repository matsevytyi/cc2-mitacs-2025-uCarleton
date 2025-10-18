import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import numpy as np

from CybORG.Shared.Enums import ProcessName

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


class TransformerStateEncoder(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, embedding_dim=64, n_heads=4, n_layers=2, initial_host_count=0):
        super().__init__(observation_space, features_dim=embedding_dim)

        self.embedding_dim = embedding_dim*2 # 1, 2, 3, 4 depending on amount of features

        self.obs_embed = nn.Linear(4, embedding_dim)
        
        self.ip_byte_embed = nn.Embedding(256, embedding_dim // 4)

        self.port_hash_size = 4096  # tune: 4096, 8192, etc.
        self.port_embed = nn.Embedding(self.port_hash_size, embedding_dim)
        
        proc_list = [p.name.lower() for p in ProcessName]   # 'unknown','svchost', ...
        self.proc_to_idx = {name: i for i, name in enumerate(proc_list)}
        self.proc_vocab_size = len(self.proc_to_idx)
        self.proc_embed = nn.Embedding(self.proc_vocab_size, embedding_dim, padding_idx=0)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim) * 0.02)

        self.error_threshold = 0.8

        self.H = initial_host_count

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            batch_first=True
        )
        
        # deployment
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # training
        self.token_head_from_cls = nn.Linear(self.embedding_dim, self.H * self.embedding_dim)

        self.recon_criterion = nn.MSELoss()

        # Separate optimizers for weighted backward propagation
        self.transformer_optimizer = optim.AdamW(
            self.transformer.parameters(), 
            lr=1e-4, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.recon_optimizer = optim.AdamW(
            self.token_head_from_cls.parameters(),
            lr=5e-5,  # 0.5x learning rate for reconstruction head
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.transformer_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.transformer_optimizer, 
            T_0=1000, 
            eta_min=1e-6
        )
        
        self.recon_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.recon_optimizer,
            T_0=1000,
            eta_min=5e-7
        )

    def forward(self, obs: dict, host_order, version="ip_local", mode="train"):
        """
        obs: dict = flattened bit vector
        """

        if mode == "train": 
            self.transformer.train()
            self.transformer_optimizer.zero_grad()
            self.recon_optimizer.zero_grad()
        
        batch_size = 1 # CC2 step default
        
        host_tokens = self.encode_features_perhost(obs, host_order, batch_size, version=version)
        
        # Apply sinusoidal positional encoding after feature encoding (as per diagram)
        seq_len = host_tokens.size(1)
        pos_encoding = self.sinusoidal_positional_encoding(
            seq_len, 
            self.embedding_dim, 
            device=host_tokens.device
        )
        host_tokens = host_tokens + pos_encoding  # Add positional encoding
        
        # CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [1, 1, D_total]
        tokens = torch.cat([cls_token, host_tokens], dim=1)  # [1, 1 + num_hosts, D_total]

        self.H = tokens.size(1) - 1
        self.B = tokens.size(2)

        # Pass through transformer
        encoded = self.transformer(tokens)  # (batch, seq_len, D)

        # Take CLS output
        cls_output = encoded[:, 0, :]  # (batch, D)
        
        if mode == "train": 
            reconstruction = self.token_head_from_cls(cls_output) #(batch, seq_length)
            reconstruction = reconstruction.view(batch_size, self.H, self.embedding_dim)

            recon_loss = self.recon_criterion(reconstruction, host_tokens)

            # Weighted backward propagation
            # Prioritize transformer encoder updates (1.0x) over reconstruction head (0.5x)
            recon_loss.backward()
            
            # Apply gradient clipping and optimizer steps with weighted updates
            torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.token_head_from_cls.parameters(), 0.5)
            
            self.transformer_optimizer.step() 
            self.recon_optimizer.step()
            
            self.transformer_scheduler.step()
            self.recon_scheduler.step()
            
            self.error_threshold = 0.9 * self.error_threshold + 0.1 * recon_loss.item()

        return cls_output
    
    def encode_features_perhost(self, obs: dict, host_order, batch_size=1, version="ip_local"):
        host_tokens_list = []
        
        for name in host_order:
            if name not in obs:
                continue
            
            # obs
            obs_chunks = obs.get(name).get('obs').reshape(batch_size, -1, 4)

            obs_chunks = torch.tensor(obs_chunks, dtype=torch.float32)
            obs_chunks = self.obs_embed(obs_chunks) # [1, 1, D_obs]
            obs_chunks = F.layer_norm(obs_chunks, obs_chunks.shape[-1:])
            
            if "obs_only" in version:
                host_tokens_list.append(obs_chunks)
                continue
            
            # ips    
            if "ip_local" in version:
                ip_chunks = obs.get(name).get('ips')[0]
            else:
                ip_chunks = obs.get(name).get('subnets')[0]
                
            ip_chunks = self.embed_ip(ip_chunks).unsqueeze(0).unsqueeze(0) # [1, 1, D_ip]
            ip_chunks = F.layer_norm(ip_chunks, ip_chunks.shape[-1:])
            
            # ports
            # ports_list = obs.get(name).get('ports', [])
            
            # if len(ports_list) > 0:
                
            #     port_indices = torch.tensor([self.port_to_index(int(p)) for p in ports_list],
            #                                 dtype=torch.long, device=self.cls_token.device)
            #     port_vecs = self.port_embed(port_indices)              # [n_ports, d_port]
            #     port_emb = port_vecs.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, d_port]
            # else:
            #     port_emb = torch.zeros(1, 1, self.embedding_dim // 4, device=self.cls_token.device)
                
            # port_emb = F.layer_norm(port_emb, port_emb.shape[-1:])
            
            # # processes
            # proc_list = obs.get(name).get('processes', [])
            
            # if proc_list is None:
            #     proc_list = []

            # if len(proc_list) > 0:
            #     proc_indices = torch.tensor([self.process_to_index(str(p)) for p in proc_list],
            #                                 dtype=torch.long, device=self.cls_token.device)
            #     proc_vecs = self.proc_embed(proc_indices)             # [n_proc, d_proc]
            #     proc_emb = proc_vecs.mean(dim=0, keepdim=True).unsqueeze(0) # [1,1,d_proc]
            # else:
            #     proc_emb = torch.zeros(1, 1, self.embedding_dim//4, device=self.cls_token.device)
                
            # proc_emb = F.layer_norm(proc_emb, proc_emb.shape[-1:])

            # combine together (stack or concat)
            #host_token = torch.cat([obs_chunks, ip_chunks, port_emb, proc_emb], dim=-1)
      
            
            # combine together (stack or concat)
            host_token = torch.cat([obs_chunks, ip_chunks], dim=-1) # [1, 1, D_ip+obs+...] or [1, 1, D_total]
            host_tokens_list.append(host_token)
            
        return torch.cat(host_tokens_list, dim=1) # [1, num_hosts, D_total]

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
    
    @staticmethod
    def sinusoidal_positional_encoding(seq_len, dim, device="cpu"):
        """Generate sinusoidal positional encoding"""
        pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float, device=device).unsqueeze(0)
        angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
        angle_rads = pos * angle_rates
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return pe.unsqueeze(0)  # [1, seq_len, dim]

    def port_to_index(self, port:int) -> int:
        """Hash a port into embedding table index."""
        return port % self.port_hash_size

    def process_to_index(self, proc_name: str) -> int:
        """Map process string -> index using ProcessName enum. Unknown -> index of 'unknown'"""
        try:
            # use the enum parser if available
            enum_val = ProcessName.parse_string(proc_name)
            name = enum_val.name.lower()
        except Exception:
            name = proc_name.lower()

        return self.proc_to_idx.get(name, self.proc_to_idx.get('unknown', 0))
