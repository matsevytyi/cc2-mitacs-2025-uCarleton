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

class DeepSetsPermInvEncoder(nn.Module):
    def __init__(self, observation_space, embedding_dim=64, device='cpu'):
        super().__init__()
        self.embedding_dim = embedding_dim * 2 # Assuming D_total from tokenizer is dim*2

        self.obs_embed = nn.Linear(4, embedding_dim)
        
        self.ip_byte_embed = nn.Embedding(256, embedding_dim // 4)

        self.port_hash_size = 4096  # tune: 4096, 8192, etc.
        self.port_embed = nn.Embedding(self.port_hash_size, embedding_dim)
        
        proc_list = [p.name.lower() for p in ProcessName]   # 'unknown','svchost', ...
        self.proc_to_idx = {name: i for i, name in enumerate(proc_list)}
        self.proc_vocab_size = len(self.proc_to_idx)
        self.proc_embed = nn.Embedding(self.proc_vocab_size, embedding_dim, padding_idx=0)
        
        # Phi: Maps tokenized node features to higher dim
        self.phi = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Equivariant Node-Level Readout (Combines Local + Global)
        # Input size is embedding_dim * 2 (concatenated local + global)
        self.equivariant_layer = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Final Readout to fix the size for PPO/DQN
        self.final_readout = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.device = device

    def forward(self, obs, host_order, version=None):
        # 1. Shared Feature Tokenization (Shape: [Batch, N, D_total])
        tokenized_obs = self.encode_features_perhost(obs, host_order)
        N = tokenized_obs.size(1)
        
        # 2. Local Processing (Phi)
        h_local = self.phi(tokenized_obs)  # Shape: [Batch, N, D_total]
        
        # 3. Permutation-Invariant Global Aggregation (Sum)
        h_global = torch.sum(h_local, dim=1, keepdim=True)  # Shape: [Batch, 1, D_total]
        
        # 4. Equivariant Concatenation (Broadcast global to all nodes)
        h_global_expanded = h_global.expand(-1, N, -1)
        h_combined = torch.cat([h_local, h_global_expanded], dim=-1) # Shape: [Batch, N, D_total * 2]
        
        # 5. Process combined features
        h_nodes_final = self.equivariant_layer(h_combined) # Shape: [Batch, N, D_total]
        
        # 6. Global Pooling for RL Agent (PPO/DQN require a fixed 1D vector)
        # We max-pool the node features into a single vector representing the state
        rl_state = torch.max(h_nodes_final, dim=1)[0] # Shape: [Batch, D_total]
        
        return self.final_readout(rl_state)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    # =================== HELPER METHODS ===================
    
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
            if "full_design" in version:
                ports_list = obs.get(name).get('ports', [])
                
                if len(ports_list) > 0:
                    
                    port_indices = torch.tensor([self.port_to_index(int(p)) for p in ports_list],
                                                dtype=torch.long, device=self.device)
                    port_vecs = self.port_embed(port_indices)              # [n_ports, d_port]
                    port_emb = port_vecs.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, d_port]
                else:
                    port_emb = torch.zeros(1, 1, self.embedding_dim // 4, device=self.device)
                    
                port_emb = F.layer_norm(port_emb, port_emb.shape[-1:])
                
                # processes
                proc_list = obs.get(name).get('processes', [])
                
                if proc_list is None:
                    proc_list = []

                if len(proc_list) > 0:
                    proc_indices = torch.tensor([self.process_to_index(str(p)) for p in proc_list],
                                                dtype=torch.long, device=self.device)
                    proc_vecs = self.proc_embed(proc_indices)             # [n_proc, d_proc]
                    proc_emb = proc_vecs.mean(dim=0, keepdim=True).unsqueeze(0) # [1,1,d_proc]
                else:
                    proc_emb = torch.zeros(1, 1, self.embedding_dim//4, device=self.device)
                    
                proc_emb = F.layer_norm(proc_emb, proc_emb.shape[-1:])
      
            # combine together
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
