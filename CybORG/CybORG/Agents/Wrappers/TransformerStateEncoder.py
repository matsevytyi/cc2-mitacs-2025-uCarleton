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


class TransformerStateEncoderV2(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        embedding_dim=64,
        n_heads=4,
        n_layers=2,
        initial_host_count=0,
        mode='train',
        top_k=8,
        focus_mode='stateful',
        use_candidate_retrieval=True
    ):
        self.top_k = top_k
        self.focus_mode = focus_mode  # 'stateful' (tracks last acted host) or 'salience' (tracks #1 anomaly)
        self.use_candidate_retrieval = use_candidate_retrieval

        total_embed_dim = embedding_dim * 2
        out_features_dim = (1 + top_k) * total_embed_dim if use_candidate_retrieval else total_embed_dim

        super().__init__(observation_space, features_dim=out_features_dim)

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
        self.recon_loss = 1.00

        self.H = initial_host_count

        # Focus host and candidate slot tracking for dynamic action wrapper
        self.focus_host = None
        self.current_candidate_hosts = []
        self.last_cls_output = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            batch_first=True,
            norm_first=True,  # Crucial for stability 
            activation="gelu",
            dropout=0.1
        )
        
        # deployment
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mode = mode

        # training: Bilinear / Hadamard IP-Query Reconstruction Head (invariant to dynamic self.H)
        self.token_head_from_cls = nn.ModuleDict({
            'ip_query_proj': nn.Linear(self.embedding_dim // 2, self.embedding_dim),
            'status_recon_head': nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        })

        self.recon_criterion = nn.MSELoss()

        # separate optimizers for weighted backward propagation
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
        
        if hasattr(F, 'scaled_dot_product_attention'):
            print("Succesfully using PyTorch 2.0+ Scaled Dot Product Attention (FlashAttention compatible)!")
        
    # =================== MAIN METHODS ===================

    def forward(self, obs: dict, host_order, version="ip_local", mode="train"):
        """
        obs: dict = flattened bit vector
        """

        if mode == "train": 
            self.transformer.train()
            self.transformer_optimizer.zero_grad()
            self.recon_optimizer.zero_grad()
        
        batch_size = 1 # CC2 step default
        
        active_host_order = [name for name in host_order if name in obs]
        host_tokens = self.encode_features_perhost(obs, host_order, batch_size, version=version)
        
        # Apply sinusoidal positional encoding after feature encoding (as per diagram)
        seq_len = host_tokens.size(1)
        
        # pos_encoding = self.sinusoidal_positional_encoding(
        #     seq_len, 
        #     self.embedding_dim, 
        #     device=host_tokens.device
        # )
        
        # removed since we can use IP embedding to identify devices
        # pos_encoding = self.get_positional_encoding(
        #     seq_len, 
        #     device=host_tokens.device
        # )
        
        # host_tokens = host_tokens + pos_encoding  # Add positional encoding
        
        # CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [1, 1, D_total]
        tokens = torch.cat([cls_token, host_tokens], dim=1)  # [1, 1 + num_hosts, D_total]

        self.H = tokens.size(1) - 1
        self.B = tokens.size(2)

        # Pass through transformer
        encoded = self.transformer(tokens)  # (batch, seq_len, D)

        # Take CLS output
        cls_output = encoded[:, 0, :]  # (batch, D)
        encoded_host_tokens = encoded[:, 1:, :] # (batch, num_hosts, D)
        self.last_cls_output = cls_output
        
        if mode == "train": 
            # Split host_tokens into 64D status targets (obs_chunks) and 64D IP queries (ip_chunks)
            half_dim = self.embedding_dim // 2
            status_targets = host_tokens[:, :, :half_dim]  # [batch, H, D // 2]
            ip_queries = host_tokens[:, :, half_dim:]      # [batch, H, D // 2]

            # Query Rep_<CLS> with each host's IP embedding via Hadamard product
            ip_filter = self.token_head_from_cls['ip_query_proj'](ip_queries)           # [batch, H, D]
            queried_cls = cls_output.unsqueeze(1) * ip_filter                           # [batch, H, D]
            reconstruction = self.token_head_from_cls['status_recon_head'](queried_cls) # [batch, H, D // 2]

            recon_loss = self.recon_criterion(reconstruction, status_targets.detach())
            self.recon_loss = recon_loss

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

        if self.use_candidate_retrieval:
            rl_state, self.current_candidate_hosts = self.retrieve_candidates(
                cls_output=cls_output,
                host_tokens=host_tokens,
                encoded_host_tokens=encoded_host_tokens,
                active_host_order=active_host_order
            )
            return rl_state, encoded_host_tokens

        return cls_output, encoded_host_tokens

    # =================== CANDIDATE RETRIEVAL & ACTION DECODING ===================

    def retrieve_candidates(self, cls_output, host_tokens, encoded_host_tokens, active_host_order):
        """
        Selects K candidate hosts without learned decision weights:
          - Slots 0 .. K/2 - 1: Top Salience hosts (Status deviation + <CLS> similarity)
          - Slots K/2 .. K - 1: Strided IP Distance Quantiles from focus_host
        Returns:
          - rl_state: [batch_size, (1 + K) * embedding_dim]
          - candidate_host_names: list of K host names corresponding to Slots 0 .. K-1
        """
        batch_size = cls_output.size(0)
        num_hosts = len(active_host_order)
        half_dim = self.embedding_dim // 2

        with torch.no_grad():
            ip_chunks = host_tokens[:, :, half_dim:]  # [1, H, 64]
            enc_hosts = encoded_host_tokens[0]        # [H, 128]
            cls_vec = cls_output[0].unsqueeze(0)      # [1, 128]

            # 1. Compute Salience Score: L2 distance from mean host token + Cosine similarity to <CLS>
            mean_token = enc_hosts.mean(dim=0, keepdim=True)                    # [1, 128]
            token_dev = torch.norm(enc_hosts - mean_token, p=2, dim=-1)         # [H]
            cls_sim = F.cosine_similarity(enc_hosts, cls_vec, dim=-1)           # [H]
            salience_scores = token_dev + cls_sim                               # [H]

            k_salience = min(self.top_k // 2, num_hosts)
            salience_indices = torch.topk(salience_scores, k=k_salience, largest=True).indices.tolist()

            # 2. Resolve focus_host index
            if self.focus_mode == 'salience':
                focus_idx = salience_indices[0]
                self.focus_host = active_host_order[focus_idx]
            else:
                # Stateful mode: use current focus_host if valid, otherwise initialize to IP Medoid
                if self.focus_host is not None and self.focus_host in active_host_order:
                    focus_idx = active_host_order.index(self.focus_host)
                else:
                    ip_vecs = ip_chunks[0]                                      # [H, 64]
                    ip_mean = ip_vecs.mean(dim=0, keepdim=True)                 # [1, 64]
                    medoid_dists = torch.norm(ip_vecs - ip_mean, p=2, dim=-1)   # [H]
                    focus_idx = int(torch.argmin(medoid_dists).item())
                    self.focus_host = active_host_order[focus_idx]

            # 3. Select remaining K - k_salience slots via IP Distance Quantiles from focus_host
            k_quantile = self.top_k - len(salience_indices)
            salience_set = set(salience_indices)
            remaining_indices = [i for i in range(num_hosts) if i not in salience_set]

            quantile_indices = []
            if k_quantile > 0 and len(remaining_indices) > 0:
                rem_tensor = torch.tensor(remaining_indices, dtype=torch.long, device=ip_chunks.device)
                focus_ip = ip_chunks[0, focus_idx].unsqueeze(0)                 # [1, 64]
                rem_ips = ip_chunks[0, rem_tensor]                              # [R, 64]
                ip_dists = torch.norm(rem_ips - focus_ip, p=2, dim=-1)          # [R]

                sorted_rem_order = torch.argsort(ip_dists, descending=False)
                sorted_rem_indices = rem_tensor[sorted_rem_order]

                if len(sorted_rem_indices) >= k_quantile:
                    q_positions = torch.linspace(
                        0, len(sorted_rem_indices) - 1, steps=k_quantile, device=ip_chunks.device
                    ).round().long()
                    quantile_indices = sorted_rem_indices[q_positions].tolist()
                else:
                    quantile_indices = sorted_rem_indices.tolist()

            candidate_indices = salience_indices + quantile_indices

            # Pad with focus_idx if total active hosts H < top_k
            while len(candidate_indices) < self.top_k:
                candidate_indices.append(focus_idx)

            candidate_idx_tensor = torch.tensor(
                candidate_indices[:self.top_k], dtype=torch.long, device=encoded_host_tokens.device
            )

        # Concatenate Rep_<CLS> with the K selected encoded_host_tokens (preserves gradients if needed)
        selected_host_tokens = encoded_host_tokens[:, candidate_idx_tensor, :]  # [B, K, D]
        flattened_slots = selected_host_tokens.reshape(batch_size, self.top_k * self.embedding_dim)
        rl_state = torch.cat([cls_output, flattened_slots], dim=-1)             # [B, (1 + K) * D]

        candidate_host_names = [active_host_order[i] for i in candidate_indices[:self.top_k]]
        return rl_state, candidate_host_names

    def decode_action(self, action: int, num_host_actions: int = 12):
        """
        Maps an integer action from Discrete(1 + top_k * num_host_actions) to CybORG:
          - action == 0 -> ('Sleep', None, None)
          - action >= 1 -> ('HostAction', target_host_name, tool_idx) and updates focus_host
        """
        action = int(action)
        if action == 0 or len(self.current_candidate_hosts) == 0:
            return "Sleep", None, None

        slot_idx = (action - 1) // num_host_actions
        tool_idx = (action - 1) % num_host_actions
        slot_idx = min(max(slot_idx, 0), len(self.current_candidate_hosts) - 1)

        target_host = self.current_candidate_hosts[slot_idx]
        if self.focus_mode == 'stateful':
            self.focus_host = target_host

        return "HostAction", target_host, tool_idx

    def reset_focus(self):
        """Call inside Wrapper.reset() to re-initialize focus_host on episode reset."""
        self.focus_host = None
        self.current_candidate_hosts = []
    
    def save_weights(self, path):
        """Save only the encoder weights"""
        torch.save({
            'transformer': self.transformer.state_dict(),
            'obs_embed': self.obs_embed.state_dict(),
            'ip_byte_embed': self.ip_byte_embed.state_dict(),
            'cls_token': self.cls_token,
            'token_head_from_cls': self.token_head_from_cls.state_dict(),
            # any other components you want to save
        }, path)

    def load_weights_from_dict(self, weights_dict):
        """Load weights from dict (no I/O) - load each separately"""
        try:
            self.transformer.load_state_dict(weights_dict['transformer'])
        except Exception as e:
            print(f"Warning: transformer failed: {e}")
        
        try:
            self.obs_embed.load_state_dict(weights_dict['obs_embed'])
        except Exception as e:
            print(f"Warning: obs_embed failed: {e}")
        
        try:
            self.ip_byte_embed.load_state_dict(weights_dict['ip_byte_embed'])
        except Exception as e:
            print(f"Warning: ip_byte_embed failed: {e}")
        
        try:
            # cls_token: convert to Parameter if needed
            cls_data = weights_dict['cls_token']
            if isinstance(cls_data, torch.nn.Parameter):
                cls_data = cls_data.data
            self.cls_token.data.copy_(cls_data)
        except Exception as e:
            print(f"Warning: cls_token failed: {e}")

        if 'token_head_from_cls' in weights_dict:
            try:
                self.token_head_from_cls.load_state_dict(weights_dict['token_head_from_cls'])
            except Exception as e:
                print(f"Warning: token_head_from_cls failed: {e}")
        


    
    def load_weights(self, path):
        """Load encoder weights"""
        checkpoint = torch.load(path)
        self.transformer.load_state_dict(checkpoint['transformer'])
        self.obs_embed.load_state_dict(checkpoint['obs_embed'])
        self.ip_byte_embed.load_state_dict(checkpoint['ip_byte_embed'])
        self.cls_token = checkpoint['cls_token']
        if self.mode == 'train' or self.mode == 'tune':
            self.token_head_from_cls.load_state_dict(checkpoint['token_head_from_cls'])
    
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
                                                dtype=torch.long, device=self.cls_token.device)
                    port_vecs = self.port_embed(port_indices)              # [n_ports, d_port]
                    port_emb = port_vecs.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, d_port]
                else:
                    port_emb = torch.zeros(1, 1, self.embedding_dim // 4, device=self.cls_token.device)
                    
                port_emb = F.layer_norm(port_emb, port_emb.shape[-1:])
                
                # processes
                proc_list = obs.get(name).get('processes', [])
                
                if proc_list is None:
                    proc_list = []

                if len(proc_list) > 0:
                    proc_indices = torch.tensor([self.process_to_index(str(p)) for p in proc_list],
                                                dtype=torch.long, device=self.cls_token.device)
                    proc_vecs = self.proc_embed(proc_indices)             # [n_proc, d_proc]
                    proc_emb = proc_vecs.mean(dim=0, keepdim=True).unsqueeze(0) # [1,1,d_proc]
                else:
                    proc_emb = torch.zeros(1, 1, self.embedding_dim//4, device=self.cls_token.device)
                    
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
    
    def get_positional_encoding(self, seq_len, device):
        # If we haven't created it yet, or the new sequence is longer than our cache
        if not hasattr(self, 'pe_cache') or self.pe_cache.size(1) < seq_len:
            # Create a buffer slightly larger than needed to avoid frequent resizing
            # e.g., round up to nearest 100
            alloc_len = ((seq_len // 100) + 1) * 100
            
            # Generate your encoding (your existing logic)
            pos = torch.arange(alloc_len, dtype=torch.float, device=device).unsqueeze(1)
            i = torch.arange(self.embedding_dim, dtype=torch.float, device=device).unsqueeze(0)
            angle_rates = 1 / (10000 ** (2 * (i // 2) / self.embedding_dim))
            angle_rads = pos * angle_rates
            pe = torch.zeros(alloc_len, self.embedding_dim, device=device)
            pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
            pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
            
            # Save it as a buffer so it moves with .to(device) automatically
            self.register_buffer('pe_cache', pe.unsqueeze(0), persistent=False)
            
        # Return the slice we need
        return self.pe_cache[:, :seq_len, :]

    def _init_weights(self):
        # 1. Initialize embeddings with very small weights so unseen categorical 
        # features (like a new port) don't inject massive random noise into the state
        nn.init.normal_(self.port_embed.weight, std=0.01)
        
        # 2. Keep the CLS token small
        nn.init.normal_(self.cls_token, std=0.01)
        
        # 3. Orthogonal initialization for the observation linear layer 
        # (Standard best-practice for Stable Baselines3 RL features)
        nn.init.orthogonal_(self.obs_embed.weight, gain=np.sqrt(2))
        nn.init.constant_(self.obs_embed.bias, 0.0)


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