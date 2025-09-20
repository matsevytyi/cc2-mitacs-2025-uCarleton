# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import inspect
# from CybORG import CybORG

# from CybORG.Agents import B_lineAgent

# extended = False

# # Locate Scenario2.yaml path
# if extended:

#     #scenario_name = "Scenario2.yaml"
#     #scenario_name = "Scenario2_Linear.yaml"
#     scenario_name = "Scenario2_Extended.yaml"

#     path = os.path.dirname(__file__) + "/scenarios/" + scenario_name

# else:

#     path = str(inspect.getfile(CybORG))
#     path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

# # Create the environment
# cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})

# # Reset the environment and take a step
# cyborg.reset()

# for _ in range(1):
#     cyborg.step(agent='Blue', action='Sleep')

# results = cyborg.step(action='Sleep', agent='Red')

# # Inspect the result

# for i in range(20):
#     results = cyborg.step(agent='Red')
#     print(f"\nStep {i+1} Observation:")
#     print(results.observation)
#     print("Action taken:", results.action)
#     print("Reward:", results.reward)
#     print("Done:", results.done)


import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Setup ---
embedding_dim = 64
ip_byte_embed = nn.Embedding(256, embedding_dim // 4)

def encode_ip(ip_str):
    ip_bytes = [int(x) for x in ip_str.split('.')]  # split into 4 octets
    embeds = [ip_byte_embed(torch.tensor(b)) for b in ip_bytes]
    return torch.cat(embeds, dim=-1)  # shape: (embedding_dim,)

def embed_ip(ip_str: str) -> torch.Tensor:
        ip_bytes = [int(x) for x in ip_str.split('.')]  # 4 octets
        embeds = [ip_byte_embed(torch.tensor(b)) for b in ip_bytes]  # 4 x (D_per_byte)
        
        # weights: first octet highest, last lowest
        weights = torch.tensor([8.0, 4.0, 2.0, 1.0]).unsqueeze(-1)  # shape [4, 1]
        
        # apply weights to each embedding
        weighted_embeds = [emb * w for emb, w in zip(embeds, weights)]
        
        # concatenate to single vector
        ip_embed = torch.cat(weighted_embeds, dim=0)  # shape [4 * D_per_byte]
        
        return ip_embed


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# --- Example IPs ---
ip1 = "11.10.10.1"
ip2 = "12.10.10.1"   # only first octet differs
ip3 = "11.10.5.5"    # same first two octets, last two differ

# --- Encode ---
e1, e2, e3 = embed_ip(ip1), embed_ip(ip2), embed_ip(ip3)

# --- Compare similarities ---
print()
print()
print("11.10.10.1 <-> 12.10.10.1:\n", cosine_sim(e1, e2))
print("\n11.10.10.1 <-> 11.10.5.5:\n", cosine_sim(e1, e3))
print();print()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class UserProcessEncoder(nn.Module):
#     def __init__(self, num_users, embedding_dim=64):
#         super().__init__()
#         self.user_embed = nn.Embedding(num_users, embedding_dim)
#         self.proc_proj = nn.Linear(1, embedding_dim)
#         self.dropout = nn.Dropout(p=0.3)


#     def forward(self, host_users, user2id):
#         """
#         host_users: list of dicts per user, e.g.:
#             [
#               {"username": "root", "proc_count": 3},
#               {"username": "vagrant", "proc_count": 1}
#             ]
#         user2id: dictionary mapping usernames -> ids
#         """
#         user_reprs = []
#         for u in host_users:
#             uid = user2id.get(u["username"], user2id["<UNK>"])
#             uid = torch.tensor(uid, dtype=torch.long)
            
#             proc_count = torch.tensor([u["proc_count"]], dtype=torch.float32)

#             u_emb = self.user_embed(uid)
#             p_emb = self.proc_proj(proc_count)

#             user_repr = self.dropout((u_emb * (1 + proc_count)) + p_emb)
            
#             user_reprs.append(user_repr)

#         if len(user_reprs) > 0:
#             host_repr = torch.stack(user_reprs).mean(dim=0)
#         else:
#             host_repr = torch.zeros(self.user_embed.embedding_dim)

#         return host_repr

# host has 2 users, one with 3 processes, another with 1
# known_users = ["root", "ubuntu", "www-data", "pi", "GreenAgent", 
#                "Administrator", "vagrant", "SYSTEM"]

# user2id = {u: i for i, u in enumerate(known_users)}
# user2id["<UNK>"] = len(user2id)  # unknown usernames

# print(user2id)
# print(type(user2id))

# encoder = UserProcessEncoder(num_users=len(user2id), embedding_dim=64)

# with torch.no_grad():
#     encoder.user_embed.weight[-1].uniform_(-10, 10)

# host_1_users = [
#     {"username": "root", "proc_count": 3},
#     {"username": "SYSTEM", "proc_count": 7},
# ]

# host_2_users = [
#     {"username": "root", "proc_count": 3},
#     {"username": "SYSTEM", "proc_count": 6},
#     {"username": "new_hacker", "proc_count": 2}  # new user
# ]

# host_3_users = [
#     {"username": "root", "proc_count": 5},
#     {"username": "SYSTEM", "proc_count": 5}, # state modified
# ]

# host_1_embedding = encoder(host_1_users, user2id)
# host_2_embedding = encoder(host_2_users, user2id)
# host_3_embedding = encoder(host_3_users, user2id)

# host_1_embedding = F.normalize(host_1_embedding, dim=0)
# host_2_embedding = F.normalize(host_2_embedding, dim=0)
# host_3_embedding = F.normalize(host_3_embedding, dim=0)

# print(host_1_embedding)
# print(host_2_embedding)
# print(host_3_embedding)

# print()
# print()
# print("host 1 <-> host 2:", F.cosine_similarity(host_1_embedding, host_2_embedding, dim=0).item())
# print("host 1 <-> host 3:", F.cosine_similarity(host_1_embedding, host_3_embedding, dim=0).item())
# print("host 2 <-> host 3:", F.cosine_similarity(host_2_embedding, host_3_embedding, dim=0).item())
# print();print()

# difference between embeddings of different users
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# pairs = [(u, p) for u in known_users for p in [1, 3, 5, 10]]
# vecs = []
# labels = []

# for u, p in pairs:

#     emb = encoder(user2id[u], p)
#     vecs.append(emb.detach().numpy()[0])
#     labels.append(f"{u}-{p}")

# vecs = np.array(vecs)
# # cosine similarity
# sim_matrix = np.dot(vecs, vecs.T)

# plt.figure(figsize=(8,6))
# sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels,
#             cmap="viridis", annot=False, square=True)
# plt.title("User+Process Embedding Similarities")
# plt.tight_layout()
# plt.show()

# # same user over time
# # timesteps = [1, 2, 3, 5, 8, 12]
# # traj_vecs = []

# # for p in timesteps:
# #     uid = torch.tensor([user2id["SYSTEM"]])
# #     proc = torch.tensor([p])
# #     emb = encoder(uid, proc)
# #     traj_vecs.append(emb.detach().numpy()[0])

# # traj_vecs = np.array(traj_vecs)

# # # reduce to 2D with PCA for visualization
# # from sklearn.decomposition import PCA
# # pca = PCA(n_components=2)
# # traj_2d = pca.fit_transform(traj_vecs)

# # plt.figure(figsize=(6,6))
# # plt.plot(traj_2d[:,0], traj_2d[:,1], marker="o", linestyle="-")
# # for i, p in enumerate(timesteps):
# #     plt.text(traj_2d[i,0]+0.02, traj_2d[i,1], f"proc={p}", fontsize=9)

# # plt.title("Host Embedding Trajectory Over Time (SYSTEM user)")
# # plt.xlabel("PCA-1")
# # plt.ylabel("PCA-2")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()
