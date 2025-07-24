
from stable_baselines3 import DQN
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper, BlueTableWrapper, EnumActionWrapper, OpenAIGymWrapper, TrueTableWrapper, RedTableWrapper
from CybORG.Agents import B_lineAgent

import os, sys, inspect

from torch._C import LiteScriptModule

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modify_ss_transformers import TransformerStateEncoder

def preprocess_table(raw_table):
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

def lod_2_dol(list_of_dicts):
    collated = {}
    for key in list_of_dicts[0].keys():
        collated[key] = [d[key] for d in list_of_dicts]
    return collated


# time till conversions
# straightforward metrics compariosn

# dreamer v3 ppo


# Load cfg
extended = True

# Locate Scenario2.yaml path
if extended:

    path = "Scenario2.yaml"
    # path = "Scenario2_Linear.yaml"
    # path = "Scenario2_Extended.yaml"

    path = os.path.dirname(__file__) + "/scenarios/" + path
else:

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'


# Initialize and wrap CybORG environment
cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
#env = ChallengeWrapper(env=cyborg, agent_name='Blue')

# Actual table
#env = TrueTableWrapper(cyborg)

# Blue Agent Table
env = BlueTableWrapper(cyborg, output_mode='table')

env.reset()
print("Blue table")
true_table = env.get_agent_state('Blue')

trans_converted_table = preprocess_table(true_table)

for i in trans_converted_table:
    print(i) 
    print("\n")

obs_dict = lod_2_dol(trans_converted_table)

# policy_kwargs = dict(
#     features_extractor_class=TransformerStateEncoder,
#     features_extractor_kwargs=dict(embedding_dim=64),
# )


# model = DQN("MultiInputPolicy", env, policy_kwargs=policy_kwargs, ...)

from gym.spaces import Dict as SpaceDict, Text, Sequence
import torch

# Dummy observation space to satisfy constructor
dummy_obs_space = SpaceDict({
    "subnet": Text(15),
    "device_ip": Text(15),
    "Hostname": Text(30),
    "OS": Text(15),
    "OSVersion": Text(15),
    "Sessions": Sequence(Text(50)),
    "Processes": Sequence(Text(30)),
    "Users": Sequence(Text(20)),
    "Groups": Sequence(Text(20)),
})

encoder = TransformerStateEncoder(dummy_obs_space)

with torch.no_grad():
    embedding = encoder(obs_dict)  # shape: (B, D) — one vector per host

print(embedding.shape)


# Red Agent Table
# env1 = RedTableWrapper(cyborg, output_mode='table')

# env1.reset()

# true_table = env1.get_agent_state('Red')
# print("Red table")
# print('-'*70)
# print(preprocess_table(true_table))