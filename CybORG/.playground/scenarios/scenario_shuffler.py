import yaml
import random

"""
Developed by Andrii Matsevytyi
As a Mitacs intern
Under prof. Gie Gao and prof. Thomas Kunz supervision
At the University of Carleton


Created for testing performance of Transformer Encoder to get modified observation space 
while preserving old action space compatibility
"""

def update_yaml_file(filepath, mode='shuffle', new_assignments=None, seed=None):
    """
    Reads YAML file, updates Subnets hosts, saves back to same file.
    Parameters:
        filepath: Path to your YAML file
        mode: 'shuffle' or 'assign'
        new_assignments: dict for manual assignment (used with mode='assign')
        seed: optional int for deterministic shuffle
    """
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    subnets = data["Subnets"]
    # Collect all hosts and subnet sizes
    all_hosts = []
    sizes = {}
    for s, sinfo in subnets.items():
        all_hosts.extend(sinfo["Hosts"])
        sizes[s] = len(sinfo["Hosts"])

    if mode == 'shuffle':
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_hosts)
        i = 0
        for s, size in sizes.items():
            subnets[s]["Hosts"] = all_hosts[i:i+size]
            i += size

    elif mode == 'assign':
        assigned = sorted([h for hs in new_assignments.values() for h in hs])
        assert sorted(all_hosts) == assigned, "Assignments must use same hosts!"
        for s in subnets:
            subnets[s]["Hosts"] = list(new_assignments.get(s, []))
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    data["Subnets"] = subnets

    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False)

# -------- Usage Example --------

# # Path to your YAML (edit as needed)
# yaml_path = "Scenario2.yaml"

# Shuffle randomly:
#update_yaml_file(yaml_path, mode='shuffle', seed=42)

# # OR, for manual assignment:

# new_assignments = {
#     'Enterprise': ['User0', 'Enterprise2', 'Enterprise1', 'Op_Host2'],
#     'Operational': ['Op_Host1', 'Enterprise0', 'User4', 'Op_Server0'],
#     'User': ['Op_Host0', 'User3', 'User1', 'Defender', 'User2']
# }
# update_yaml_file(yaml_path, mode='assign', new_assignments=new_assignments)

