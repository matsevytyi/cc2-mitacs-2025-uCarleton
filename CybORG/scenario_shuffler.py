import yaml
import numpy as np
import random

"""
Developed by Andrii Matsevytyi
As a Mitacs intern
Under prof. Gie Gao and prof. Thomas Kunz supervision
At the University of Carleton


Created for testing performance of Transformer Encoder to get modified observation space 
while preserving old action space compatibility
"""

def churn_hosts(filepath):

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    subnets = data["Subnets"]
    new_assignments = {}

    # TODO: move to the config
    enterprise_churn_rate = (0.02, 0.03)
    user_networks_churn_rate = (0.05, 0.5)
    operational_host_churn_rate = (0.02, 0.06)


    sizes, all_hosts = extract_hosts(subnets) 

    counter = 0

    for key in sizes.keys():

        new_assignments[key] = []

        if key == 'Enterprise':
            for i in range(sizes[key]):
                new_assignments[key].append(all_hosts[i])
                counter += 1
            
            churn_action, churn_quantity = calculate_absolute_churn(enterprise_churn_rate[0], enterprise_churn_rate[1], sizes[key])
            #print(f"[{key.capitalize()} subnet] selected action '{churn_action}' for '{churn_quantity} devices'")
            
        if key == 'Operational':
            for i in range(counter, counter + sizes[key]):
                new_assignments[key].append(all_hosts[i])
                counter += 1

            churn_action, churn_quantity = calculate_absolute_churn(operational_host_churn_rate[0], operational_host_churn_rate[1], sizes[key])
            #print(f"[{key.capitalize()} subnet] selected action '{churn_action}' for '{churn_quantity} devices'")

        if key == 'User':
            for i in range(counter, counter + sizes[key]):
                new_assignments[key].append(all_hosts[i])
                counter += 1
            
            churn_action, churn_quantity = calculate_absolute_churn(user_networks_churn_rate[0], user_networks_churn_rate[1], sizes[key])
            #print(f"[{key.capitalize()} subnet] selected action '{churn_action}' for '{churn_quantity} devices'")

            if churn_quantity != 0:
                modify_subnet_hosts(filepath, key, action=churn_action, num_hosts=churn_quantity)

    return new_assignments

def calculate_absolute_churn(min_rate, max_rate, current_hosts_in_subnet):

    # random 
    churn_type = random.choice(['join', 'leave'])

    # random choice based on distribution
    churn_rate = np.random.uniform(min_rate, max_rate)

    # Calculate expected number of hosts affected
    num_affected = churn_rate * current_hosts_in_subnet

    # max - 3 hosts is left most - 11 hosts are kept
    if churn_type == 'leave':
        num_affected = min(int(round(num_affected)), current_hosts_in_subnet - 3)
    else:
        num_affected = min(int(round(num_affected)), 14 - current_hosts_in_subnet)

    return churn_type, num_affected

# -- low-level assignment (ecpilicitely tell what to reassign)
def extract_hosts(subnets):
    # data is yaml markup

    all_hosts = []
    sizes = {}

    for s, sinfo in subnets.items():
        all_hosts.extend(sinfo["Hosts"])
        sizes[s] = len(sinfo["Hosts"])

    return sizes, all_hosts

def modify_subnet_hosts(yaml_path, subnet_name, action='join', num_hosts=1):
    import yaml, copy
    
    with open(yaml_path, 'r') as f:
        scenario = yaml.safe_load(f)
    
    hosts_list = scenario['Subnets'][subnet_name]['Hosts']
    
    if action == 'join':
        max_num = max([int(''.join(filter(str.isdigit, h))) for h in hosts_list] + [-1])
        template_host = hosts_list[0]
        template = scenario['Hosts'][template_host]
        
        for i in range(num_hosts):
            new_host = f"{subnet_name}{max_num + i + 1}"
            hosts_list.append(new_host)
            
            # Copy host config
            scenario['Hosts'][new_host] = copy.deepcopy(template)
            scenario['Hosts'][new_host]['info'] = {new_host: template['info'][template_host]}
            
            # Add to Blue INT (Velociraptor)
            scenario['Agents']['Blue']['INT']['Hosts'][new_host] = {
                'Interfaces': 'All',
                'System info': 'All',
                'User info': 'All'
            }
            scenario['Agents']['Blue']['starting_sessions'].append({
                'hostname': new_host,
                'name': f'Velo{new_host}',
                'parent': 'VeloServer',
                'type': 'VelociraptorClient',
                'username': template.get('username', 'ubuntu')
            })
            
            # Add to Green INT
            scenario['Agents']['Green']['INT']['Hosts'][new_host] = {
                'Interfaces': 'All',
                'System info': 'All',
                'User info': 'All'
            }
            scenario['Agents']['Green']['starting_sessions'].append({
                'hostname': new_host,
                'name': 'GreenSession',
                'type': 'green_session',
                'username': 'GreenAgent'
            })
    
    elif action == 'leave':
        for _ in range(num_hosts):
            host = hosts_list.pop()
            del scenario['Hosts'][host]
            del scenario['Agents']['Blue']['INT']['Hosts'][host]
            del scenario['Agents']['Green']['INT']['Hosts'][host]
            scenario['Agents']['Blue']['starting_sessions'] = [s for s in scenario['Agents']['Blue']['starting_sessions'] if s['hostname'] != host]
            scenario['Agents']['Green']['starting_sessions'] = [s for s in scenario['Agents']['Green']['starting_sessions'] if s['hostname'] != host]
    
    scenario['Subnets'][subnet_name]['Size'] = len(hosts_list)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
    
    #print(f"{action.upper()}: {subnet_name} now has {len(hosts_list)} hosts")

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

    # Collect all hosts and subnet sizes

    subnets = data["Subnets"]

    sizes, all_hosts = extract_hosts(subnets)
    #print(sizes, all_hosts)
    
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
#yaml_path = "cage-challenge-2/CybORG/.playground/scenarios/Scenario2.yaml"
# yaml_path = "cage-challenge-2/CybORG/.playground/scenarios/Scenario2_more_hosts.yaml"

# res = churn_hosts(yaml_path)

# print(res)
            




# Shuffle randomly:
#update_yaml_file(yaml_path, mode='shuffle', seed=42)

# # OR, for manual assignment:

# new_assignments = {
#     'Enterprise': ['User0', 'Enterprise2', 'Enterprise1', 'Op_Host2'],
#     'Operational': ['Op_Host1', 'Enterprise0', 'User4', 'Op_Server0'],
#     'User': ['Op_Host0', 'User3', 'User1', 'Defender', 'User2']
# }
# update_yaml_file(yaml_path, mode='assign', new_assignments=new_assignments)

