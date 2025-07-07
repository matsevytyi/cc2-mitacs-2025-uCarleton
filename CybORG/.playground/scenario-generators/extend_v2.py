#!/usr/bin/env python3
"""
Extended CAGE Challenge 2 Scenario Generator
Generates a new scenario YAML file with Left and Right subnets added to the original Scenario2.yaml

Network Topology:
Left Subnet (Linux) <--> User Subnet <--> Enterprise Subnet <--> Operational Subnet <--> Right Subnet (Windows)
"""

# Credits - Luka Santiago

import yaml
import copy
from pathlib import Path

def create_extended_scenario(input_file="Scenario2.yaml", output_file="Scenario2_Extended_v2.yaml"):
    """
    Creates an extended scenario with Left and Right subnets based on Scenario2.yaml
    
    Args:
        input_file (str): Path to the original Scenario2.yaml file
        output_file (str): Path for the new extended scenario file
    """
    
    # Load the original scenario
    try:
        with open(input_file, 'r') as f:
            scenario = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please ensure the file exists in the current directory.")
        return
    
    # Create a deep copy to avoid modifying the original
    extended_scenario = copy.deepcopy(scenario)
    
    # ========================================
    # 1. EXTEND BLUE AGENT CONFIGURATION
    # ========================================
    
    # Add Left and Right subnets to allowed subnets
    extended_scenario['Agents']['Blue']['AllowedSubnets'].extend(['Left', 'Right'])
    
    # Add Left subnet hosts to Blue agent intelligence
    left_hosts = {
        'Left0': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'},
        'Left1': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'},
        'Left2': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'},
        'Left3': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'}
    }
    extended_scenario['Agents']['Blue']['INT']['Hosts'].update(left_hosts)
    
    # Add Right subnet hosts to Blue agent intelligence
    right_hosts = {
        'Right0': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'},
        'Right1': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'},
        'Right2': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'},
        'Right3': {'Interfaces': 'All', 'System info': 'All', 'User info': 'All'}
    }
    extended_scenario['Agents']['Blue']['INT']['Hosts'].update(right_hosts)
    
    # Add VelociraptorClient sessions for Left subnet
    left_sessions = [
        {'hostname': 'Left0', 'name': 'VeloLeft0', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'ubuntu'},
        {'hostname': 'Left1', 'name': 'VeloLeft1', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'ubuntu'},
        {'hostname': 'Left2', 'name': 'VeloLeft2', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'ubuntu'},
        {'hostname': 'Left3', 'name': 'VeloLeft3', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'ubuntu'}
    ]
    extended_scenario['Agents']['Blue']['starting_sessions'].extend(left_sessions)
    
    # Add VelociraptorClient sessions for Right subnet
    right_sessions = [
        {'hostname': 'Right0', 'name': 'VeloRight0', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'SYSTEM'},
        {'hostname': 'Right1', 'name': 'VeloRight1', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'SYSTEM'},
        {'hostname': 'Right2', 'name': 'VeloRight2', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'SYSTEM'},
        {'hostname': 'Right3', 'name': 'VeloRight3', 'parent': 'VeloServer', 'type': 'VelociraptorClient', 'username': 'SYSTEM'}
    ]
    extended_scenario['Agents']['Blue']['starting_sessions'].extend(right_sessions)
    
    # ========================================
    # 2. EXTEND GREEN AGENT CONFIGURATION
    # ========================================
    
    # Add Left and Right subnets to Green agent allowed subnets
    extended_scenario['Agents']['Green']['AllowedSubnets'].extend(['Left', 'Right'])
    
    # Add Left and Right hosts to Green agent intelligence
    extended_scenario['Agents']['Green']['INT']['Hosts'].update(left_hosts)
    extended_scenario['Agents']['Green']['INT']['Hosts'].update(right_hosts)
    
    # Add Green sessions for Left subnet
    left_green_sessions = [
        {'hostname': 'Left0', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'},
        {'hostname': 'Left1', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'},
        {'hostname': 'Left2', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'},
        {'hostname': 'Left3', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'}
    ]
    extended_scenario['Agents']['Green']['starting_sessions'].extend(left_green_sessions)
    
    # Add Green sessions for Right subnet
    right_green_sessions = [
        {'hostname': 'Right0', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'},
        {'hostname': 'Right1', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'},
        {'hostname': 'Right2', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'},
        {'hostname': 'Right3', 'name': 'GreenSession', 'type': 'green_session', 'username': 'GreenAgent'}
    ]
    extended_scenario['Agents']['Green']['starting_sessions'].extend(right_green_sessions)
    
    # ========================================
    # 3. EXTEND RED AGENT CONFIGURATION
    # ========================================
    
    # Add Left and Right subnets to Red agent allowed subnets
    extended_scenario['Agents']['Red']['AllowedSubnets'].extend(['Left', 'Right'])
    
    # ========================================
    # 4. ADD NEW HOST DEFINITIONS
    # ========================================
    
    # Define Left subnet hosts (Linux)
    left_host_definitions = {
        'Left0': {
            'AWS_Info': [],
            'image': 'linux_user_host3',
            'info': {
                'Left0': {'Interfaces': 'All'},
                'User1': {'Interfaces': 'IP Address'},  # Connection to Windows users
                'User2': {'Interfaces': 'IP Address'}
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        },
        'Left1': {
            'AWS_Info': [],
            'image': 'linux_user_host4',
            'info': {
                'Left1': {'Interfaces': 'All'},
                'User1': {'Interfaces': 'IP Address'},  # Connection to Windows users
                'User2': {'Interfaces': 'IP Address'}
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        },
        'Left2': {
            'AWS_Info': [],
            'image': 'linux_user_host3',
            'info': {
                'Left2': {'Interfaces': 'All'},
                'User3': {'Interfaces': 'IP Address'},  # Connection to Linux users
                'User4': {'Interfaces': 'IP Address'}
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        },
        'Left3': {
            'AWS_Info': [],
            'image': 'linux_user_host4',
            'info': {
                'Left3': {'Interfaces': 'All'},
                'User3': {'Interfaces': 'IP Address'},  # Connection to Linux users
                'User4': {'Interfaces': 'IP Address'}
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        }
    }
    extended_scenario['Hosts'].update(left_host_definitions)
    
    # Define Right subnet hosts (Windows)
    right_host_definitions = {
        'Right0': {
            'AWS_Info': [],
            'image': 'windows_user_host3',
            'info': {
                'Right0': {'Interfaces': 'All'},
                'Op_Host0': {'Interfaces': 'IP Address'},  # Connection to operational hosts
                'Op_Host1': {'Interfaces': 'IP Address'}
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        },
        'Right1': {
            'AWS_Info': [],
            'image': 'windows_user_host4',
            'info': {
                'Right1': {'Interfaces': 'All'},
                'Op_Host0': {'Interfaces': 'IP Address'},  # Connection to operational hosts
                'Op_Host1': {'Interfaces': 'IP Address'}
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        },
        'Right2': {
            'AWS_Info': [],
            'image': 'windows_user_host3',
            'info': {
                'Right2': {'Interfaces': 'All'},
                'Op_Host2': {'Interfaces': 'IP Address'}  # Connection to operational host
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        },
        'Right3': {
            'AWS_Info': [],
            'image': 'windows_user_host4',
            'info': {
                'Right3': {'Interfaces': 'All'},
                'Op_Host2': {'Interfaces': 'IP Address'}  # Connection to operational host
            },
            'ConfidentialityValue': 'None',
            'AvailabilityValue': 'None'
        }
    }
    extended_scenario['Hosts'].update(right_host_definitions)
    
    # ========================================
    # 5. UPDATE EXISTING OPERATIONAL HOSTS
    # ========================================
    
    # Add Right subnet knowledge to operational hosts
    extended_scenario['Hosts']['Op_Host0']['info'].update({
        'Right0': {'Interfaces': 'IP Address'},
        'Right1': {'Interfaces': 'IP Address'}
    })
    
    extended_scenario['Hosts']['Op_Host1']['info'].update({
        'Right0': {'Interfaces': 'IP Address'},
        'Right1': {'Interfaces': 'IP Address'}
    })
    
    extended_scenario['Hosts']['Op_Host2']['info'].update({
        'Right2': {'Interfaces': 'IP Address'},
        'Right3': {'Interfaces': 'IP Address'}
    })
    
    # ========================================
    # 6. ADD NEW SUBNET DEFINITIONS
    # ========================================
    
    # Define Left subnet
    left_subnet = {
        'Hosts': ['Left0', 'Left1', 'Left2', 'Left3'],
        'NACLs': {
            'all': {'in': 'all', 'out': 'all'}
        },
        'Size': 4
    }
    extended_scenario['Subnets']['Left'] = left_subnet
    
    # Define Right subnet with security restrictions
    right_subnet = {
        'Hosts': ['Right0', 'Right1', 'Right2', 'Right3'],
        'NACLs': {
            'Enterprise': {'in': 'None', 'out': 'all'},  # Prevent direct Enterprise access
            'all': {'in': 'all', 'out': 'all'}
        },
        'Size': 4
    }
    extended_scenario['Subnets']['Right'] = right_subnet
    
    # ========================================
    # 7. SAVE THE EXTENDED SCENARIO
    # ========================================
    
    try:
        with open(output_file, 'w') as f:
            yaml.dump(extended_scenario, f, default_flow_style=False, indent=2, sort_keys=False)
        
        print(f"✅ Extended scenario successfully created: {output_file}")
        print(f"📊 Network summary:")
        print(f"   • Left Subnet: 4 Linux hosts (Left0-3)")
        print(f"   • User Subnet: 5 hosts (User0-4) - Original")
        print(f"   • Enterprise Subnet: 4 hosts - Original")
        print(f"   • Operational Subnet: 4 hosts - Original")
        print(f"   • Right Subnet: 4 Windows hosts (Right0-3)")
        print(f"   • Total hosts: 21 (13 original + 8 new)")
        print(f"   • Monitoring: All hosts covered by Velociraptor")
        
    except Exception as e:
        print(f"❌ Error writing extended scenario: {e}")

def validate_scenario(file_path):
    """
    Validates the generated scenario file
    
    Args:
        file_path (str): Path to the scenario file to validate
    """
    try:
        with open(file_path, 'r') as f:
            scenario = yaml.safe_load(f)
        
        print(f"\n🔍 Validating {file_path}...")
        
        # Check agents
        agents = scenario.get('Agents', {})
        print(f"   • Agents: {list(agents.keys())}")
        
        # Check subnets
        subnets = scenario.get('Subnets', {})
        print(f"   • Subnets: {list(subnets.keys())}")
        
        # Check total hosts
        hosts = scenario.get('Hosts', {})
        print(f"   • Total hosts: {len(hosts)}")
        
        # Check Blue agent coverage
        blue_intel = scenario.get('Agents', {}).get('Blue', {}).get('INT', {}).get('Hosts', {})
        print(f"   • Blue agent intelligence covers: {len(blue_intel)} hosts")
        
        # Check monitoring sessions
        blue_sessions = scenario.get('Agents', {}).get('Blue', {}).get('starting_sessions', [])
        velo_sessions = [s for s in blue_sessions if s.get('type') == 'VelociraptorClient']
        print(f"   • VelociraptorClient sessions: {len(velo_sessions)}")
        
        print("✅ Scenario validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation error: {e}")

def print_network_topology(scenario_file):
    """
    Prints a summary of the network topology
    
    Args:
        scenario_file (str): Path to the scenario file
    """
    try:
        with open(scenario_file, 'r') as f:
            scenario = yaml.safe_load(f)
        
        print(f"\n🌐 Network Topology for {scenario_file}:")
        print("=" * 60)
        
        subnets = scenario.get('Subnets', {})
        hosts = scenario.get('Hosts', {})
        
        for subnet_name, subnet_info in subnets.items():
            print(f"\n📍 {subnet_name.upper()} SUBNET:")
            print(f"   Size: {subnet_info.get('Size', 'Unknown')}")
            print(f"   Hosts: {', '.join(subnet_info.get('Hosts', []))}")
            
            # Show host details
            for host in subnet_info.get('Hosts', []):
                if host in hosts:
                    host_info = hosts[host]
                    image = host_info.get('image', 'Unknown')
                    print(f"     - {host}: {image}")
        
        print("\n🔗 Attack Path Summary:")
        print("   Left → User → Enterprise → Operational → Right")
        print("   (Linux) → (Mixed) → (Business) → (Critical) → (Windows)")
        
    except Exception as e:
        print(f"❌ Error reading topology: {e}")

if __name__ == "__main__":
    # Generate the extended scenario
    create_extended_scenario()
    
    # Validate the generated file
    validate_scenario("Scenario2_Extended.yaml")
    
    # Print network topology
    print_network_topology("Scenario2_Extended.yaml")
    
    print(f"\n🎯 Usage Instructions:")
    print(f"   1. Ensure Scenario2.yaml is in the current directory")
    print(f"   2. Run this script to generate Scenario2_Extended.yaml")
    print(f"   3. Use the extended scenario in your CAGE Challenge environment")
    print(f"   4. Configure CybORG to use the new scenario file")