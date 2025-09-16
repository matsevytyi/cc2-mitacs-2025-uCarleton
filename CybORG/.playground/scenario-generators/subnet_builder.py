"""
Enhanced Interactive CAGE Challenge Subnet Builder V3.0

This script allows users to dynamically add new subnets to existing CAGE Challenge scenarios.
Key improvements:
- Preserves original scenario connection patterns
- Works with any base scenario (not just Scenario2)
- Smart connection strategies that don't disrupt existing topology
- Enhanced connection options for base scenario elements

"""

import yaml, textwrap, datetime, sys, copy, string, random, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class EnhancedSubnetBuilder:
    """Enhanced subnet builder for CAGE Challenge scenarios with pattern preservation"""
    
    def __init__(self):
        self.scenario_data = None
        self.scenario_path = None
        self.original_connections = {}  # Store original connectivity patterns
        self.critical_paths = []  # Track important connection paths
        self.original_bidirectional_pairs = []
        self.host_types = {
            'linux': ['linux_user_host1', 'linux_user_host2', 'linux_user_host3', 'linux_user_host4'],
            'windows': ['windows_user_host1', 'windows_user_host2', 'windows_user_host3', 'windows_user_host4'],
            'gateway': ['Gateway'],
            'internal': ['Internal'],
            'server': ['OP_Server', 'Velociraptor_Server']
        }
        self.current_nacl_rules = {}  
        self.original_nacl_rules = {}  
        self.nacl_change_history = []
        self.ensure_additional_image_files()
        self.action_space_path = Path("../ActionSpace.py")
        self.ensure_additional_image_files()
        
    def update_action_space_limits(self) -> bool:
        """Update ActionSpace.py MAX_HOSTS and MAX_SUBNETS based on current scenario"""
        if not self.scenario_data:
            print("WARNING: No scenario loaded, skipping ActionSpace.py update")
            return False
        
        try:
            # Calculate required limits from current scenario
            current_hosts = len(self.scenario_data.get('Hosts', {}))
            current_subnets = len(self.scenario_data.get('Subnets', {}))
            
            # Add buffer (20% minimum, at least +5)
            required_max_hosts = max(current_hosts + 5, int(current_hosts * 1.2))
            required_max_subnets = max(current_subnets + 2, int(current_subnets * 1.2))
            
            print(f"Checking ActionSpace.py limits...")
            print(f"  Current scenario: {current_hosts} hosts, {current_subnets} subnets")
            print(f"  Required limits: MAX_HOSTS >= {required_max_hosts}, MAX_SUBNETS >= {required_max_subnets}")
            
            # Check if ActionSpace.py exists
            if not self.action_space_path.exists():
                print(f"WARNING: ActionSpace.py not found at {self.action_space_path}")
                print("Skipping ActionSpace.py update")
                return False
            
            # Read current limits
            current_max_hosts, current_max_subnets = self._read_action_space_limits()
            
            if current_max_hosts is None or current_max_subnets is None:
                print("WARNING: Could not read current ActionSpace.py limits")
                return False
            
            print(f"  Current limits: MAX_HOSTS = {current_max_hosts}, MAX_SUBNETS = {current_max_subnets}")
            
            # Check if update is needed
            needs_update = False
            new_max_hosts = current_max_hosts
            new_max_subnets = current_max_subnets
            
            if current_max_hosts < required_max_hosts:
                new_max_hosts = required_max_hosts
                needs_update = True
                print(f"  MAX_HOSTS needs update: {current_max_hosts} -> {new_max_hosts}")
            
            if current_max_subnets < required_max_subnets:
                new_max_subnets = required_max_subnets
                needs_update = True
                print(f"  MAX_SUBNETS needs update: {current_max_subnets} -> {new_max_subnets}")
            
            if not needs_update:
                print("  ActionSpace.py limits are sufficient")
                return True
            
            # Update ActionSpace.py
            if self._update_action_space_file(new_max_hosts, new_max_subnets):
                print(f"    ActionSpace.py updated successfully!")
                print(f"    New limits: MAX_HOSTS = {new_max_hosts}, MAX_SUBNETS = {new_max_subnets}")
                return True
            else:
                print(f"    Failed to update ActionSpace.py")
                return False
                
        except Exception as e:
            print(f"ERROR updating ActionSpace.py: {str(e)}")
            return False

    def _read_action_space_limits(self) -> Tuple[Optional[int], Optional[int]]:
        """Read current MAX_HOSTS and MAX_SUBNETS from ActionSpace.py"""
        try:
            with open(self.action_space_path, 'r') as f:
                content = f.read()
            
            max_hosts = None
            max_subnets = None
            
            # Parse MAX_HOSTS
            import re
            hosts_match = re.search(r'^MAX_HOSTS\s*=\s*(\d+)', content, re.MULTILINE)
            if hosts_match:
                max_hosts = int(hosts_match.group(1))
            
            # Parse MAX_SUBNETS  
            subnets_match = re.search(r'^MAX_SUBNETS\s*=\s*(\d+)', content, re.MULTILINE)
            if subnets_match:
                max_subnets = int(subnets_match.group(1))
            
            return max_hosts, max_subnets
            
        except Exception as e:
            print(f"Error reading ActionSpace.py: {str(e)}")
            return None, None

    def _update_action_space_file(self, new_max_hosts: int, new_max_subnets: int) -> bool:
        """Update MAX_HOSTS and MAX_SUBNETS in ActionSpace.py"""
        try:
            # Read current file
            with open(self.action_space_path, 'r') as f:
                content = f.read()
            
            # Create backup
            backup_path = self.action_space_path.with_suffix('.py.backup')
            with open(backup_path, 'w') as f:
                f.write(content)
            
            import re
            
            # Update MAX_HOSTS
            hosts_pattern = r'^(MAX_HOSTS\s*=\s*)\d+'
            hosts_replacement = f'\\g<1>{new_max_hosts}'
            content = re.sub(hosts_pattern, hosts_replacement, content, flags=re.MULTILINE)
            
            # Update MAX_SUBNETS
            subnets_pattern = r'^(MAX_SUBNETS\s*=\s*)\d+'
            subnets_replacement = f'\\g<1>{new_max_subnets}'
            content = re.sub(subnets_pattern, subnets_replacement, content, flags=re.MULTILINE)
            
            # Write updated file
            with open(self.action_space_path, 'w') as f:
                f.write(content)
            
            # Verify the update worked
            verify_max_hosts, verify_max_subnets = self._read_action_space_limits()
            
            if verify_max_hosts == new_max_hosts and verify_max_subnets == new_max_subnets:
                print(f"    Backup created: {backup_path}")
                return True
            else:
                # Restore backup if verification failed
                with open(backup_path, 'r') as f:
                    original_content = f.read()
                with open(self.action_space_path, 'w') as f:
                    f.write(original_content)
                print(f"    Update verification failed, restored from backup")
                return False
                
        except Exception as e:
            print(f"Error updating ActionSpace.py: {str(e)}")
            return False

    def _create_action_space_if_missing(self, required_max_hosts: int, required_max_subnets: int) -> bool:
        """Create a basic ActionSpace.py if it doesn't exist"""
        try:
            print(f"Creating basic ActionSpace.py...")
            
            action_space_content = f'''# ActionSpace.py - Auto-generated by Enhanced Subnet Builder
    # Defines maximum limits for CAGE Challenge scenarios

    # Maximum number of hosts in the environment
    MAX_HOSTS = {required_max_hosts}

    # Maximum number of subnets in the environment  
    MAX_SUBNETS = {required_max_subnets}

    # Additional ActionSpace configurations can be added here
    # This file was auto-created because no ActionSpace.py was found
    '''
            
            with open(self.action_space_path, 'w') as f:
                f.write(action_space_content)
            
            print(f"  OK Created ActionSpace.py with MAX_HOSTS={required_max_hosts}, MAX_SUBNETS={required_max_subnets}")
            return True
            
        except Exception as e:
            print(f"Error creating ActionSpace.py: {str(e)}")
            return False
    
    def ensure_additional_image_files(self):
        """Ensure additional image files (3 & 4) exist for linux and windows hosts"""
        try:
            print("Checking for additional image files...")
            
            # Define the image directory path
            images_dir = Path("images")
            if not images_dir.exists():
                print(f"Warning: Images directory '{images_dir}' not found. Skipping image file creation.")
                return
            
            # Define required files
            required_files = [
                "linux_user_host_image3.yaml",
                "linux_user_host_image4.yaml", 
                "windows_user_host_image3.yaml",
                "windows_user_host_image4.yaml"
            ]
            
            # Check which files are missing
            missing_files = []
            for file_name in required_files:
                file_path = images_dir / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if not missing_files:
                print(" All additional image files already exist")
                self._update_host_types_with_additional_images()
                return
            
            print(f"Creating {len(missing_files)} missing image files...")
            
            # Create missing files
            for file_name in missing_files:
                if self._create_image_file(images_dir, file_name):
                    print(f"   Created: {file_name}")
                else:
                    print(f"   Failed: {file_name}")
            
            # Update images.yaml
            if self._update_images_yaml(images_dir):
                print(" Updated images.yaml registry")
            else:
                print(" Failed to update images.yaml")
            
            # Update host types
            self._update_host_types_with_additional_images()
            
            print("Additional image files setup completed!")
            
        except Exception as e:
            print(f"Error ensuring additional image files: {str(e)}")
            
    def _update_images_yaml(self, images_dir: Path) -> bool:
        """Update images.yaml with new image entries"""
        try:
            images_yaml_path = images_dir / "images.yaml"
            
            # Load existing images.yaml
            if images_yaml_path.exists():
                with open(images_yaml_path, 'r') as f:
                    images_data = yaml.safe_load(f) or {}
            else:
                images_data = {}
            
            # Add new entries (only if they don't exist)
            new_entries = {
                'linux_user_host3': {'path': 'linux_user_host_image3'},
                'linux_user_host4': {'path': 'linux_user_host_image4'},
                'windows_user_host3': {'path': 'windows_user_host_image3'},
                'windows_user_host4': {'path': 'windows_user_host_image4'}
            }
            
            added_count = 0
            for key, value in new_entries.items():
                if key not in images_data:
                    images_data[key] = value
                    added_count += 1
            
            if added_count > 0:
                # Save updated images.yaml
                with open(images_yaml_path, 'w') as f:
                    yaml.dump(images_data, f, default_flow_style=False, sort_keys=False)
                print(f"  Added {added_count} new entries to images.yaml")
            
            return True
            
        except Exception as e:
            print(f"Error updating images.yaml: {str(e)}")
            return False

    def _update_host_types_with_additional_images(self):
        """Update the host_types dictionary to include new image options"""
        self.host_types = {
            'linux': ['linux_user_host1', 'linux_user_host2', 'linux_user_host3', 'linux_user_host4'],
            'windows': ['windows_user_host1', 'windows_user_host2', 'windows_user_host3', 'windows_user_host4'],
            'gateway': ['Gateway'],
            'internal': ['Internal'],
            'server': ['OP_Server', 'Velociraptor_Server']
        }

    def _create_image_file(self, images_dir: Path, file_name: str) -> bool:
        """Create a single image file based on template"""
        try:
            # Determine source template
            if file_name == "linux_user_host_image3.yaml":
                template_file = "linux_user_host_image1.yaml"
            elif file_name == "linux_user_host_image4.yaml":
                template_file = "linux_user_host_image2.yaml"
            elif file_name == "windows_user_host_image3.yaml":
                template_file = "windows_user_host_image1.yaml"
            elif file_name == "windows_user_host_image4.yaml":
                template_file = "windows_user_host_image2.yaml"
            else:
                return False
            
            # Load template
            template_path = images_dir / template_file
            if not template_path.exists():
                print(f"Template file not found: {template_file}")
                return False
            
            with open(template_path, 'r') as f:
                template_content = yaml.safe_load(f)
            
            # Randomize credentials
            if "linux" in file_name:
                template_content = self._randomize_linux_credentials(template_content)
            else:  # windows
                template_content = self._randomize_windows_credentials(template_content)
            
            # Save new file
            new_file_path = images_dir / file_name
            with open(new_file_path, 'w') as f:
                yaml.dump(template_content, f, default_flow_style=False, sort_keys=False)
            
            return True
            
        except Exception as e:
            print(f"Error creating {file_name}: {str(e)}")
            return False

    def _randomize_linux_credentials(self, content: dict) -> dict:
        """Randomize credentials for Linux image files"""
        try:
            user_info = content.get('Test_Host', {}).get('User Info', [])
            
            for user in user_info:
                # Only randomize the 'pi' user that has 'Bruteforceable: True'
                if (user.get('Username') == 'pi' and 
                    user.get('Bruteforceable') == True and 
                    'Password' in user):
                    
                    # Generate new credentials
                    new_username = self._generate_username()
                    new_password = self._generate_password()
                    
                    user['Username'] = new_username
                    user['Password'] = new_password
                    
                    # Update the group name to match new username
                    if 'Groups' in user and len(user['Groups']) > 0:
                        user['Groups'][0]['Group Name'] = new_username
            
            return content
            
        except Exception as e:
            print(f"Error randomizing Linux credentials: {str(e)}")
            return content

    def _randomize_windows_credentials(self, content: dict) -> dict:
        """Randomize credentials for Windows image files"""
        try:
            user_info = content.get('Test_Host', {}).get('User Info', [])
            
            for user in user_info:
                # Only randomize the 'vagrant' user that has 'Bruteforceable: True'
                if (user.get('Username') == 'vagrant' and 
                    user.get('Bruteforceable') == True and 
                    'Password' in user):
                    
                    # Generate new credentials
                    new_username = self._generate_username()
                    new_password = self._generate_password()
                    
                    user['Username'] = new_username
                    user['Password'] = new_password
            
            return content
            
        except Exception as e:
            print(f"Error randomizing Windows credentials: {str(e)}")
            return content

    def _generate_username(self) -> str:
        """Generate random username (8-12 characters, alphanumeric)"""
        length = random.randint(8, 12)
        # Start with a letter, then alphanumeric
        first_char = random.choice(string.ascii_lowercase)
        remaining_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length-1))
        return first_char + remaining_chars

    def _generate_password(self) -> str:
        """Generate secure random password (12-16 characters)"""
        length = random.randint(12, 16)
        
        # Ensure password has at least one of each type
        password_chars = []
        password_chars.append(random.choice(string.ascii_uppercase))  # At least one uppercase
        password_chars.append(random.choice(string.ascii_lowercase))  # At least one lowercase  
        password_chars.append(random.choice(string.digits))          # At least one digit
        password_chars.append(random.choice('!@#$%^&*'))            # At least one symbol
        
        # Fill remaining length with random mix
        remaining_length = length - 4
        all_chars = string.ascii_letters + string.digits + '!@#$%^&*'
        password_chars.extend(random.choices(all_chars, k=remaining_length))
        
        # Shuffle to avoid predictable patterns
        random.shuffle(password_chars)
        
        return ''.join(password_chars)
        
    def _analyze_original_patterns(self) -> None:
        """Analyze and store original connectivity patterns for preservation"""
        self.original_connections = {}
        self.original_bidirectional_pairs = []  # NEW: Track bidirectional pairs
        hosts = self.scenario_data.get('Hosts', {})
        
        # Store all original host-to-host connections
        for host_name, host_data in hosts.items():
            connections = []
            host_info = host_data.get('info', {})
            
            for connected_host, details in host_info.items():
                if connected_host != host_name and details.get('Interfaces') == 'IP Address':
                    connections.append(connected_host)
            
            self.original_connections[host_name] = connections.copy()
        
        # NEW: Identify bidirectional pairs that MUST be preserved
        self._identify_bidirectional_pairs()
        
        # Identify critical paths (hosts with multiple subnet connections)
        self._identify_critical_paths()
        
    def _identify_bidirectional_pairs(self) -> None:
        """Identify bidirectional connection pairs that must be preserved"""
        self.original_bidirectional_pairs = []
        processed_pairs = set()
        
        for host_a, connections_a in self.original_connections.items():
            for connected_host_b in connections_a:
                # Check if reverse connection exists
                connections_b = self.original_connections.get(connected_host_b, [])
                if host_a in connections_b:
                    # This is a bidirectional pair
                    pair = tuple(sorted([host_a, connected_host_b]))
                    if pair not in processed_pairs:
                        self.original_bidirectional_pairs.append(pair)
                        processed_pairs.add(pair)
                        print(f"   Detected bidirectional pair: {pair[0]} -- {pair[1]}")   
                        
    def analyze_current_nacls(self) -> None:
        """Analyze and store current NACL configurations for preservation"""
        self.current_nacl_rules = {}
        self.original_nacl_rules = {}
        
        subnets = self.scenario_data.get('Subnets', {})
        
        print("   Analyzing NACL configurations...")
        
        for subnet_name, subnet_data in subnets.items():
            nacl_rules = subnet_data.get('NACLs', {})
            
            # Store current rules
            self.current_nacl_rules[subnet_name] = copy.deepcopy(nacl_rules)
            self.original_nacl_rules[subnet_name] = copy.deepcopy(nacl_rules)
            
            rule_count = len(nacl_rules)
            print(f"     {subnet_name}: {rule_count} NACL rules")

    def get_available_nacl_targets(self, subnet_name: str) -> List[str]:
        """Return list of valid NACL target options for a subnet"""
        targets = ['all']  # Always include 'all' as default option
        
        # Add all other subnet names as targets
        subnets = self.scenario_data.get('Subnets', {})
        for other_subnet in subnets.keys():
            if other_subnet != subnet_name:
                targets.append(other_subnet)
        
        return targets                    
                         
    def load_scenario(self, file_path: str) -> bool:
        """Load CAGE scenario from YAML file and analyze connectivity patterns"""
        try:
            self.scenario_path = Path(file_path)
            if not self.scenario_path.exists():
                print(f"ERROR: File '{file_path}' not found!")
                return False
                
            with open(self.scenario_path, 'r') as file:
                self.scenario_data = yaml.safe_load(file)
            
            # Analyze and store original connectivity patterns
            self._analyze_original_patterns()
            #self.analyze_current_nacls()
            
            print(f"Successfully loaded scenario: {file_path}")
            print(f"Detected {len(self.scenario_data.get('Subnets', {}))} subnets")
            print(f"Detected {len(self.scenario_data.get('Hosts', {}))} hosts")
            return True
            
        except Exception as e:
            print(f"Error loading scenario: {str(e)}")
            return False

    def _identify_critical_paths(self) -> None:
        """Identify critical connection paths that should be preserved"""
        self.critical_paths = []
        subnets = self.scenario_data.get('Subnets', {})
        
        # Find hosts that connect multiple subnets (critical gateway hosts)
        for host_name, connections in self.original_connections.items():
            host_subnet = self.find_host_subnet(host_name)
            external_subnets = set()
            
            for connected_host in connections:
                connected_subnet = self.find_host_subnet(connected_host)
                if connected_subnet != host_subnet and connected_subnet != 'Unknown':
                    external_subnets.add(connected_subnet)
            
            if len(external_subnets) >= 1:  # Host connects to external subnet(s)
                self.critical_paths.append({
                    'host': host_name,
                    'subnet': host_subnet,
                    'external_connections': list(external_subnets),
                    'connected_hosts': connections.copy()
                })

    def get_host_os_type(self, image: str) -> str:
        """Determine OS type from image name"""
        if 'windows' in image.lower():
            return 'Win'
        elif 'linux' in image.lower():
            return 'Lin'
        elif image in ['Gateway', 'Internal']:
            return 'Net'
        elif 'Server' in image:
            return 'Srv'
        else:
            return '???'

    def get_host_role(self, image: str, host_name: str) -> str:
        """Determine host role from image and name"""
        if 'Gateway' in image:
            return 'Gateway'
        elif 'Internal' in image:
            return 'Internal'
        elif 'OP_Server' in image:
            return 'Critical'
        elif 'Velociraptor' in image:
            return 'Monitor'
        elif 'Defender' in host_name:
            return 'Monitor'
        elif 'User' in host_name:
            return 'User'
        elif 'Enterprise' in host_name:
            return 'Enterprise'
        elif 'Op_' in host_name:
            return 'Ops'
        else:
            return 'Host'

    def build_connection_map(self) -> Dict[str, List[str]]:
        """Build a map of current host connections"""
        connection_map = {}
        hosts = self.scenario_data.get('Hosts', {})
        
        for host_name, host_data in hosts.items():
            connections = []
            host_info = host_data.get('info', {})
            
            for connected_host, details in host_info.items():
                if connected_host != host_name and details.get('Interfaces') == 'IP Address':
                    connections.append(connected_host)
            
            connection_map[host_name] = connections
        
        return connection_map

    def find_host_subnet(self, host_name: str) -> str:
        """Find which subnet a host belongs to"""
        subnets = self.scenario_data.get('Subnets', {})
        for subnet_name, subnet_data in subnets.items():
            if host_name in subnet_data.get('Hosts', []):
                return subnet_name
        return 'Unknown'

    def get_external_connections(self, subnet_name: str) -> Dict[str, List[str]]:
        """Get external connections for a subnet"""
        external_connections = {}
        subnet_hosts = self.scenario_data['Subnets'][subnet_name]['Hosts']
        connection_map = self.build_connection_map()
        
        for host in subnet_hosts:
            external_targets = []
            for connected_host in connection_map.get(host, []):
                target_subnet = self.find_host_subnet(connected_host)
                if target_subnet != subnet_name and target_subnet != 'Unknown':
                    external_targets.append(f"{connected_host} ({target_subnet})")
            
            if external_targets:
                external_connections[host] = external_targets
        
        return external_connections

    def display_current_subnets(self) -> None:
        """Display current subnets with enhanced hierarchical tree view"""
        if not self.scenario_data:
            print("ERROR: No scenario loaded!")
            return
            
        print("\n" + "="*80)
        print("NETWORK TOPOLOGY - HIERARCHICAL VIEW")
        print("="*80)
        
        subnets = self.scenario_data.get('Subnets', {})
        hosts = self.scenario_data.get('Hosts', {})
        total_hosts = len(hosts)
        total_subnets = len(subnets)
        
        print(f"Network Overview: {total_subnets} subnets | {total_hosts} total hosts")
        print()
        
        # Build subnet connectivity graph
        connection_map = self.build_connection_map()
        subnet_connections = {}
        
        # Determine which subnets connect to which
        for subnet_name in subnets.keys():
            subnet_connections[subnet_name] = set()
            subnet_hosts = subnets[subnet_name]['Hosts']
            
            for host in subnet_hosts:
                for connected_host in connection_map.get(host, []):
                    connected_subnet = self.find_host_subnet(connected_host)
                    if connected_subnet != 'Unknown' and connected_subnet != subnet_name:
                        subnet_connections[subnet_name].add(connected_subnet)
        
        # Find the root subnet (one with no incoming connections or contains "user")
        ordered_subnets = []
        root_subnet = None
        
        # First try to find subnet with no incoming connections
        for subnet_name in subnets.keys():
            has_incoming = False
            for other_subnet, connections in subnet_connections.items():
                if subnet_name in connections and other_subnet != subnet_name:
                    has_incoming = True
                    break
            if not has_incoming:
                root_subnet = subnet_name
                break
        
        # Fallback: look for User-like subnet
        if not root_subnet:
            for subnet_name in subnets.keys():
                if 'user' in subnet_name.lower():
                    root_subnet = subnet_name
                    break
            if not root_subnet:
                root_subnet = list(subnets.keys())[0]
        
        # Build ordered list using BFS
        ordered_subnets = [root_subnet]
        visited = {root_subnet}
        queue = [root_subnet]
        
        while queue:
            current = queue.pop(0)
            for connected in subnet_connections.get(current, []):
                if connected not in visited:
                    ordered_subnets.append(connected)
                    visited.add(connected)
                    queue.append(connected)
        
        # Add any remaining unconnected subnets
        for subnet_name in subnets.keys():
            if subnet_name not in ordered_subnets:
                ordered_subnets.append(subnet_name)
        
        # Display subnets in order
        for i, subnet_name in enumerate(ordered_subnets):
            subnet_data = subnets[subnet_name]
            host_list = subnet_data.get('Hosts', [])
            host_count = len(host_list)
            
            # Subnet header with tree structure
            is_last_subnet = (i == len(ordered_subnets) - 1)
            subnet_prefix = "└──" if is_last_subnet else "├──"
            
            print(f"{subnet_prefix} {subnet_name.upper()} SUBNET ({host_count} hosts)")
            
            # Get external connections for this subnet
            external_connections = self.get_external_connections(subnet_name)
            
            # Display hosts in tree format
            for j, host in enumerate(host_list):
                host_info = hosts.get(host, {})
                image = host_info.get('image', 'Unknown')
                conf_val = host_info.get('ConfidentialityValue', 'None')
                avail_val = host_info.get('AvailabilityValue', 'None')
                
                # Determine host characteristics
                os_type = self.get_host_os_type(image)
                role = self.get_host_role(image, host)
                
                # Tree structure for hosts
                is_last_host = (j == len(host_list) - 1)
                if is_last_subnet:
                    host_prefix = "    └──" if is_last_host else "    ├──"
                else:
                    host_prefix = "│   └──" if is_last_host else "│   ├──"
                
                # Security indicator
                security_indicator = ""
                if conf_val == 'High' or avail_val == 'High':
                    security_indicator = " H"
                elif conf_val == 'Medium' or avail_val == 'Medium':
                    security_indicator = " M"
                else:
                    security_indicator = " L"
                
                # Host line
                print(f"{host_prefix} {host} ({os_type}/{role}){security_indicator}")
                
                # Show external connections for this host
                if host in external_connections:
                    for k, connection in enumerate(external_connections[host]):
                        is_last_connection = (k == len(external_connections[host]) - 1)
                        
                        if is_last_host and is_last_subnet:
                            conn_prefix = "        └─> " if is_last_connection else "        ├─> "
                        elif is_last_host:
                            conn_prefix = "│       └─> " if is_last_connection else "│       ├─> "
                        else:
                            conn_prefix = "│   │   └─> " if is_last_connection else "│   │   ├─> "
                        
                        print(f"{conn_prefix} {connection}")
            
            # Add spacing between subnets
            if not is_last_subnet:
                print("│")
        
        print()
        print("="*80)
        
        # Show legend
        print("LEGEND:")
        print("  OS Types: Win=Windows, Lin=Linux, Net=Network, Srv=Server")
        print("  Roles: User, Enterprise, Gateway, Internal, Critical, Monitor, Ops")
        print("  Security: High (H), Medium (M), Low/None (L)")
        print("  Connections: ─> indicates external subnet connections")
        print("="*80)

    # FIXED CONNECTION METHODS
    def check_existing_connection(self, subnet_a: str, subnet_b: str) -> bool:
        """Check if two subnets are already connected"""
        if subnet_a not in self.scenario_data['Subnets'] or subnet_b not in self.scenario_data['Subnets']:
            return False
        
        hosts_a = self.scenario_data['Subnets'][subnet_a]['Hosts']
        hosts_b = self.scenario_data['Subnets'][subnet_b]['Hosts']
        
        print(f"Checking existing connection between {subnet_a} and {subnet_b}...")
        
        # Check if any host in subnet_a connects to any host in subnet_b
        for host_a in hosts_a:
            host_info = self.scenario_data['Hosts'].get(host_a, {}).get('info', {})
            for connected_host, details in host_info.items():
                if (connected_host in hosts_b and 
                    details.get('Interfaces') == 'IP Address'):
                    print(f"  Found existing connection: {host_a} -> {connected_host}")
                    return True
        
        print(f"  No existing connection found")
        return False

    def get_subnet_connection_choice(self) -> Tuple[str, str]:
        """Get user choice for which subnets to connect"""
        subnets = list(self.scenario_data.get('Subnets', {}).keys())
        
        if len(subnets) < 2:
            print("ERROR: Need at least 2 subnets to create connections!")
            return None, None
        
        print("\nAVAILABLE SUBNETS FOR CONNECTION:")
        print("="*40)
        
        # Show current connectivity status
        for i, subnet in enumerate(subnets, 1):
            hosts = self.scenario_data['Subnets'][subnet]['Hosts']
            host_count = len(hosts)
            
            # Count external connections
            external_connections = 0
            for host in hosts:
                host_info = self.scenario_data['Hosts'].get(host, {}).get('info', {})
                for conn_host, details in host_info.items():
                    if (details.get('Interfaces') == 'IP Address' and 
                        self.find_host_subnet(conn_host) != subnet):
                        external_connections += 1
            
            print(f"{i}. {subnet} ({host_count} hosts, {external_connections} external connections)")
        
        # Get source subnet
        print(f"\nSELECT SOURCE SUBNET:")
        while True:
            try:
                choice = input(f"Enter choice (1-{len(subnets)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(subnets):
                    source_subnet = subnets[choice_idx]
                    break
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(subnets)}")
            except ValueError:
                print("ERROR: Please enter a valid number")
        
        # Get target subnet (excluding source)
        remaining_subnets = [s for s in subnets if s != source_subnet]
        
        print(f"\nSELECT TARGET SUBNET:")
        print(f"Connect {source_subnet} to which subnet?")
        
        for i, subnet in enumerate(remaining_subnets, 1):
            hosts = self.scenario_data['Subnets'][subnet]['Hosts']
            host_count = len(hosts)
            
            # Check if already connected
            already_connected = self.check_existing_connection(source_subnet, subnet)
            status = " [ALREADY CONNECTED]" if already_connected else " [NOT CONNECTED]"
            
            print(f"{i}. {subnet} ({host_count} hosts){status}")
        
        while True:
            try:
                choice = input(f"Enter choice (1-{len(remaining_subnets)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(remaining_subnets):
                    target_subnet = remaining_subnets[choice_idx]
                    break
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(remaining_subnets)}")
            except ValueError:
                print("ERROR: Please enter a valid number")
        
        return source_subnet, target_subnet

    def select_best_connection_hosts(self, source_subnet: str, target_subnet: str) -> Tuple[str, str]:
        """Select the best hosts for connection with smart algorithm"""
        source_hosts = self.scenario_data['Subnets'][source_subnet]['Hosts']
        target_hosts = self.scenario_data['Subnets'][target_subnet]['Hosts']
        hosts_data = self.scenario_data.get('Hosts', {})
        
        print(f"\nSELECTING OPTIMAL CONNECTION HOSTS:")
        print(f"Source subnet: {source_subnet} ({len(source_hosts)} hosts)")
        print(f"Target subnet: {target_subnet} ({len(target_hosts)} hosts)")
        
        # Strategy 1: Auto-select optimal hosts (recommended)
        print(f"\nCONNECTION STRATEGY:")
        print("1. Auto-select optimal hosts (recommended)")
        print("2. Manual host selection")
        
        while True:
            try:
                strategy = input("Enter choice (1-2): ").strip()
                if strategy in ['1', '2']:
                    break
                else:
                    print("ERROR: Please enter 1 or 2")
            except ValueError:
                print("ERROR: Please enter a valid choice")
        
        if strategy == '1':
            # Auto-select using smart algorithm
            source_host = self._select_optimal_host(source_subnet, source_hosts)
            target_host = self._select_optimal_host(target_subnet, target_hosts)
            
            print(f"\nOPTIMAL SELECTION:")
            print(f"  Source: {source_host} ({source_subnet})")
            print(f"  Target: {target_host} ({target_subnet})")
            
            return source_host, target_host
        
        else:
            # Manual selection with detailed host info
            print(f"\nMANUAL HOST SELECTION")
            print(f"Select host from {source_subnet.upper()}:")
            
            for i, host in enumerate(source_hosts, 1):
                host_info = hosts_data.get(host, {})
                image = host_info.get('image', 'Unknown')
                os_type = self.get_host_os_type(image)
                role = self.get_host_role(image, host)
                
                # Count current connections
                current_connections = len([k for k, v in host_info.get('info', {}).items() 
                                         if v.get('Interfaces') == 'IP Address'])
                
                print(f"{i}. {host} ({os_type}/{role}) - {current_connections} connections")
            
            while True:
                try:
                    choice = input(f"Enter choice (1-{len(source_hosts)}): ").strip()
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(source_hosts):
                        source_host = source_hosts[choice_idx]
                        break
                    else:
                        print(f"ERROR: Please enter a number between 1 and {len(source_hosts)}")
                except ValueError:
                    print("ERROR: Please enter a valid number")
            
            print(f"\nSelect host from {target_subnet.upper()}:")
            
            for i, host in enumerate(target_hosts, 1):
                host_info = hosts_data.get(host, {})
                image = host_info.get('image', 'Unknown')
                os_type = self.get_host_os_type(image)
                role = self.get_host_role(image, host)
                
                # Count current connections
                current_connections = len([k for k, v in host_info.get('info', {}).items() 
                                         if v.get('Interfaces') == 'IP Address'])
                
                print(f"{i}. {host} ({os_type}/{role}) - {current_connections} connections")
            
            while True:
                try:
                    choice = input(f"Enter choice (1-{len(target_hosts)}): ").strip()
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(target_hosts):
                        target_host = target_hosts[choice_idx]
                        break
                    else:
                        print(f"ERROR: Please enter a number between 1 and {len(target_hosts)}")
                except ValueError:
                    print("ERROR: Please enter a valid number")
            
            return source_host, target_host

    def _select_optimal_host(self, subnet: str, hosts: List[str]) -> str:
        """Select optimal host for connection using smart algorithm"""
        hosts_data = self.scenario_data.get('Hosts', {})
        
        # Score each host based on multiple criteria
        host_scores = {}
        
        for host in hosts:
            host_info = hosts_data.get(host, {})
            score = 0
            
            # Criteria 1: Prefer hosts with fewer existing connections (load balancing)
            existing_connections = len([k for k, v in host_info.get('info', {}).items() 
                                      if v.get('Interfaces') == 'IP Address'])
            score += max(0, 10 - existing_connections)  # Prefer fewer connections
            
            # Criteria 2: Prefer Gateway and Internal hosts for networking
            image = host_info.get('image', '')
            if 'Gateway' in image:
                score += 15
            elif 'Internal' in image:
                score += 10
            elif 'Server' in image:
                score += 8
            else:
                score += 5
            
            # Criteria 3: Prefer hosts that already have inter-subnet connections
            for conn_host, details in host_info.get('info', {}).items():
                if (details.get('Interfaces') == 'IP Address' and 
                    self.find_host_subnet(conn_host) != subnet):
                    score += 5
                    break
            
            # Criteria 4: Avoid OP_Server (critical operational)
            if 'OP_Server' in image:
                score -= 5
            
            host_scores[host] = score
            print(f"  {host}: score={score} (connections={existing_connections}, image={image})")
        
        # Select host with highest score
        optimal_host = max(host_scores.keys(), key=lambda x: host_scores[x])
        print(f"  Selected: {optimal_host} (score={host_scores[optimal_host]})")
        
        return optimal_host

    def validate_connection_safety(self, source_host: str, target_host: str, 
                                  source_subnet: str, target_subnet: str) -> bool:
        """Simplified validation for connection safety"""
        hosts_data = self.scenario_data.get('Hosts', {})
        
        print(f"\nVALIDATING CONNECTION SAFETY:")
        print(f"  {source_host} ({source_subnet}) <--> {target_host} ({target_subnet})")
        
        # Check 1: Hosts exist
        if source_host not in hosts_data or target_host not in hosts_data:
            print(f"  FAIL ERROR: One or both hosts not found!")
            return False
        
        # Check 2: Hosts have info sections (can make connections)
        source_info = hosts_data[source_host].get('info', {})
        target_info = hosts_data[target_host].get('info', {})
        
        if not source_info or not target_info:
            print(f"  FAIL ERROR: Hosts must have network connectivity capabilities!")
            return False
        
        # Check 3: No existing direct connection
        if (target_host in source_info and 
            source_info[target_host].get('Interfaces') == 'IP Address'):
            print(f"  !! WARNING: {source_host} already connects to {target_host}")
            proceed = input("  Create connection anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return False
        
        print(f"  OK Connection validation passed")
        return True

    def create_subnet_connection_safely(self, source_host: str, target_host: str, 
                                       source_subnet: str, target_subnet: str) -> bool:
        """Create bidirectional connection with proper error handling"""
        try:
            print(f"\nCREATING BIDIRECTIONAL CONNECTION:")
            print(f"  {source_host} ({source_subnet}) <--> {target_host} ({target_subnet})")
            
            # Ensure both hosts have info sections
            if 'info' not in self.scenario_data['Hosts'][source_host]:
                self.scenario_data['Hosts'][source_host]['info'] = {}
            if 'info' not in self.scenario_data['Hosts'][target_host]:
                self.scenario_data['Hosts'][target_host]['info'] = {}
            
            # Create forward connection (source -> target)
            print(f"  Creating: {source_host} -> {target_host}")
            self.scenario_data['Hosts'][source_host]['info'][target_host] = {
                'Interfaces': 'IP Address'
            }
            
            # Create reverse connection (target -> source)  
            print(f"  Creating: {target_host} -> {source_host}")
            self.scenario_data['Hosts'][target_host]['info'][source_host] = {
                'Interfaces': 'IP Address'
            }
            
            # Verify connections were created
            source_info = self.scenario_data['Hosts'][source_host]['info']
            target_info = self.scenario_data['Hosts'][target_host]['info']
            
            forward_ok = (target_host in source_info and 
                         source_info[target_host].get('Interfaces') == 'IP Address')
            reverse_ok = (source_host in target_info and 
                         target_info[source_host].get('Interfaces') == 'IP Address')
            
            if forward_ok and reverse_ok:
                print(f"   Bidirectional connection successfully created!")
                return True
            else:
                print(f"   ERROR: Connection creation failed!")
                print(f"    Forward OK: {forward_ok}, Reverse OK: {reverse_ok}")
                return False
                
        except Exception as e:
            print(f"   ERROR: Exception during connection creation: {str(e)}")
            return False

    def connect_existing_subnets(self) -> bool:
        """Main function to connect two existing subnets reliably"""
        if not self.scenario_data:
            print("ERROR: No scenario loaded!")
            return False
        
        try:
            print("\n" + "="*60)
            print("CONNECTING EXISTING SUBNETS")
            print("="*60)
            print("This will create reliable bidirectional connections")
            print("while preserving network functionality.")
            
            # Step 1: Get subnet choices
            source_subnet, target_subnet = self.get_subnet_connection_choice()
            if not source_subnet or not target_subnet:
                return False
            
            # Step 2: Check existing connections and warn user
            if self.check_existing_connection(source_subnet, target_subnet):
                print(f"\n WARNING: {source_subnet} and {target_subnet} are already connected!")
                proceed = input("Create additional connection anyway? (y/n): ").strip().lower()
                if proceed != 'y':
                    print("Connection cancelled.")
                    return False
            
            # Step 3: Select optimal connection hosts
            source_host, target_host = self.select_best_connection_hosts(source_subnet, target_subnet)
            if not source_host or not target_host:
                return False
            
            # Step 4: Validate connection safety
            if not self.validate_connection_safety(source_host, target_host, source_subnet, target_subnet):
                print("Connection validation failed.")
                return False
            
            # Step 5: Create the connection
            print(f"\nPROCEEDING WITH CONNECTION:")
            print(f"  {source_host} ({source_subnet}) <--> {target_host} ({target_subnet})")
            
            if self.create_subnet_connection_safely(source_host, target_host, source_subnet, target_subnet):
                
                # Step 6: Verify and preserve patterns (less strict)
                print(f"\nVERIFYING CONNECTION INTEGRITY:")
                
                # Only preserve critical CAGE 2 patterns, don't block new connections
                print("  OK Preserving critical CAGE 2 patterns...")
                self.ensure_original_cage2_pattern()
                
                # Verify the new connection still exists
                source_info = self.scenario_data['Hosts'][source_host]['info']
                target_info = self.scenario_data['Hosts'][target_host]['info']
                
                connection_intact = (
                    target_host in source_info and source_info[target_host].get('Interfaces') == 'IP Address' and
                    source_host in target_info and target_info[source_host].get('Interfaces') == 'IP Address'
                )
                
                if connection_intact:
                    print("   New connection verified and preserved!")
                    self.update_action_space_limits()
                    print("\n" + "="*60)
                    print(" SUCCESS: SUBNET CONNECTION CREATED!")
                    print("="*60)
                    print(f" Connected: {source_subnet} <--> {target_subnet}")
                    print(f" Host link: {source_host} <--> {target_host}")
                    print(f" Bidirectional: Yes")
                    print(f" CAGE patterns: Preserved")
                    print("\nUse 'View network topology' to see the new connection.")
                    return True
                else:
                    print("   ERROR: New connection was overwritten during pattern preservation!")
                    return False
            else:
                print("Failed to create connection.")
                return False
                
        except Exception as e:
            print(f"ERROR: Exception in subnet connection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # HELPER METHODS
    def debug_connection_state(self, host1: str, host2: str) -> None:
        """Debug helper to show connection state between two hosts"""
        print(f"\nDEBUG: Connection state between {host1} and {host2}")
        print("-" * 50)
        
        hosts_data = self.scenario_data.get('Hosts', {})
        
        if host1 in hosts_data:
            host1_info = hosts_data[host1].get('info', {})
            if host2 in host1_info:
                interfaces = host1_info[host2].get('Interfaces', 'None')
                print(f"{host1} -> {host2}: {interfaces}")
            else:
                print(f"{host1} -> {host2}: No connection")
        else:
            print(f"{host1}: Host not found")
        
        if host2 in hosts_data:
            host2_info = hosts_data[host2].get('info', {})
            if host1 in host2_info:
                interfaces = host2_info[host1].get('Interfaces', 'None')
                print(f"{host2} -> {host1}: {interfaces}")
            else:
                print(f"{host2} -> {host1}: No connection")
        else:
            print(f"{host2}: Host not found")

    def show_subnet_connectivity_matrix(self) -> None:
        """Show a matrix of all subnet-to-subnet connections"""
        subnets = list(self.scenario_data.get('Subnets', {}).keys())
        
        print(f"\nSUBNET CONNECTIVITY MATRIX:")
        print("=" * 40)
        
        # Header
        print("From/To".ljust(12), end="")
        for subnet in subnets:
            print(subnet[:8].ljust(10), end="")
        print()
        
        # Matrix
        for source_subnet in subnets:
            print(source_subnet[:10].ljust(12), end="")
            
            for target_subnet in subnets:
                if source_subnet == target_subnet:
                    print("    -    ", end="")
                else:
                    connected = self.check_existing_connection(source_subnet, target_subnet)
                    print("   OK     " if connected else "    NOP    ", end="")
            print()
        
        print("\nOK = Connected, NOP = Not connected, - = Same subnet")

    def ensure_original_cage2_pattern(self) -> None:
        """Ensure original CAGE Challenge 2 connectivity patterns AND NACL rules are preserved"""
        print("   Ensuring original CAGE Challenge 2 patterns...")
        
        # Original CAGE 2 connectivity that MUST be preserved
        required_connections = {
            # User to Enterprise connections (CRITICAL)
            'User1': ['Enterprise1'],
            'User2': ['Enterprise1'],  
            'User3': ['Enterprise0'],
            'User4': ['Enterprise0'],  
            
            # Enterprise to Operational connections
            'Enterprise2': ['Op_Server0'],
        }
        
        hosts = self.scenario_data.get('Hosts', {})
        fixed_connections = 0
        
        for host_name, required_conns in required_connections.items():
            if host_name in hosts:
                host_info = hosts[host_name].get('info', {})
                
                for required_conn in required_conns:
                    if required_conn in hosts:  # Target host exists
                        # Check forward connection
                        if (required_conn not in host_info or 
                            host_info[required_conn].get('Interfaces') != 'IP Address'):
                            
                            print(f"   Restoring critical path: {host_name} -> {required_conn}")
                            hosts[host_name]['info'][required_conn] = {'Interfaces': 'IP Address'}
                            fixed_connections += 1
                        
                        # Check reverse connection (bidirectional)
                        target_info = hosts[required_conn].get('info', {})
                        if (host_name not in target_info or 
                            target_info[host_name].get('Interfaces') != 'IP Address'):
                            
                            print(f"   Restoring reverse path: {required_conn} -> {host_name}")
                            hosts[required_conn]['info'][host_name] = {'Interfaces': 'IP Address'}
                            fixed_connections += 1
        
        # NEW: Ensure critical NACL rules are preserved for CAGE 2 functionality
        print("   Preserving critical CAGE 2 NACL rules...")
        nacl_fixes = 0
        
        # Ensure User subnets can reach Enterprise subnets
        user_subnets = [name for name in self.current_nacl_rules.keys() if 'user' in name.lower()]
        enterprise_subnets = [name for name in self.current_nacl_rules.keys() if 'enterprise' in name.lower()]
        operational_subnets = [name for name in self.current_nacl_rules.keys() if 'operational' in name.lower()]
        
        for user_subnet in user_subnets:
            for enterprise_subnet in enterprise_subnets:
                # Ensure User can reach Enterprise
                if user_subnet in self.current_nacl_rules:
                    if enterprise_subnet not in self.current_nacl_rules[user_subnet]:
                        self.current_nacl_rules[user_subnet][enterprise_subnet] = {}
                    
                    # Ensure outbound traffic allowed (User -> Enterprise)
                    if self.current_nacl_rules[user_subnet][enterprise_subnet].get('out') == 'None':
                        print(f"   Restoring NACL: {user_subnet} -> {enterprise_subnet} (outbound)")
                        self.current_nacl_rules[user_subnet][enterprise_subnet]['out'] = 'all'
                        nacl_fixes += 1
        
        for enterprise_subnet in enterprise_subnets:
            for operational_subnet in operational_subnets:
                # Ensure Enterprise can reach Operational
                if enterprise_subnet in self.current_nacl_rules:
                    if operational_subnet not in self.current_nacl_rules[enterprise_subnet]:
                        self.current_nacl_rules[enterprise_subnet][operational_subnet] = {}
                    
                    # Ensure outbound traffic allowed (Enterprise -> Operational)
                    if self.current_nacl_rules[enterprise_subnet][operational_subnet].get('out') == 'None':
                        print(f"   Restoring NACL: {enterprise_subnet} -> {operational_subnet} (outbound)")
                        self.current_nacl_rules[enterprise_subnet][operational_subnet]['out'] = 'all'
                        nacl_fixes += 1
        
        # Apply NACL fixes to scenario data
        if nacl_fixes > 0:
            self._apply_nacl_rules_to_scenario()
            print(f"    Fixed {nacl_fixes} critical CAGE 2 NACL rules")
        
        if fixed_connections > 0:
            print(f"    Fixed {fixed_connections} critical CAGE 2 connections")
        else:
            print(f"    All critical CAGE 2 connections intact")

    # SUBNET BUILDING METHODS
    def get_safe_connection_point(self, target_subnet: str) -> Tuple[str, str]:
        """Get safe connection point that preserves original patterns"""
        subnets = list(self.scenario_data.get('Subnets', {}).keys())
        
        print("\nSELECT CONNECTION POINT")
        print("Choose where to connect your new subnet:")
        print("(Connection will preserve existing network patterns)")
        
        for i, subnet in enumerate(subnets, 1):
            # Show if subnet has critical paths
            critical_hosts = [cp['host'] for cp in self.critical_paths 
                            if cp['subnet'] == subnet]
            critical_info = f" [Critical paths: {len(critical_hosts)}]" if critical_hosts else ""
            print(f"{i}. Connect to {subnet} subnet{critical_info}")
        
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(subnets)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(subnets):
                    target_subnet = subnets[choice_idx]
                    break
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(subnets)}")
            except ValueError:
                print("ERROR: Please enter a valid number")
        
        # Default to bidirectional connection (safest for pattern preservation)
        connection_type = 'bidirectional'
        print(f"Connection type: bidirectional (preserves network topology)")
        
        return target_subnet, connection_type

    def get_subnet_configuration(self) -> Dict:
        """Get new subnet configuration from user"""
        config = {}
        
        # Subnet name
        while True:
            subnet_name = input("\nEnter new subnet name (e.g., 'DMZ', 'Remote', 'TestLab'): ").strip()
            if subnet_name and subnet_name not in self.scenario_data.get('Subnets', {}):
                config['name'] = subnet_name
                break
            elif subnet_name in self.scenario_data.get('Subnets', {}):
                print(f"ERROR: Subnet '{subnet_name}' already exists!")
            else:
                print("ERROR: Please enter a valid subnet name")
        
        # Number of hosts
        while True:
            try:
                host_count = int(input("Enter number of hosts (1-4): ").strip())
                if 1 <= host_count <= 4:
                    config['host_count'] = host_count
                    break
                else:
                    print("ERROR: Please enter a number between 1 and 4")
            except ValueError:
                print("ERROR: Please enter a valid number")
        
        # Use default naming automatically
        config['use_default_naming'] = True
        config['host_prefix'] = config['name']
        print(f"Host naming: {config['name']}0, {config['name']}1, etc. (default)")
        
        # Host types
        print("\nSELECT HOST TYPES")
        print("Available host types:")
        for i, (category, types) in enumerate(self.host_types.items(), 1):
            print(f"{i}. {category.upper()}: {', '.join(types)}")
        
        while True:
            try:
                type_choice = input(f"Enter choice (1-{len(self.host_types)}): ").strip()
                type_idx = int(type_choice) - 1
                categories = list(self.host_types.keys())
                
                if 0 <= type_idx < len(categories):
                    config['host_category'] = categories[type_idx]
                    config['host_types'] = self.host_types[categories[type_idx]]
                    break
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(self.host_types)}")
            except ValueError:
                print("ERROR: Please enter a valid number")
        
        # Security level
        print("\nSELECT SECURITY LEVEL")
        print("1. Low (None) - No penalty when compromised")
        print("2. Medium - Standard penalty when compromised") 
        print("3. High - Maximum penalty when compromised")
        
        while True:
            try:
                sec_choice = input("Enter choice (1-3): ").strip()
                if sec_choice in ['1', '2', '3']:
                    security_levels = {'1': 'None', '2': 'Medium', '3': 'High'}
                    config['security_level'] = security_levels[sec_choice]
                    break
                else:
                    print("ERROR: Please enter 1, 2, or 3")
            except ValueError:
                print("ERROR: Please enter a valid choice")
        
        return config

    def select_safe_connection_hosts(self, target_subnet: str, config: Dict) -> Tuple[List[str], str]:
        """Select connection hosts that preserve critical paths"""
        target_hosts = self.scenario_data['Subnets'][target_subnet]['Hosts']
        
        print(f"\nSELECT CONNECTION STRATEGY FOR {target_subnet.upper()}")
        print("1. Safe distributed (preserves critical paths, recommended)")
        print("2. Single gateway (minimal impact)")
        print("3. Manual selection (advanced)")
        
        while True:
            try:
                strategy = input("Enter choice (1-3): ").strip()
                if strategy in ['1', '2', '3']:
                    break
                else:
                    print("ERROR: Please enter 1, 2, or 3")
            except ValueError:
                print("ERROR: Please enter a valid choice")
        
        if strategy == '1':
            # Safe distributed: avoid critical path hosts
            safe_hosts = self._get_safe_hosts(target_subnet)
            if safe_hosts:
                print(f"Using safe distributed connection with {len(safe_hosts)} hosts")
                return safe_hosts, 'distributed'
            else:
                print("No safe hosts found, falling back to single gateway")
                strategy = '2'
        
        if strategy == '2':
            # Single gateway: choose least critical host
            gateway_host = self._get_least_critical_host(target_subnet)
            print(f"Using single gateway: {gateway_host}")
            return [gateway_host], 'gateway'
        
        else:  # Manual selection
            print(f"\nSELECT HOST FROM {target_subnet.upper()} SUBNET:")
            hosts_data = self.scenario_data.get('Hosts', {})
            
            for i, host in enumerate(target_hosts, 1):
                host_info = hosts_data.get(host, {})
                image = host_info.get('image', 'Unknown')
                os_type = self.get_host_os_type(image)
                role = self.get_host_role(image, host)
                
                # Mark critical hosts
                is_critical = any(cp['host'] == host for cp in self.critical_paths)
                critical_mark = " [CRITICAL]" if is_critical else ""
                
                print(f"{i}. {host} ({os_type}/{role}){critical_mark}")
            
            while True:
                try:
                    choice = input(f"Enter choice (1-{len(target_hosts)}): ").strip()
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(target_hosts):
                        selected_host = target_hosts[choice_idx]
                        return [selected_host], 'manual'
                    else:
                        print(f"ERROR: Please enter a number between 1 and {len(target_hosts)}")
                except ValueError:
                    print("ERROR: Please enter a valid number")

    def _get_safe_hosts(self, target_subnet: str) -> List[str]:
        """Get hosts that are safe to connect to (don't disrupt critical paths)"""
        target_hosts = self.scenario_data['Subnets'][target_subnet]['Hosts']
        critical_hosts = {cp['host'] for cp in self.critical_paths if cp['subnet'] == target_subnet}
        
        # Return non-critical hosts, or if all are critical, return ones with fewer connections
        safe_hosts = [host for host in target_hosts if host not in critical_hosts]
        
        if not safe_hosts:
            # If all hosts are critical, choose ones with fewer external connections
            host_connection_count = {}
            for host in target_hosts:
                external_count = len([cp for cp in self.critical_paths 
                                    if cp['host'] == host])
                host_connection_count[host] = external_count
            
            # Return host(s) with minimum connections
            min_connections = min(host_connection_count.values())
            safe_hosts = [host for host, count in host_connection_count.items() 
                         if count == min_connections]
        
        return safe_hosts

    def _get_least_critical_host(self, target_subnet: str) -> str:
        """Get the least critical host for gateway connection"""
        safe_hosts = self._get_safe_hosts(target_subnet)
        if safe_hosts:
            return safe_hosts[0]
        else:
            # Return first host as fallback
            return self.scenario_data['Subnets'][target_subnet]['Hosts'][0]

    def create_hosts_with_pattern_preservation(self, config: Dict, target_subnet: str, connection_hosts: List[str], connection_mode: str) -> Dict:
        """Create host definitions while preserving original connection patterns"""
        hosts = {}
        subnet_name = config['name']
        host_types = config['host_types']
        host_prefix = config['host_prefix']
        
        for i in range(config['host_count']):
            host_name = f"{subnet_name}{i}"
            
            # Cycle through available host types
            host_image = host_types[i % len(host_types)]
            
            # Determine username based on host type
            if 'windows' in host_image.lower():
                username = 'SYSTEM'
            else:
                username = 'ubuntu'
            
            # Create connectivity info (self-connection always included)
            info = {host_name: {'Interfaces': 'All'}}
            
            # Set up external connections based on mode and pattern preservation
            if connection_mode == 'distributed' and len(connection_hosts) > 1:
                # Distribute connections across available safe hosts
                target_host = connection_hosts[i % len(connection_hosts)]
                info[target_host] = {'Interfaces': 'IP Address'}
                
            elif connection_mode in ['gateway', 'manual'] or len(connection_hosts) == 1:
                # Single gateway connection (safer for preserving patterns)
                info[connection_hosts[0]] = {'Interfaces': 'IP Address'}
                    
            else:  # Fallback distributed
                target_host = connection_hosts[i % len(connection_hosts)]
                info[target_host] = {'Interfaces': 'IP Address'}
            
            hosts[host_name] = {
                'AWS_Info': [],
                'image': host_image,
                'info': info,
                'ConfidentialityValue': config['security_level'],
                'AvailabilityValue': config['security_level'],
                'username': username
            }
        
        return hosts

    def update_connections_safely(self, new_hosts: Dict, target_subnet: str, connection_hosts: List[str], connection_type: str, connection_mode: str) -> None:
        """Update target subnet connectivity while preserving original patterns"""
        if connection_type in ['bidirectional', 'inbound_only']:
            new_host_names = list(new_hosts.keys())
            
            for i, target_host in enumerate(connection_hosts):
                if (target_host in self.scenario_data['Hosts'] and 
                    'info' in self.scenario_data['Hosts'][target_host]):
                    
                    # Determine which new host(s) to connect
                    if connection_mode == 'distributed' and len(connection_hosts) > 1:
                        # Connect each target host to corresponding new host
                        new_host = new_host_names[i % len(new_host_names)]
                        self.scenario_data['Hosts'][target_host]['info'][new_host] = {
                            'Interfaces': 'IP Address'
                        }
                    elif connection_mode in ['gateway', 'manual']:
                        # FIXED: Connect gateway host to ALL new hosts (not just first)
                        if i == 0:  # Only first (gateway) host in target subnet
                            for new_host in new_host_names:  # Connect to ALL new hosts
                                self.scenario_data['Hosts'][target_host]['info'][new_host] = {
                                    'Interfaces': 'IP Address'
                                }
                    else:  # Fallback
                        new_host = new_host_names[i % len(new_host_names)]
                        self.scenario_data['Hosts'][target_host]['info'][new_host] = {
                            'Interfaces': 'IP Address'
                        }

    def verify_pattern_preservation(self) -> bool:
        """Verify that original connection patterns are preserved INCLUDING bidirectional pairs"""
        current_connections = {}
        hosts = self.scenario_data.get('Hosts', {})
        
        # Build current connection map
        for host_name, host_data in hosts.items():
            if host_name in self.original_connections:  # Only check original hosts
                connections = []
                host_info = host_data.get('info', {})
                
                for connected_host, details in host_info.items():
                    if connected_host != host_name and details.get('Interfaces') == 'IP Address':
                        # Only count connections to original hosts
                        if connected_host in self.original_connections:
                            connections.append(connected_host)
                
                current_connections[host_name] = connections
        
        preserved = True
        
        # Check if original unidirectional connections are preserved
        for host_name, original_conns in self.original_connections.items():
            current_conns = current_connections.get(host_name, [])
            
            for orig_conn in original_conns:
                if orig_conn not in current_conns:
                    print(f"WARNING: Lost original connection {host_name} -> {orig_conn}")
                    preserved = False
        
        # NEW: Check if bidirectional pairs are preserved
        for pair in self.original_bidirectional_pairs:
            host_a, host_b = pair
            
            # Check forward connection (A -> B)
            current_conns_a = current_connections.get(host_a, [])
            if host_b not in current_conns_a:
                print(f"CRITICAL: Lost bidirectional connection {host_a} -> {host_b}")
                preserved = False
            
            # Check reverse connection (B -> A)
            current_conns_b = current_connections.get(host_b, [])
            if host_a not in current_conns_b:
                print(f"CRITICAL: Lost bidirectional connection {host_b} -> {host_a}")
                preserved = False
        
        return preserved

    def update_agent_configurations(self, subnet_name: str, new_hosts: Dict) -> None:
        """Update all agent configurations to include new subnet"""
        
        # Update allowed subnets for all agents
        for agent_name in ['Blue', 'Green', 'Red']:
            if 'AllowedSubnets' in self.scenario_data['Agents'][agent_name]:
                if subnet_name not in self.scenario_data['Agents'][agent_name]['AllowedSubnets']:
                    self.scenario_data['Agents'][agent_name]['AllowedSubnets'].append(subnet_name)
        
        # Add hosts to Blue and Green INT sections
        host_access_config = {
            'Interfaces': 'All',
            'System info': 'All',
            'User info': 'All'
        }
        
        for host_name in new_hosts.keys():
            self.scenario_data['Agents']['Blue']['INT']['Hosts'][host_name] = host_access_config.copy()
            self.scenario_data['Agents']['Green']['INT']['Hosts'][host_name] = host_access_config.copy()
        
        # Add starting sessions for Blue agent (Velociraptor clients)
        for host_name, host_data in new_hosts.items():
            blue_session = {
                'hostname': host_name,
                'name': f'Velo{host_name}',
                'parent': 'VeloServer',
                'type': 'VelociraptorClient',
                'username': host_data['username']
            }
            self.scenario_data['Agents']['Blue']['starting_sessions'].append(blue_session)
        
        # Add starting sessions for Green agent
        for host_name in new_hosts.keys():
            green_session = {
                'hostname': host_name,
                'name': 'GreenSession',
                'type': 'green_session',
                'username': 'GreenAgent'
            }
            self.scenario_data['Agents']['Green']['starting_sessions'].append(green_session)


    def get_existing_subnet_choice(self) -> str:
        """Get user choice for which existing subnet to add hosts to"""
        subnets = list(self.scenario_data.get('Subnets', {}).keys())
        
        print("\nSELECT EXISTING SUBNET")
        print("Choose which subnet to add hosts to:")
        print("(New hosts will inherit subnet patterns and characteristics)")
        
        for i, subnet in enumerate(subnets, 1):
            hosts = self.scenario_data['Subnets'][subnet]['Hosts']
            host_count = len(hosts)
            
            # Check if subnet has critical paths
            critical_hosts = [cp['host'] for cp in self.critical_paths 
                            if cp['subnet'] == subnet]
            critical_info = f" [Critical paths: {len(critical_hosts)}]" if critical_hosts else ""
            
            print(f"{i}. {subnet} subnet ({host_count} hosts){critical_info}")
        
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(subnets)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(subnets):
                    return subnets[choice_idx]
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(subnets)}")
            except ValueError:
                print("ERROR: Please enter a valid number")

    def get_host_addition_config(self, subnet_name: str) -> Dict:
        """Get configuration for adding hosts to existing subnet"""
        config = {}
        
        print(f"\nCONFIGURING NEW HOSTS FOR {subnet_name.upper()}")
        
        # Number of hosts to add
        while True:
            try:
                host_count = int(input("Enter number of hosts to add (1-4): ").strip())
                if 1 <= host_count <= 4:
                    config['host_count'] = host_count
                    break
                else:
                    print("ERROR: Please enter a number between 1 and 4")
            except ValueError:
                print("ERROR: Please enter a valid number")
        
        # Host types selection
        print("\nSELECT HOST TYPES")
        print("Available host types:")
        for i, (category, types) in enumerate(self.host_types.items(), 1):
            print(f"{i}. {category.upper()}: {', '.join(types)}")
        
        while True:
            try:
                type_choice = input(f"Enter choice (1-{len(self.host_types)}): ").strip()
                type_idx = int(type_choice) - 1
                categories = list(self.host_types.keys())
                
                if 0 <= type_idx < len(categories):
                    config['host_category'] = categories[type_idx]
                    config['host_types'] = self.host_types[categories[type_idx]]
                    break
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(self.host_types)}")
            except ValueError:
                print("ERROR: Please enter a valid number")
        
        return config

    def detect_naming_pattern(self, subnet_name: str) -> Tuple[str, int]:
        """Detect naming pattern from existing hosts in subnet"""
        existing_hosts = self.scenario_data['Subnets'][subnet_name]['Hosts']
        
        if not existing_hosts:
            return subnet_name, 0
        
        print(f"   Analyzing naming pattern from {len(existing_hosts)} existing hosts...")
        
        # Try to detect pattern from existing hosts
        base_name = ""
        highest_num = -1
        
        for host in existing_hosts:
            # Find where numbers start from the end
            i = len(host) - 1
            while i >= 0 and host[i].isdigit():
                i -= 1
            
            if i < len(host) - 1:  # Found digits
                current_base = host[:i+1]
                current_num = int(host[i+1:])
                
                if not base_name:
                    base_name = current_base
                
                if current_base == base_name:
                    highest_num = max(highest_num, current_num)
        
        # If no pattern found, use subnet name
        if not base_name:
            base_name = subnet_name
            highest_num = -1
        
        next_num = highest_num + 1
        print(f"   Detected pattern: {base_name}X (next: {base_name}{next_num})")
        
        return base_name, next_num

    def inherit_subnet_characteristics(self, subnet_name: str) -> Tuple[str, str]:
        """Inherit security level and other characteristics from existing hosts in subnet"""
        existing_hosts = self.scenario_data['Subnets'][subnet_name]['Hosts']
        
        if not existing_hosts:
            return 'None', 'None'
        
        print(f"   Inheriting characteristics from existing hosts...")
        
        # Get security level from first host that has it defined
        for host_name in existing_hosts:
            host_data = self.scenario_data['Hosts'].get(host_name, {})
            conf_val = host_data.get('ConfidentialityValue', 'None')
            avail_val = host_data.get('AvailabilityValue', 'None')
            
            if conf_val != 'None' or avail_val != 'None':
                print(f"   Inherited security: Confidentiality={conf_val}, Availability={avail_val}")
                return conf_val, avail_val
        
        print(f"   No security settings found, using defaults")
        return 'None', 'None'

    def analyze_subnet_connectivity_pattern(self, subnet_name: str) -> Dict:
        """Analyze connectivity pattern from existing hosts in subnet"""
        existing_hosts = self.scenario_data['Subnets'][subnet_name]['Hosts']
        connectivity_pattern = {}
        
        if not existing_hosts:
            return connectivity_pattern
        
        print(f"   Analyzing connectivity patterns...")
        
        # Analyze connectivity from all existing hosts to find patterns
        external_subnet_connections = {}
        
        for host_name in existing_hosts:
            host_info = self.scenario_data['Hosts'].get(host_name, {}).get('info', {})
            
            for connected_host, details in host_info.items():
                if connected_host != host_name and details.get('Interfaces') == 'IP Address':
                    # Find which subnet this connected host belongs to
                    connected_subnet = self.find_host_subnet(connected_host)
                    if connected_subnet != subnet_name and connected_subnet != 'Unknown':
                        if connected_subnet not in external_subnet_connections:
                            external_subnet_connections[connected_subnet] = []
                        
                        external_subnet_connections[connected_subnet].append({
                            'source_host': host_name,
                            'target_host': connected_host
                        })
        
        # Determine the pattern (distributed vs single gateway)
        for ext_subnet, connections in external_subnet_connections.items():
            unique_source_hosts = set(conn['source_host'] for conn in connections)
            unique_target_hosts = set(conn['target_host'] for conn in connections)
            
            connectivity_pattern[ext_subnet] = {
                'connections': connections,
                'source_hosts': list(unique_source_hosts),
                'target_hosts': list(unique_target_hosts),
                'pattern_type': 'distributed' if len(unique_source_hosts) > 1 else 'gateway'
            }
            
            print(f"   Found {len(connections)} connections to {ext_subnet} ({connectivity_pattern[ext_subnet]['pattern_type']})")
        
        return connectivity_pattern

    def create_additional_hosts_with_pattern_preservation(self, config: Dict, subnet_name: str, 
                                                        base_name: str, start_num: int) -> Dict:
        """Create additional host definitions for existing subnet with pattern preservation"""
        hosts = {}
        host_types = config['host_types']
        
        # Get inherited characteristics
        conf_val, avail_val = self.inherit_subnet_characteristics(subnet_name)
        
        print(f"   Creating {config['host_count']} new hosts...")
        
        for i in range(config['host_count']):
            host_name = f"{base_name}{start_num + i}"
            
            # Cycle through available host types
            host_image = host_types[i % len(host_types)]
            
            # Determine username based on host type
            if 'windows' in host_image.lower():
                username = 'SYSTEM'
            else:
                username = 'ubuntu'
            
            # Start with self-connection
            info = {host_name: {'Interfaces': 'All'}}
            
            hosts[host_name] = {
                'AWS_Info': [],
                'image': host_image,
                'info': info,
                'ConfidentialityValue': conf_val,
                'AvailabilityValue': avail_val,
                'username': username
            }
            
            print(f"     Created: {host_name} ({host_image})")
        
        return hosts

    def update_external_connectivity_with_pattern_preservation(self, new_hosts: Dict, 
                                                            subnet_name: str, 
                                                            connectivity_pattern: Dict) -> None:
        """Update external connectivity for new hosts following existing patterns"""
        if not connectivity_pattern:
            print("   No external connectivity patterns found - new hosts will be isolated within subnet")
            return
        
        new_host_names = list(new_hosts.keys())
        print(f"   Updating external connectivity for {len(new_host_names)} new hosts...")
        
        for ext_subnet, pattern_info in connectivity_pattern.items():
            pattern_type = pattern_info['pattern_type']
            existing_connections = pattern_info['connections']
            source_hosts = pattern_info['source_hosts']
            target_hosts = pattern_info['target_hosts']
            
            print(f"   Applying {pattern_type} pattern for {ext_subnet} subnet...")
            
            if pattern_type == 'distributed':
                # Distribute new hosts across existing target hosts
                for i, new_host in enumerate(new_host_names):
                    target_host = target_hosts[i % len(target_hosts)]
                    
                    # Add forward connection (new host -> target)
                    new_hosts[new_host]['info'][target_host] = {'Interfaces': 'IP Address'}
                    
                    # Add reverse connection (target -> new host)
                    if target_host in self.scenario_data['Hosts']:
                        if 'info' not in self.scenario_data['Hosts'][target_host]:
                            self.scenario_data['Hosts'][target_host]['info'] = {}
                        
                        self.scenario_data['Hosts'][target_host]['info'][new_host] = {
                            'Interfaces': 'IP Address'
                        }
                    
                    print(f"     Connected: {new_host} <--> {target_host}")
            
            elif pattern_type == 'gateway':
                # Use the same gateway pattern - connect all new hosts to the gateway host(s)
                gateway_host = source_hosts[0]  # Primary gateway host
                primary_target = target_hosts[0]  # Primary target in external subnet
                
                for new_host in new_host_names:
                    # New host connects to the primary target (following gateway pattern)
                    new_hosts[new_host]['info'][primary_target] = {'Interfaces': 'IP Address'}
                    
                    # Primary target connects back to new host
                    if primary_target in self.scenario_data['Hosts']:
                        if 'info' not in self.scenario_data['Hosts'][primary_target]:
                            self.scenario_data['Hosts'][primary_target]['info'] = {}
                        
                        self.scenario_data['Hosts'][primary_target]['info'][new_host] = {
                            'Interfaces': 'IP Address'
                        }
                    
                    print(f"     Connected: {new_host} <--> {primary_target} (via gateway pattern)")

    def update_agent_configurations_for_additional_hosts(self, new_hosts: Dict) -> None:
        """Update agent configurations for additional hosts"""
        print(f"   Updating agent configurations for {len(new_hosts)} new hosts...")
        
        # Add hosts to Blue and Green INT sections
        host_access_config = {
            'Interfaces': 'All',
            'System info': 'All',
            'User info': 'All'
        }
        
        for host_name in new_hosts.keys():
            self.scenario_data['Agents']['Blue']['INT']['Hosts'][host_name] = host_access_config.copy()
            self.scenario_data['Agents']['Green']['INT']['Hosts'][host_name] = host_access_config.copy()
        
        # Add starting sessions for Blue agent (Velociraptor clients)
        for host_name, host_data in new_hosts.items():
            blue_session = {
                'hostname': host_name,
                'name': f'Velo{host_name}',
                'parent': 'VeloServer',
                'type': 'VelociraptorClient',
                'username': host_data['username']
            }
            self.scenario_data['Agents']['Blue']['starting_sessions'].append(blue_session)
        
        # Add starting sessions for Green agent
        for host_name in new_hosts.keys():
            green_session = {
                'hostname': host_name,
                'name': 'GreenSession',
                'type': 'green_session',
                'username': 'GreenAgent'
            }
            self.scenario_data['Agents']['Green']['starting_sessions'].append(green_session)
        
        print(f"     Added {len(new_hosts)} Blue sessions and {len(new_hosts)} Green sessions")

    def add_hosts_to_existing_subnet(self) -> bool:
        """Main function to add hosts to existing subnet with pattern preservation"""
        if not self.scenario_data:
            print("ERROR: No scenario loaded!")
            return False
        
        try:
            print("\n" + "="*60)
            print("ADDING HOSTS TO EXISTING SUBNET")
            print("="*60)
            print("This will add hosts while preserving existing connectivity patterns")
            
            # Step 1: Get subnet choice
            subnet_name = self.get_existing_subnet_choice()
            
            # Step 2: Get configuration for new hosts
            config = self.get_host_addition_config(subnet_name)
            
            # Step 3: Analyze existing patterns
            print(f"\nANALYZING EXISTING PATTERNS IN {subnet_name.upper()}:")
            base_name, start_num = self.detect_naming_pattern(subnet_name)
            connectivity_pattern = self.analyze_subnet_connectivity_pattern(subnet_name)
            
            print(f"\nADDING HOSTS TO: {subnet_name}")
            print(f"   New hosts: {config['host_count']}")
            print(f"   Type: {config['host_category']}")
            print(f"   Naming pattern: {base_name}{start_num}, {base_name}{start_num+1}, ...")
            print(f"   External connections: {len(connectivity_pattern)} subnets")
            
            # Step 4: Create new hosts with inherited characteristics
            print(f"\nCREATING NEW HOSTS:")
            new_hosts = self.create_additional_hosts_with_pattern_preservation(
                config, subnet_name, base_name, start_num)
            
            # Step 5: Update external connectivity following existing patterns
            print(f"\nUPDATING CONNECTIVITY:")
            self.update_external_connectivity_with_pattern_preservation(
                new_hosts, subnet_name, connectivity_pattern)
            
            # Step 6: Add hosts to scenario data structures
            print(f"\nINTEGRATING INTO SCENARIO:")
            print("   Adding hosts to scenario...")
            self.scenario_data['Hosts'].update(new_hosts)
            
            # Step 7: Update subnet definition
            print("   Updating subnet definition...")
            self.scenario_data['Subnets'][subnet_name]['Hosts'].extend(new_hosts.keys())
            original_size = self.scenario_data['Subnets'][subnet_name]['Size']
            self.scenario_data['Subnets'][subnet_name]['Size'] = original_size + config['host_count']
            
            # Step 8: Update agent configurations
            print("   Updating agent configurations...")
            self.update_agent_configurations_for_additional_hosts(new_hosts)
            
            # Step 9: Ensure pattern preservation (critical for CAGE 2)
            print(f"\nENSURING PATTERN PRESERVATION:")
            print("   Enforcing CAGE 2 pattern preservation...")
            self.ensure_original_cage2_pattern()
            self.enforce_bidirectional_preservation()
            
            # Step 10: Verify everything is still intact
            print("   Verifying pattern preservation...")
            if self.verify_pattern_preservation():
                print("     Original connection patterns preserved")
            else:
                print("     WARNING: Some original patterns may be affected")
            
            # Final verification
            new_total_hosts = len(self.scenario_data['Subnets'][subnet_name]['Hosts'])
            print(f"\nVERIFICATION:")
            print(f"   {subnet_name} subnet now has {new_total_hosts} hosts")
            print(f"   Added {config['host_count']} new hosts successfully")
            self.update_action_space_limits()
            print("\n" + "="*60)
            print("SUCCESS: HOSTS ADDED TO EXISTING SUBNET!")
            print("="*60)
            print(f"Subnet: {subnet_name}")
            print(f"New hosts: {', '.join(new_hosts.keys())}")
            print(f"Pattern preservation: Maintained")
            print(f"Agent integration: Complete")
            print("\nUse 'View network topology' to see the updated subnet.")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error adding hosts to subnet: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def enforce_bidirectional_preservation(self) -> None:
        """ENFORCE bidirectional connectivity preservation by adding missing connections"""
        print("   Enforcing bidirectional connectivity preservation...")
        
        hosts = self.scenario_data.get('Hosts', {})
        restored_connections = 0
        
        for pair in self.original_bidirectional_pairs:
            host_a, host_b = pair
            
            # Skip if either host doesn't exist (shouldn't happen with original hosts)
            if host_a not in hosts or host_b not in hosts:
                continue
            
            # Ensure A -> B connection exists
            host_a_info = hosts[host_a].get('info', {})
            if host_b not in host_a_info or host_a_info[host_b].get('Interfaces') != 'IP Address':
                print(f"   Restoring: {host_a} -> {host_b}")
                hosts[host_a]['info'][host_b] = {'Interfaces': 'IP Address'}
                restored_connections += 1
            
            # Ensure B -> A connection exists
            host_b_info = hosts[host_b].get('info', {})
            if host_a not in host_b_info or host_b_info[host_a].get('Interfaces') != 'IP Address':
                print(f"   Restoring: {host_b} -> {host_a}")
                hosts[host_b]['info'][host_a] = {'Interfaces': 'IP Address'}
                restored_connections += 1
        
        if restored_connections > 0:
            print(f"    Restored {restored_connections} missing bidirectional connections")
        else:
            print(f"    All bidirectional connections intact")

    def create_subnet_definition(self, config: Dict, new_hosts: Dict) -> Dict:
        """Create subnet definition with appropriate NACLs"""
        subnet_name = config['name']
        
        # Create basic subnet definition
        subnet_def = {
            'Hosts': list(new_hosts.keys()),
            'NACLs': {
                'all': {
                    'in': 'all',
                    'out': 'all'
                }
            },
            'Size': config['host_count']
        }
        
        # Add security-based NACLs for high-security subnets
        if config['security_level'] == 'High':
            subnet_def['NACLs']['User'] = {
                'in': 'None',
                'out': 'all'
            }
        
        return subnet_def

    def build_subnet(self) -> bool:
        """Main function to build and integrate new subnet with pattern preservation"""
        if not self.scenario_data:
            print("ERROR: No scenario loaded!")
            return False
        
        try:
            print("\nSTARTING ENHANCED SUBNET BUILDER")
            print("="*50)
            print("This builder preserves original network topology patterns")
            
            # Get user choices with safety considerations
            target_subnet, connection_type = self.get_safe_connection_point('')
            config = self.get_subnet_configuration()
            connection_hosts, connection_mode = self.select_safe_connection_hosts(target_subnet, config)
            
            print(f"\nBUILDING SUBNET: {config['name']}")
            print(f"   Connecting to: {target_subnet}")
            print(f"   Connection type: {connection_type}")
            print(f"   Connection mode: {connection_mode}")
            print(f"   Connection hosts: {connection_hosts}")
            print(f"   Hosts: {config['host_count']}")
            print(f"   Type: {config['host_category']}")
            print(f"   Security: {config['security_level']}")
            print(f"   Naming: {config['host_prefix']}0, {config['host_prefix']}1, ...")
            
            # Create hosts with pattern preservation
            print("   Creating host definitions...")
            new_hosts = self.create_hosts_with_pattern_preservation(config, target_subnet, connection_hosts, connection_mode)
            
            # Update target subnet connectivity safely
            print("   Setting up safe connectivity...")
            self.update_connections_safely(new_hosts, target_subnet, connection_hosts, connection_type, connection_mode)
            
            # Add hosts to scenario
            print("   Adding hosts to scenario...")
            self.scenario_data['Hosts'].update(new_hosts)
            
            # Create subnet definition
            print("   Creating subnet definition...")
            subnet_def = self.create_subnet_definition(config, new_hosts)
            self.scenario_data['Subnets'][config['name']] = subnet_def
            
            # Update agent configurations
            print("   Updating agent configurations...")
            self.update_agent_configurations(config['name'], new_hosts)
            
            # Verify pattern preservation
            print("   Enforcing CAGE 2 pattern preservation...")
            self.ensure_original_cage2_pattern()  
            self.enforce_bidirectional_preservation()
            print("   Verifying pattern preservation...")
            if self.verify_pattern_preservation():
                print("     Original connection patterns preserved")
            else:
                print("     WARNING: Some original patterns may be affected")
            
            self.update_action_space_limits()
            print("SUCCESS: Subnet built successfully with pattern preservation!")
            return True
            
        except Exception as e:
            print(f"ERROR: Error building subnet: {str(e)}")
            return False

    # SAVE AND SUMMARY METHODS
    def save_scenario(self, output_path: Optional[str] = None) -> bool:
        """Save the modified scenario to file"""
        if not self.scenario_data:
            print("ERROR: No scenario data to save!")
            return False
        
        try:
            if not output_path:
                base_name = self.scenario_path.stem
                output_path = f"{base_name}_enhanced.yaml"
            
            # Ensure .yaml extension
            if not output_path.endswith('.yaml') and not output_path.endswith('.yml'):
                output_path += '.yaml'
            
            # Save scenario
            with open(output_path, 'w') as file:
                yaml.dump(self.scenario_data, file, default_flow_style=False, sort_keys=False, width=120)
            
            # Add header comment
            #self.add_header_comment(output_path)
            
            print(f"SUCCESS: Enhanced scenario saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Error saving scenario: {str(e)}")
            return False
    def apply_nacl_security_template(self) -> bool:
        """Apply predefined NACL security template"""
        templates = {
            '1': ('CAGE 2 Original', 'Restore original CAGE 2 NACL configuration'),
            '2': ('Open Network', 'All subnets can communicate freely'),
            '3': ('High Security', 'Minimal inter-subnet communication'),
            '4': ('Zero Trust', 'Block all by default, explicit allow only')
        }
        
        print("\nAVAILABLE SECURITY TEMPLATES:")
        print("="*50)
        
        for key, (name, desc) in templates.items():
            print(f"{key}. {name}")
            print(f"   {desc}")
            print()
        
        while True:
            try:
                choice = input(f"Select template (1-{len(templates)}): ").strip()
                if choice in templates:
                    template_name = templates[choice][0]
                    return self.apply_nacl_template(template_name)
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(templates)}")
            except ValueError:
                print("ERROR: Please enter a valid number")

    def apply_nacl_template(self, template_name: str) -> bool:
        """Apply predefined NACL template"""
        print(f"\nApplying template: {template_name}")
        
        # Backup current state
        backup_id = self.backup_current_nacl_state()
        
        try:
            if template_name == 'CAGE 2 Original':
                # Restore original NACL rules
                self.current_nacl_rules = copy.deepcopy(self.original_nacl_rules)
                
            elif template_name == 'Open Network':
                # All subnets allow all traffic
                for subnet_name in self.scenario_data.get('Subnets', {}):
                    self.current_nacl_rules[subnet_name] = {
                        'all': {'in': 'all', 'out': 'all'}
                    }
                    
            elif template_name == 'High Security':
                # Minimal communication, preserve critical paths
                for subnet_name in self.scenario_data.get('Subnets', {}):
                    if subnet_name.lower() == 'operational':
                        # Operational subnet: block user access, allow enterprise
                        self.current_nacl_rules[subnet_name] = {
                            'User': {'in': 'None', 'out': 'all'},
                            'all': {'in': 'all', 'out': 'all'}
                        }
                    else:
                        # Other subnets: allow minimal communication
                        self.current_nacl_rules[subnet_name] = {
                            'all': {'in': 'all', 'out': 'all'}
                        }
                        
            elif template_name == 'Zero Trust':
                # Block all by default
                for subnet_name in self.scenario_data.get('Subnets', {}):
                    self.current_nacl_rules[subnet_name] = {
                        'all': {'in': 'None', 'out': 'None'}
                    }
            
            # Apply changes to scenario data
            self._apply_nacl_rules_to_scenario()
            
            # Preserve critical CAGE 2 patterns
            self.ensure_original_cage2_pattern()
            
            print(f" Template '{template_name}' applied successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR applying template: {str(e)}")
            # Rollback on error
            self.rollback_nacl_changes(backup_id)
            return False

    def reset_to_original_nacls(self) -> bool:
        """Reset NACL rules to original CAGE 2 defaults"""
        try:
            self.current_nacl_rules = copy.deepcopy(self.original_nacl_rules)
            self._apply_nacl_rules_to_scenario()
            print(" NACL rules reset to original configuration")
            return True
        except Exception as e:
            print(f"ERROR resetting NACL rules: {str(e)}")
            return False

    def view_detailed_subnet_nacls(self) -> None:
        """View detailed NACL rules for selected subnet"""
        subnet_name = self.select_subnet_for_nacl_modification()
        if subnet_name:
            self.display_subnet_nacl_details(subnet_name)
        
    def select_nacl_target_and_direction(self, source_subnet: str) -> Tuple[str, str]:
        """Choose target subnet and traffic direction - FIXED VERSION"""
        
        print(f"\nCONFIGURE NACL RULES FOR: {source_subnet.upper()}")
        print("Select what type of rule to modify:")
        print()
        print("1. Default rule (all) - Controls general subnet behavior")
        print("2. Specific subnet rule - Controls traffic to/from specific subnet")
        
        while True:
            try:
                rule_type = input("Enter choice (1-2): ").strip()
                if rule_type in ['1', '2']:
                    break
                else:
                    print("ERROR: Please enter 1 or 2")
            except ValueError:
                print("ERROR: Please enter a valid choice")
        
        if rule_type == '1':
            # User wants to modify the default 'all' rule
            target_subnet = 'all'
            print(f"\nModifying DEFAULT rule for {source_subnet}")
            print("This affects traffic to/from all other subnets (unless specific rules exist)")
            
        else:
            # User wants specific subnet rule
            available_targets = [s for s in self.get_available_nacl_targets(source_subnet) if s != 'all']
            
            print(f"\nSelect specific subnet to create rule for:")
            for i, target in enumerate(available_targets, 1):
                print(f"{i}. {target}")
            
            while True:
                try:
                    choice = input(f"\nEnter choice (1-{len(available_targets)}): ").strip()
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(available_targets):
                        target_subnet = available_targets[choice_idx]
                        break
                    else:
                        print(f"ERROR: Please enter a number between 1 and {len(available_targets)}")
                except ValueError:
                    print("ERROR: Please enter a valid number")
        
        # Step 2: Select direction (same as before)
        print(f"\nSELECT TRAFFIC DIRECTION:")
        print("Which traffic direction to configure?")
        print()
        print("1. Inbound (traffic coming TO this subnet)")
        print("2. Outbound (traffic leaving FROM this subnet)")
        
        while True:
            try:
                direction_choice = input("Enter choice (1-2): ").strip()
                if direction_choice == '1':
                    return target_subnet, 'inbound'
                elif direction_choice == '2':
                    return target_subnet, 'outbound'
                else:
                    print("ERROR: Please enter 1 or 2")
            except ValueError:
                print("ERROR: Please enter a valid choice")

    def apply_nacl_rule_change(self, source_subnet: str, target_subnet: str, direction: str, new_rule: str) -> bool:
        """Execute single NACL rule modification - FIXED VERSION"""
        try:
            # Validate safety
            is_safe, warnings = self.validate_nacl_change_safety(source_subnet, target_subnet, direction, new_rule)
            
            # Show warnings but allow override
            if warnings:
                print("\n" + "!"*50)
                print("SAFETY WARNINGS:")
                for warning in warnings:
                    print(f"  {warning}")
                print("!"*50)
                
                proceed = input("\nProceed anyway? (y/N): ").strip().lower()
                if proceed != 'y':
                    print("Change cancelled.")
                    return False
            
            # Ensure subnet exists in current rules
            if source_subnet not in self.current_nacl_rules:
                self.current_nacl_rules[source_subnet] = {}
            
            # FIXED: Handle 'all' rule properly
            if target_subnet == 'all':
                # Modifying the default 'all' rule
                if 'all' not in self.current_nacl_rules[source_subnet]:
                    # Create default 'all' rule if it doesn't exist
                    self.current_nacl_rules[source_subnet]['all'] = {'in': 'all', 'out': 'all'}
                
                # Update the specific direction
                if direction == 'inbound':
                    self.current_nacl_rules[source_subnet]['all']['in'] = new_rule
                    print(f"Modified DEFAULT inbound rule for {source_subnet}: {new_rule}")
                elif direction == 'outbound':
                    self.current_nacl_rules[source_subnet]['all']['out'] = new_rule
                    print(f"Modified DEFAULT outbound rule for {source_subnet}: {new_rule}")
            
            else:
                # Creating/modifying specific subnet rule
                if target_subnet not in self.current_nacl_rules[source_subnet]:
                    self.current_nacl_rules[source_subnet][target_subnet] = {}
                
                # Apply the rule change
                if direction == 'inbound':
                    self.current_nacl_rules[source_subnet][target_subnet]['in'] = new_rule
                    print(f"Set specific rule: {source_subnet} <- {target_subnet} (inbound: {new_rule})")
                elif direction == 'outbound':
                    self.current_nacl_rules[source_subnet][target_subnet]['out'] = new_rule
                    print(f"Set specific rule: {source_subnet} -> {target_subnet} (outbound: {new_rule})")
            
            # Apply to scenario data
            self._apply_nacl_rules_to_scenario()
            
            # Log the change
            self.nacl_change_history.append({
                'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
                'source': source_subnet,
                'target': target_subnet,
                'direction': direction,
                'new_rule': new_rule
            })
            
            return True
            
        except Exception as e:
            print(f"ERROR applying NACL change: {str(e)}")
            return False

    def configure_specific_nacl_rule(self, source_subnet: str, target_subnet: str, direction: str) -> bool:
        """Set specific rule (all/None) for subnet pair and direction - FIXED VERSION"""
        current_rules = self.current_nacl_rules.get(source_subnet, {}).get(target_subnet, {})
        
        print(f"\nCONFIGURE RULE: {source_subnet}")
        if target_subnet == 'all':
            print(f"Type: DEFAULT rule (affects all other subnets)")
        else:
            print(f"Target: {target_subnet} (specific subnet rule)")
        print(f"Direction: {direction}")
        print()
        
        # Show current rules
        if target_subnet == 'all':
            # Show current default rule
            default_rules = self.current_nacl_rules.get(source_subnet, {}).get('all', {})
            if direction == 'inbound':
                current_rule = default_rules.get('in', 'undefined')
                print(f"Current DEFAULT inbound rule: {current_rule}")
            elif direction == 'outbound':
                current_rule = default_rules.get('out', 'undefined')
                print(f"Current DEFAULT outbound rule: {current_rule}")
        else:
            # Show specific rule
            if direction == 'inbound':
                current_rule = current_rules.get('in', 'undefined')
                print(f"Current {target_subnet} inbound rule: {current_rule}")
            elif direction == 'outbound':
                current_rule = current_rules.get('out', 'undefined')
                print(f"Current {target_subnet} outbound rule: {current_rule}")
        
        # Get new rule setting
        print(f"\nSet {direction.upper()} rule:")
        new_rule = self.get_nacl_rule_choice()
        
        return self.apply_nacl_rule_change(source_subnet, target_subnet, direction, new_rule)

    def display_nacl_rules_matrix(self) -> None:
        """Show comprehensive NACL rules in matrix format - ENHANCED VERSION"""
        if not self.scenario_data:
            print("ERROR: No scenario loaded!")
            return
        
        subnets = list(self.scenario_data.get('Subnets', {}).keys())
        
        print("\n" + "="*80)
        print("SUBNET NACL RULES MATRIX")
        print("="*80)
        
        print(f"\nNetwork NACL Overview: {len(subnets)} subnets")
        print()
        
        # Show NACL rules for each subnet
        for subnet_name in subnets:
            subnet_nacls = self.current_nacl_rules.get(subnet_name, {})
            
            print(f"┌─ {subnet_name.upper()} SUBNET NACL RULES:")
            
            if not subnet_nacls:
                print("│  └─ No NACL rules defined")
            else:
                # Show 'all' rule first (most important)
                if 'all' in subnet_nacls:
                    rules = subnet_nacls['all']
                    inbound = rules.get('in', 'undefined')
                    outbound = rules.get('out', 'undefined')
                    
                    print(f"│  ├─ DEFAULT (all): IN={inbound} | OUT={outbound}")
                
                # Show specific subnet rules
                specific_rules = {k: v for k, v in subnet_nacls.items() if k != 'all'}
                for i, (target, rules) in enumerate(specific_rules.items()):
                    is_last = (i == len(specific_rules) - 1) and 'all' in subnet_nacls
                    prefix = "│  └─" if is_last else "│  ├─"
                    
                    inbound = rules.get('in', 'undefined')
                    outbound = rules.get('out', 'undefined')
                    
                    print(f"{prefix} SPECIFIC ({target}): IN={inbound} | OUT={outbound}")
            
            print("│")
        
        print("="*80)
        print("LEGEND:")
        print("  DEFAULT (all): Applies to all other subnets unless specific rule exists")
        print("  SPECIFIC: Overrides default rule for specific subnet")
        print("  IN: Inbound traffic | OUT: Outbound traffic")
        print("  all: Allow traffic | None: Block traffic")
        print("="*80)

    def display_subnet_nacl_details(self, subnet_name: str) -> None:
        """Show detailed NACL rules for specific subnet"""
        if subnet_name not in self.scenario_data.get('Subnets', {}):
            print(f"ERROR: Subnet '{subnet_name}' not found!")
            return
        
        nacl_rules = self.current_nacl_rules.get(subnet_name, {})
        
        print(f"\n" + "="*60)
        print(f"DETAILED NACL RULES: {subnet_name.upper()} SUBNET")
        print("="*60)
        
        if not nacl_rules:
            print("No NACL rules configured for this subnet.")
            print("Default behavior: Block all traffic")
        else:
            print(f"Active NACL rules: {len(nacl_rules)}")
            print()
            
            for target, rules in nacl_rules.items():
                inbound = rules.get('in', 'undefined')
                outbound = rules.get('out', 'undefined')
                
                print(f"Target: {target}")
                print(f"  ├─ Inbound:  {inbound}")
                print(f"  └─ Outbound: {outbound}")
                
                # Explain what these rules mean
                if target == 'all':
                    print(f"     (Default rule for all unspecified subnets)")
                else:
                    print(f"     (Specific rule for {target} subnet)")
                print()
        
        print("="*60)

    def add_header_comment(self, file_path: str) -> None:
        """Add informative header to the generated file"""
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Count subnets and hosts
        subnet_count = len(self.scenario_data.get('Subnets', {}))
        host_count = len(self.scenario_data.get('Hosts', {}))
        
        header = textwrap.dedent(f"""\
        # Enhanced CAGE Challenge Scenario
        # Generated by Enhanced Subnet Builder V3.0
        # Original patterns preserved for agent compatibility
        #
        # Network Summary:
        # - Total Subnets: {subnet_count}
        # - Total Hosts:   {host_count}
        # - Enhanced from: {self.scenario_path.name}
        # - Critical paths: {len(self.critical_paths)}
        #
        # Usage:
        # cyborg = CybORG('{file_path}', 'sim')
        #
        """)
        
        with open(file_path, 'w') as file:
            file.write(header + "\n" + content)
            
    def validate_nacl_change_safety(self, source_subnet: str, target_subnet: str, direction: str, new_rule: str) -> Tuple[bool, List[str]]:
        """Check if NACL change breaks critical functionality"""
        warnings = []
        
        # Check critical path preservation
        if (source_subnet.lower() == 'user' and target_subnet.lower() == 'enterprise' and 
            direction in ['outbound', 'both'] and new_rule == 'None'):
            warnings.append(" This will block User->Enterprise path (critical for CAGE 2)")
        
        if (source_subnet.lower() == 'enterprise' and target_subnet.lower() == 'operational' and 
            direction in ['outbound', 'both'] and new_rule == 'None'):
            warnings.append(" This will block Enterprise->Operational path (critical for CAGE 2)")
        
        # Check for complete subnet isolation
        if target_subnet == 'all' and new_rule == 'None':
            if direction in ['both', 'inbound', 'outbound']:
                warnings.append(f" This will isolate {source_subnet} subnet completely")
        
        # Return safety status
        is_safe = len(warnings) == 0
        return is_safe, warnings

    

    def _apply_nacl_rules_to_scenario(self) -> None:
        """Apply current NACL rules to scenario data"""
        subnets = self.scenario_data.get('Subnets', {})
        
        for subnet_name, nacl_rules in self.current_nacl_rules.items():
            if subnet_name in subnets:
                subnets[subnet_name]['NACLs'] = copy.deepcopy(nacl_rules)

    def backup_current_nacl_state(self) -> str:
        """Save current NACL configuration for rollback"""
        import datetime
        backup_id = f"backup_{datetime.datetime.now().strftime('%H%M%S')}"
        
        # Store backup (simple implementation)
        if not hasattr(self, 'nacl_backups'):
            self.nacl_backups = {}
        
        self.nacl_backups[backup_id] = copy.deepcopy(self.current_nacl_rules)
        return backup_id

    def rollback_nacl_changes(self, backup_id: str) -> bool:
        """Restore previous NACL configuration"""
        try:
            if hasattr(self, 'nacl_backups') and backup_id in self.nacl_backups:
                self.current_nacl_rules = copy.deepcopy(self.nacl_backups[backup_id])
                self._apply_nacl_rules_to_scenario()
                print(f" NACL rules restored from backup: {backup_id}")
                return True
            else:
                print(f"ERROR: Backup {backup_id} not found")
                return False
        except Exception as e:
            print(f"ERROR during rollback: {str(e)}")
            return False
        
    def create_subnet_definition(self, config: Dict, new_hosts: Dict) -> Dict:
        """Create subnet definition with appropriate NACLs"""
        subnet_name = config['name']
        
        # Create basic subnet definition
        subnet_def = {
            'Hosts': list(new_hosts.keys()),
            'NACLs': {
                'all': {
                    'in': 'all',
                    'out': 'all'
                }
            },
            'Size': config['host_count']
        }
        
        # Add security-based NACLs for high-security subnets
        if config['security_level'] == 'High':
            subnet_def['NACLs']['User'] = {
                'in': 'None',
                'out': 'all'
            }
        
        # NEW: Store in current NACL rules tracking
        self.current_nacl_rules[subnet_name] = copy.deepcopy(subnet_def['NACLs'])
        
        return subnet_def    
            
    def configure_nacl_rules(self) -> bool:
        """Main NACL configuration interface"""
        if not self.scenario_data:
            print("ERROR: No scenario loaded!")
            return False
        
        while True:
            print("\n" + "="*60)
            print("NACL CONFIGURATION MENU")
            print("="*60)
            
            if self.modify_specific_subnet_rules():
                        print("NACL rules updated successfully!")
            
            return True    
            
            """print("1. Modify specific subnet rules")
            print("2. Apply security template")
            print("3. Reset to original CAGE 2 defaults")
            print("4. View detailed rules for subnet")
            print("5. Return to main menu")
            
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1': #Only permit this option
                    if self.modify_specific_subnet_rules():
                        print("NACL rules updated successfully!")
                    
                elif choice == '2':
                    if self.apply_nacl_security_template():
                        print("Security template applied successfully!")
                    
                elif choice == '3':
                    if self.reset_to_original_nacls():
                        print("NACL rules reset to original CAGE 2 defaults!")
                    
                elif choice == '4':
                    self.view_detailed_subnet_nacls()
                    
                elif choice == '5':
                    return True
                    
                else:
                    print("ERROR: Please enter a valid option (1-5)")
                    
            except KeyboardInterrupt:
                print("\nReturning to main menu...")
                return True
            except Exception as e:
                print(f"ERROR: {str(e)}")
                return False"""

    def modify_specific_subnet_rules(self) -> bool:
        """Modify NACL rules for specific subnet"""
        # Step 1: Select source subnet
        source_subnet = self.select_subnet_for_nacl_modification()
        if not source_subnet:
            return False
        
        # Step 2: Select target and direction
        target_subnet, direction = self.select_nacl_target_and_direction(source_subnet)
        if not target_subnet or not direction:
            return False
        
        # Step 3: Configure the rule
        return self.configure_specific_nacl_rule(source_subnet, target_subnet, direction)

    def select_subnet_for_nacl_modification(self) -> str:
        """Let user choose which subnet's NACLs to modify"""
        subnets = list(self.scenario_data.get('Subnets', {}).keys())
        
        print("\nSELECT SUBNET TO MODIFY:")
        print("Choose which subnet's NACL rules to configure:")
        print()
        
        for i, subnet in enumerate(subnets, 1):
            nacl_count = len(self.current_nacl_rules.get(subnet, {}))
            critical_info = ""
            
            # Mark critical subnets
            if subnet.lower() == 'operational':
                critical_info = " [CRITICAL - Has security restrictions]"
            elif len([cp for cp in self.critical_paths if cp['subnet'] == subnet]) > 0:
                critical_info = " [Has critical paths]"
            
            print(f"{i}. {subnet} ({nacl_count} NACL rules){critical_info}")
        
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(subnets)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(subnets):
                    return subnets[choice_idx]
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(subnets)}")
            except ValueError:
                print("ERROR: Please enter a valid number")

    

    def get_nacl_rule_choice(self) -> str:
        """Get user choice for NACL rule (all/None)"""
        print("1. all  (Allow all traffic)")
        print("2. None (Block all traffic)")
        
        while True:
            try:
                choice = input("Enter choice (1-2): ").strip()
                if choice == '1':
                    return 'all'
                elif choice == '2':
                    return 'None'
                else:
                    print("ERROR: Please enter 1 or 2")
            except ValueError:
                print("ERROR: Please enter a valid choice")

    def display_summary(self) -> None:
        """Display summary of the enhanced scenario"""
        if not self.scenario_data:
            return
            
        print("\n" + "="*70)
        print("ENHANCED SCENARIO SUMMARY")
        print("="*70)
        
        subnets = self.scenario_data.get('Subnets', {})
        hosts = self.scenario_data.get('Hosts', {})
        
        print(f"\nNetwork Overview:")
        print(f"   Total Subnets: {len(subnets)}")
        print(f"   Total Hosts: {len(hosts)}")
        print(f"   Critical Paths: {len(self.critical_paths)}")
        
        print(f"\nSubnet Breakdown:")
        for subnet_name, subnet_data in subnets.items():
            host_count = len(subnet_data.get('Hosts', []))
            critical_hosts = [cp['host'] for cp in self.critical_paths 
                            if cp['subnet'] == subnet_name]
            critical_info = f" ({len(critical_hosts)} critical)" if critical_hosts else ""
            print(f"   {subnet_name:15} | {host_count:2} hosts{critical_info}")
        
        print(f"\nAgent Configuration:")
        for agent_name, agent_data in self.scenario_data['Agents'].items():
            subnet_access = len(agent_data.get('AllowedSubnets', []))
            sessions = len(agent_data.get('starting_sessions', []))
            print(f"   {agent_name:6} | {subnet_access} subnets | {sessions:2} sessions")
        
        """print(f"\nCritical Path Analysis:")
        if self.critical_paths:
            for i, path in enumerate(self.critical_paths, 1):
                host = path['host']
                subnet = path['subnet']
                ext_conns = len(path['external_connections'])
                print(f"   {i}. {host} ({subnet}) -> {ext_conns} external connections")
        else:
            print("   No critical paths identified")"""
        
        print("\n" + "="*70)

    

def main():
    """Main interactive function for Enhanced Subnet Builder"""
    print("CAGE CHALLENGE ENHANCED SUBNET BUILDER V3.7")
    print("=" * 55)
    print("Advanced subnet builder with pattern preservation")
    print("Maintains original network topology for agent compatibility")
    print("Works with any CAGE scenario file\n")
    
    builder = EnhancedSubnetBuilder()
    
    # Load scenario file with default option, cause I am leasy :p
    while True:
        file_path = input("Enter path to CAGE scenario file (or press Enter for 'Scenario2.yaml' as a default): ").strip()
        
        # If user presses Enter without typing anything, use default
        if not file_path:
            file_path = "Scenario2.yaml"
            print(f"Using default scenario: {file_path}")
            
            # Check if default file exists before trying to load it
            if not Path(file_path).exists():
                print(f"ERROR: Default file '{file_path}' not found in current directory!")
                print("Please specify the full path to your scenario file.")
                continue  # Go back to asking for input
        
        if builder.load_scenario(file_path):
            break
        print("Please try again with a valid file path.\n")
    
    # Main interaction loop
    while True:
        print("\nENHANCED MAIN MENU")
        print("1. View network topology")
        print("2. Add new subnet")
        print("3. Add hosts to existing subnet")
        print("4. Connect existing subnets")     
        print("5. Save enhanced scenario")    
        print("6. View scenario summary") 
        print("7. Configure NACL rules (Firewall)")      
        print("8. View NACL matrix")      
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == '1':
            builder.display_current_subnets()
            
        elif choice == '2':
            if builder.build_subnet():
                print("\nSUCCESS: Subnet added with pattern preservation!")
                print("Original network topology has been preserved.")
            else:
                print("\nERROR: Failed to add subnet. Please try again.")
                
        elif choice == '3':
            if builder.add_hosts_to_existing_subnet():
                print("\nSUCCESS: Hosts added with pattern preservation!")
                print("The new hosts inherit existing subnet characteristics.")
            else:
                print("\nERROR: Failed to add hosts. Please try again.")
                
        elif choice == '4':
            if builder.connect_existing_subnets():
                print("\nSUCCESS: Subnets connected with pattern preservation!")
                print("Original network topology has been preserved.")
            else:
                print("\nERROR: Failed to connect subnets. Please try again.")
            
        elif choice == '5':
            output_file = input("\nEnter output filename (or press Enter for auto-name): ").strip()
            output_file = output_file if output_file else None
            
            if builder.save_scenario(output_file):
                print("SUCCESS: Enhanced scenario saved!")
                print("The scenario maintains compatibility with existing agents.")
            else:
                print("ERROR: Failed to save scenario.")
                
        elif choice == '6':
            builder.display_summary()
            
        elif choice == '7':
            if builder.configure_nacl_rules():
                print("\nNACL configuration completed.")
            else:
                print("\nERROR: Failed to configure NACL rules.")
                
        elif choice == '8':
            builder.display_nacl_rules_matrix()
            
        elif choice == '9':
            print("\nSee you in other life :D")
            break
            
        else:
            print("ERROR: Please enter a valid option (1-9)")


if __name__ == "__main__":
    main()