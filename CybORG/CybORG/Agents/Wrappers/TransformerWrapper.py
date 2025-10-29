from types import NoneType
from gymnasium import Env, spaces
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper
from scenario_shuffler import churn_hosts

import numpy as np

import csv
from datetime import datetime


from CybORG.Agents.Wrappers.TransformerStateEncoder import TransformerStateEncoder

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Note: modified challengewrapper to use encoder
class TransformerWrapper(Env,BaseWrapper):
    def __init__(self, agent_name: str, raw_cyborg, agent=None,
            reward_threshold=None, max_steps = None, max_actions=None, 
            action_space_mode="pad", 
            env_creator=None, yaml_path=None,
            device='cpu', version="ip_local", weights_path=None):
        super().__init__(raw_cyborg, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')
        
        self.version = version
        self.raw_cyborg = raw_cyborg
        
        self.host_order = tuple(self.raw_cyborg.environment_controller.state.hosts.keys()) # to freeze the order

        # upscale the wrapper similarly to challenge wrapper
        env = table_wrapper(raw_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)
        
        self.table_env = table_wrapper(raw_cyborg, output_mode='table')

        self.action_history = []
        self.recon_loss_history = []

        if env_creator and yaml_path:
            self.env_creator = env_creator
            self.yaml_path = yaml_path
        
        self.env = env
        # fixed action space padding/cutoff
        self.max_actions = max_actions
        self.action_space_mode = action_space_mode  # "pad" or "cutoff"
        if self.max_actions is not None:
            self.action_space = spaces.Discrete(int(self.max_actions))
        else:
            self.action_space = self.env.action_space

        self.device = device

        embedding_dim = 64 # transformer embedding dimension
        observation_shape_size = embedding_dim*2 # or *1 depending on the architecture

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_shape_size, ),  # embedding_dim from transformer encoder
            dtype=np.float32
        )

        self.transformer_encoder = TransformerStateEncoder(
            observation_space=self.observation_space,
            embedding_dim=embedding_dim,
            initial_host_count=len(self.host_order)
        ).to(self.device)
        

        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = 0

        if weights_path:
            self.transformer_encoder.load_weights(weights_path)


    def step(self,action=None, debug=False, verbose=False):

        self.action_history.append(self.decode_action(action)[1])
        self.recon_loss_history.append(self.transformer_encoder.recon_loss.item())
        
        # Map out-of-range actions to a valid one based on the selected mode
        if action is not None:
            try:
                n_valid = int(self.env.action_space.n)
            except Exception:
                n_valid = None
            if self.max_actions is not None and n_valid is not None and n_valid > 0:
                if action >= n_valid:
                    if self.action_space_mode == "cutoff":
                        action = n_valid - 1
                    else:
                        action = self.env.action_space.sample()
                        
        obs, reward, terminated, info = self.env.step(action=action)
        
        # self.env.env.env.env.env is same as active self.raw_cyborg

        # enrich obs with other contextual information
        obs = self.extract_host_state(raw_cyborg=self.env.env.env.env.env, obs=obs)
        if verbose:
            print("action", action, self.decode_action(action))
            print('step obs', obs)
    
        self.step_counter += 1
        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True
        
        # with torch.no_grad():
        #     encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)
        encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)

        return encoded_obs.detach().cpu().numpy(), reward, terminated, truncated, info

    def reset(self, **kwargs):

        if self.step_counter > 0:
            self._reload_environment()

            self.step_counter = 0
        obs = self.env.reset(**kwargs)
        
        # enrich obs with other contextual information
        obs = self.extract_host_state(raw_cyborg=self.env.env.env.env.env, obs=obs)

        # with torch.no_grad():
        #     encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)
        encoded_obs = self.transformer_encoder(obs, self.host_order, version=self.version)

        if self.step_counter > 0:
            ## update csv with actions from self.action_history and set self.action_history to []
            if self.action_history:
                # Create output directory if not exists
                csv_dir = "action_logs"
                os.makedirs(csv_dir, exist_ok=True)

                # File unique per agent_name (or add timestamp if you want)
                csv_path = os.path.join(csv_dir, f"actions_{self.agent_name}_HOTRELOAD_Transformer_RedMeander_DQN.csv")

                # Append mode is safe (creates file if not exists)
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Optional: write header if file is new
                    if os.stat(csv_path).st_size == 0:
                        writer.writerow(['timestamp', 'episode', 'step', 'action_type', 'host_count', 'recon_loss'])
                    timestamp = datetime.now().isoformat()
                    for step, action in enumerate(self.action_history):
                        writer.writerow([timestamp, getattr(self, 'episode', None), step, action, len(self.host_order), self.recon_loss_history[step]])

            # Clear history after save
            self.action_history = []
            self.recon_loss_history = []

            # reload env        
            
        return encoded_obs.detach().cpu().numpy(), {}
    
    def _reload_environment(self, verbose=False):
        """
        INTERNAL: Reload CybORG environment from scratch
        Called on reset() to pick up YAML changes
        """
        try: 
            if self.env_creator is None or self.yaml_path is None:
                print("Reload Ignored")
                return False
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to reload environment: {e}")
            return False    
        
        # try:

        #print("Reload in place")
        # churn devices
        churn_hosts(self.yaml_path)
        
        # create fresh CybORG instance
        fresh_cyborg = self.env_creator(self.yaml_path)
        self.raw_cyborg = fresh_cyborg
        
        # Recreate wrapper stack
        if self.agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            table_wrapper = RedTableWrapper
        
        env = table_wrapper(fresh_cyborg, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=self.agent_name, env=env)
        
        self.env = env
        self.table_env = table_wrapper(fresh_cyborg, output_mode='table')
        
        # Update host order (may have changed)
        self.host_order = tuple(self.raw_cyborg.environment_controller.state.hosts.keys())

        # dump dynamically old weights
        old_weights = {
            'transformer': self.transformer_encoder.transformer.state_dict(),
            'obs_embed': self.transformer_encoder.obs_embed.state_dict(),
            'ip_byte_embed': self.transformer_encoder.ip_byte_embed.state_dict(),
            'cls_token': self.transformer_encoder.cls_token.data,
            'token_head_from_cls': self.transformer_encoder.token_head_from_cls.state_dict(),
        }
        
        # Reinitialize encoder with new host count
        self.transformer_encoder = TransformerStateEncoder(
            observation_space=self.observation_space,
            embedding_dim=64,
            initial_host_count=len(self.host_order)
        ).to(self.device)

        self.transformer_encoder.load_weights_from_dict(old_weights)
        
        return True
        # except Exception as e:
        #     print(f"Warning: Failed to reload environment: {e}")
        #     return False

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)

    def extract_host_state(self, raw_cyborg, obs):
        hosts_dict = {}
        for i, hname in enumerate(self.host_order):
            hstate = raw_cyborg.environment_controller.state.hosts[hname]
            
            obs_chunk = obs[4 * i : 4 * (i + 1)] # original signal
            
            ips = [
                iface.ip_address.__str__()
                for iface in hstate.interfaces
                if iface.name.startswith("eth")
            ]
            
            subnets = [
                iface.subnet.__str__().split("/")[0]
                for iface in hstate.interfaces
                if iface.name.startswith("eth")
            ]
            
            ports = np.array([
                conn.get("local_port")
                for proc in hstate.processes
                for conn in proc.connections
            ])
            
            processes = np.array([proc.name for proc in hstate.processes])
            
            hosts_dict[hname] = {
                "obs": obs_chunk,
                "ips": ips,
                "subnets": subnets,
                "ports": ports,
                "processes": processes,
            }
            
        return hosts_dict

    # Total actions = 2 + (N * 11)
# where N is the number of hosts in the network

    # Device-agnostic actions (always indices 0-1)
    SLEEP = 0
    MONITOR = 1

    # Device-specific actions for host i (where i ranges from 0 to N-1)
    def get_action_index(self, host_id, action_type):
        """
        host_id: int from 0 to N-1
        action_type: int from 0 to 10 representing one of the 11 actions
        
        Action types:
        0: Analyze
        1: Remove  
        2: Restore
        3-10: Eight different Decoy services
        """
        return 2 + (host_id * 11) + action_type

    # Inverse mapping: from flat action index to (device, action)
    def decode_action(self, action_idx):

        """
        Convert flat action index to (device_id, action_type) tuple
        
        Returns:
            (device_id, action_type) where:
            - device_id is None for device-agnostic actions, or 0 to num_hosts-1
            - action_type: 0=Sleep, 1=Monitor, or 0-10 for device actions
        """
        if action_idx == 0:
            return (None, 'Sleep')
        elif action_idx == 1:
            return (None, 'Monitor')
        else:

            action_names = ['Analyze', 'Remove', 'Restore', 
                        'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMTP',
                        'DecoySmss', 'DecoySSHD', 'DecoySvchost', 
                        'DecoyTomcat', 'DecoyVsftpd']

            # Device-specific action
            adjusted_idx = action_idx - 2
            device_id = adjusted_idx // 11
            action_type = adjusted_idx % 11
            
            return (device_id, action_names[action_type])
