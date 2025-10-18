from gymnasium import Env, spaces
import numpy as np

from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper

# cuts off devices that are out of the observation
class CutoffWrapper(Env,BaseWrapper):
    
    def __init__(self, agent_name: str, env, max_devices = 6, agent=None,
            reward_threshold=None, max_steps = None):
        super().__init__(env, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')
        
        self.max_devices = max_devices

        env = table_wrapper(env, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self.env = env
        self.action_space = self.env.action_space
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_devices * 4, ),  # embedding_dim from transformer encoder
            dtype=np.float32
        )
        
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

    def step(self,action=None):

        if action >= self.env.action_space.n:
            action = self.env.action_space.sample()
        
        obs, reward, terminated, info = self.env.step(action=action)
        obs = self.cutoff_observation(obs, self.max_devices)
        
        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True
            
        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        
        self.step_counter = 0
        
        obs = self.env.reset(**kwargs)
        obs = self.cutoff_observation(obs, self.max_devices)
        
        return obs, {}

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
    
    import numpy as np

    def cutoff_observation(self, obs, max_devices):

        max_len = max_devices * 4
        
        if len(obs) > max_len:
            obs = obs[:max_len]
            
        return obs