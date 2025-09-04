
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_episode_reward += reward
        
        self.logger.dump(self.current_episode_reward)
        if self.verbose:
            print(f"Episode reward: {self.current_episode_reward}")
        if done:
            self.current_episode_reward = 0.0 

        return True