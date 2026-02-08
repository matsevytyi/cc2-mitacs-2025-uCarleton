import os, sys, inspect
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from CybORG.Agents.Wrappers import TransformerWrapper, ChallengeWrapper, PaddingWrapper

from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from gymnasium.wrappers.compatibility import LegacyEnv
from gymnasium.wrappers import EnvCompatibility

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from misc.RewardLoggingCallback import RewardLoggingCallback

class PerformanceMonitoringCallback(BaseCallback):
    """
    Callback to monitor performance and print actions when TransformerWrapper degrades
    """
    def __init__(self, transformer_env, default_env, verbose=0, window_size=100):
        super().__init__(verbose)
        self.transformer_env = transformer_env
        self.default_env = default_env
        self.window_size = window_size
        self.transformer_rewards = []
        self.default_rewards = []
        self.current_episode_reward = 0.0
        self.episode_count = 0
        self.print_actions = False
        
    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        action = self.locals["actions"][0]
        
        self.current_episode_reward += reward
        
        if done:
            self.transformer_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            
            # Calculate moving averages
            if len(self.transformer_rewards) >= self.window_size:
                transformer_avg = np.mean(self.transformer_rewards[-self.window_size:])
                
                # For comparison, we would need to run the default environment
                # This is a simplified version - in practice you'd want to run both in parallel
                if self.episode_count > self.window_size and len(self.transformer_rewards) >= 2:
                    recent_avg = np.mean(self.transformer_rewards[-self.window_size//2:])
                    earlier_avg = np.mean(self.transformer_rewards[-self.window_size:-self.window_size//2])
                    
                    # Check if performance is degrading
                    if recent_avg < earlier_avg * 0.95:  # 5% performance drop
                        self.print_actions = True
                        if self.verbose:
                            print(f"\n*** PERFORMANCE DEGRADATION DETECTED ***")
                            print(f"Recent average reward: {recent_avg:.4f}")
                            print(f"Earlier average reward: {earlier_avg:.4f}")
                            print("*** ENABLING ACTION PRINTING ***\n")
                    else:
                        self.print_actions = False
            
            self.current_episode_reward = 0.0
        
        # Print actions when performance is degrading
        if self.print_actions:
            print(f"Step {self.num_timesteps}: Action={action}, Reward={reward:.4f}, Episode={self.episode_count}")
            
        return True

print("Starting PPO Training with TransformerWrapper")

# Load configuration
extended = True
transformer = True  # Using TransformerWrapper
pad = False

# Locate Scenario2.yaml path
if extended:
    path = "Scenario2.yaml"
    path = os.path.dirname(os.path.dirname(__file__)) + "/.playground/scenarios/" + path
else:
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

# Create environment with TransformerWrapper
cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
cyborg.reset()

if transformer:
    gym_env = TransformerWrapper(raw_cyborg=cyborg, agent_name='Blue', max_steps=100, version="subnet")
elif pad:
    gym_env = PaddingWrapper(env=cyborg, agent_name='Blue', max_devices=100, max_steps=100)
else:
    gym_env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=100)
    gym_env = EnvCompatibility(gym_env)

# Create a default environment for comparison
cyborg_default = CybORG(path, 'sim', agents={'Red': B_lineAgent})
cyborg_default.reset()
default_env = ChallengeWrapper(env=cyborg_default, agent_name='Blue', max_steps=100)
default_env = EnvCompatibility(default_env)

print(f"Environment setup complete. Using TransformerWrapper: {transformer}")
print(f"Action space: {gym_env.action_space}")
print(f"Observation space: {gym_env.observation_space}")

# Create PPO model
model = PPO(
    policy='MlpPolicy',
    env=gym_env,
    verbose=2,
    tensorboard_log="./logs/",
    device='cpu',
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5
)

# Setup callbacks
reward_callback = RewardLoggingCallback(verbose=1)
performance_callback = PerformanceMonitoringCallback(
    transformer_env=gym_env, 
    default_env=default_env,
    verbose=1,
    window_size=50
)

# Configure logging
log_path = "./logs/"
new_logger = configure(log_path, ["stdout", "tensorboard"])
model.set_logger(new_logger)

print("Starting training...")

# Train the model
model.learn(
    total_timesteps=100000,
    tb_log_name="PPO_TransformerWrapper",
    log_interval=10,
    callback=[reward_callback, performance_callback]
)

# Save the model
filename = "ppo_transformer_model_100000"
model.save(filename)

print(f"Training complete. Model saved as {filename}")