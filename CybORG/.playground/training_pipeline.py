import os, sys, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from CybORG.Agents.Wrappers import TransformerWrapper, ChallengeWrapper, PaddingWrapper

from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from gymnasium.wrappers.compatibility import LegacyEnv
from gymnasium.wrappers import EnvCompatibility


from stable_baselines3 import DQN, PPO
#from sb3_contrib import C51


from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor


# Load cfg
extended = True
transformer = True
pad = False

algorithm = DQN
#algorithm = PPO

# subnet_id = 0

# Locate Scenario2.yaml path
if extended:

    path = "Scenario2.yaml"

    path = os.path.dirname(os.path.dirname(__file__)) + "/.playground/scenarios/" + path
else:

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
cyborg.reset()



if transformer:
    gym_env = TransformerWrapper(raw_cyborg=cyborg, agent_name='Blue', max_steps=50)
elif pad:
    gym_env = PaddingWrapper(env=cyborg, agent_name='Blue', max_devices=100, max_steps=50)
else:
    gym_env = ChallengeWrapper(env=cyborg, agent_name='Blue', max_steps=50)
    gym_env = EnvCompatibility(gym_env)
    #gym_env = StepAPICompatibility(gym_env, output_truncation_bool=True)
    
gym_env.reset()

print(cyborg)

# ppo/c51 - try

model1 = algorithm(
    policy='MlpPolicy',
    env=gym_env,
    verbose=2,
    tensorboard_log="./logs/",
    device='cpu',
    # Replay Buffer
    buffer_size=500_000,      # Big but fits on T4
    learning_starts=10_000,   # fill buffer with diversity

    # Training
    #batch_size=64,        
    train_freq=4,           
    gradient_steps=1,         # 1 gradient update per train step
    target_update_interval=5_000, 

    # Optimization
    learning_rate=1e-5,       # Default, safe for DQN; lower (5e-5) if unstable
    gamma=0.99,               
    exploration_fraction=0.2, # How fast epsilon decays 0.2 > 0.3 > 0.1
    exploration_final_eps=0.025, 
)

# model1 = PPO(
#     policy='MlpPolicy',
#     env=gym_env,
#     verbose=2,
#     tensorboard_log="./logs/",
#     device='cpu',
#     learning_rate=3e-4,
#     n_steps=2048,
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     ent_coef=0.0,
#     vf_coef=0.5,
#     max_grad_norm=0.5
# )

#run_id = "padding to 0 version on {subnet_id}"
run_id = "CLS[1, 2*64] + simple obs + FULL subnet + norm + training_pipeline + random CLS on 5000 epochs_50 episodes + 2 transformer heads + 4 layers + bigger decay = 0.2 + smaller exploration_final_eps=0.025"
#run_id = "padding to -1 version"

model1.learn(
    total_timesteps=500_000,
    tb_log_name=f"{algorithm.__name__}_run_{run_id}",
    log_interval=10
    )

filename = f"dqn_transformer_model_100000_{run_id}" if transformer else "dqn_padding_model_100000"
model1.save(filename)