from stable_baselines3 import PPO
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper

import os, sys, inspect

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modify_ss_transformers import TransformerStateEncoder

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


# Initialize CybORG environment
cyborg = CybORG(path, 'sim')

# environment for OpenAI Gym compatibility
env = ChallengeWrapper(env=cyborg, agent_name='Blue')

# obs = cyborg.reset()

# print(obs)

# encoder = TransformerStateEncoder()  # or pass any needed config
# encoded_obs = encoder(obs)
# print("Encoded obs for agent:", encoded_obs)


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Then load as LoadBlueAgent

# Save model
# model.save("ppo_blue_agent")

#  Load and use example
# model = PPO.load("ppo_blue_agent")
# agent = BlueLoadAgent(model_file='ppo_blue_agent.zip')
