import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

import inspect
from CybORG import CybORG

from CybORG.Agents import TestAgent
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper

MAX_STEPS_PER_GAME = 100
MAX_EPS = 250

def run_training_example(scenario):
    print("Setup")
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    agent_name = 'Red'
    cyborg = OpenAIGymWrapper(agent_name=agent_name, env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, 'sim'))))

    observation = cyborg.reset(agent=agent_name)
    action_space = cyborg.get_action_space(agent_name)
    print(f"Observation size {len(observation)}, Action Size {action_space}")
    
    action_count = 0
    agent = TestAgent()
    
    for i in range(MAX_EPS):  # laying multiple games
        
        reward = 0
        for j in range(MAX_STEPS_PER_GAME):  # step in 1 game
            
            action = agent.get_action(observation, action_space)
            next_observation, r, done, info = cyborg.step(action=action)
            print(np.shape(observation), action, np.shape(next_observation), r, done, info.keys())
            
            action_space = info.get('action_space')
            reward += r

            agent.train(observation)  # training the agent
            observation = next_observation
            
            if done or j == MAX_STEPS_PER_GAME - 1:
                if reward != 0.0:
                    print(f"\rTraining Game: {i}", end='', flush=True)
                    print(f"\tTraining reward: {reward}")
                break
            
        observation = cyborg.reset(agent=agent_name)
        agent.end_episode()

if __name__ == "__main__":
    run_training_example('Scenario2')
