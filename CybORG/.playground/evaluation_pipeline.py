import os, sys, inspect
import time
from statistics import mean, stdev
import numpy as np

# CybORG Imports
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

# Your Custom Imports (Adjust paths as necessary)
from CybORG.Agents.Wrappers import TransformerWrapper, DeepSetsPermInvWrapper, PaddingWrapper
from stable_baselines3 import PPO, DQN

# ========== CONFIGURATION ==========
MAX_EPS = 100
agent_name = 'Blue'
scenario = 'Scenario2'

# Which architecture are you evaluating?
mode = "transformer" # "transformer", "ds", or "padding"
ALGORITHM = DQN

transformer = (mode == "transformer")
deepset = (mode == "ds")

# Model and Weights Paths
ARCH_STR = "Transformer" if transformer else "DeepSets" if deepset else "Padding"
MODEL_PATH = f"{ALGORITHM.__name__}_{ARCH_STR}_tuning_x150_extended_dynamic_topology.zip"
WEIGHTS_PATH = f"{ALGORITHM.__name__}_{ARCH_STR}_tuning_x150_extended_dynamic_topology.encoder.pth"

base_dir = os.path.dirname(os.path.dirname(__file__))
SCENARIO_PATH = os.path.join(base_dir, f".playground/scenarios/Scenario2_{ARCH_STR}_{ALGORITHM.__name__}.yaml")

def create_cyborg_env(yaml_path: str, red_agent_class):
    """Creates a raw CybORG environment with a specific Red agent"""
    cyborg = CybORG(yaml_path, 'sim', agents={'Red': red_agent_class})
    cyborg.reset()
    return cyborg

def wrap(cyborg_env, red_agent_class):
    """Wraps the raw CybORG env with your specific Neural Encoder Wrapper"""
    if transformer:
        return TransformerWrapper(
            agent_name=agent_name,
            raw_cyborg=cyborg_env,
            max_steps=100,
            knowledge_update_mode="eval", # Important: disables topology tuning logic
            env_creator=lambda path: create_cyborg_env(path, red_agent_class),
            yaml_path=SCENARIO_PATH,  
            max_actions=240,       
            weights_path=WEIGHTS_PATH
        )
    elif deepset:
        return DeepSetsPermInvWrapper(
            agent_name=agent_name,
            raw_cyborg=cyborg_env,
            max_steps=100,
            knowledge_update_mode="eval",
            env_creator=lambda path: create_cyborg_env(path, red_agent_class),
            yaml_path=SCENARIO_PATH,  
            max_actions=240,       
            weights_path=WEIGHTS_PATH
        )
    else:
        return PaddingWrapper(
            agent_name=agent_name,
            env=cyborg_env,
            max_steps=100,
            knowledge_update_mode="eval",
            env_creator=lambda path: create_cyborg_env(path, red_agent_class),
            yaml_path=SCENARIO_PATH,  
        )

if __name__ == "__main__":
    print(f"Loading {ALGORITHM.__name__} model: {MODEL_PATH}")
    
    # 1. We must load the model using a dummy environment first
    dummy_raw = create_cyborg_env(SCENARIO_PATH, B_lineAgent)
    dummy_wrapped = wrap(dummy_raw, B_lineAgent)
    model = ALGORITHM.load(MODEL_PATH, env=dummy_wrapped)
    
    # Setup Output File
    file_name = f'Evaluation_{time.strftime("%Y%m%d_%H%M%S")}_{ARCH_STR}_{ALGORITHM.__name__}.txt'
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{CYBORG_VERSION}, {scenario}\n')
        data.write(f'Model: {MODEL_PATH}\n\n')

    print(f'Starting official benchmark for {ARCH_STR}...\n')
    
    # Standard official CAGE loops
    for num_steps in [30, 50, 100]:
        for red_agent in [RedMeanderAgent, SleepAgent]:

            # Re-initialize the raw and wrapped envs for the specific Red Agent
            cyborg = create_cyborg_env(SCENARIO_PATH, red_agent)
            wrapped_cyborg = wrap(cyborg, red_agent)
            
            # Since max_steps varies by loop, update the wrapper manually
            wrapped_cyborg.max_steps = num_steps

            total_reward = []
            
            for i in range(MAX_EPS):
                r = []
                
                # Unlike SB3, we must handle the reset tuples manually here
                reset_result = wrapped_cyborg.reset()
                # Gymnasium usually returns (obs, info) on reset
                observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result

                for j in range(num_steps):
                    # Use the SB3 model predict function
                    # deterministic=True is standard for evaluation
                    action, _states = model.predict(observation, deterministic=True)

                    if isinstance(action, np.ndarray):
                        action = action.item()
                    
                    # Step the wrapper
                    step_result = wrapped_cyborg.step(action)
                    observation, rew, done, truncated, info = step_result
                    
                    r.append(rew)
                    
                    if done or truncated:
                        break
                        
                total_reward.append(sum(r))

            mean_r = mean(total_reward)
            std_r = stdev(total_reward)
            print(f'Steps: {num_steps:<3} | Agent: {red_agent.__name__:<15} | Mean: {mean_r:7.2f} | Std: {std_r:6.2f}')
            
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean_r}, standard deviation {std_r}\n')
