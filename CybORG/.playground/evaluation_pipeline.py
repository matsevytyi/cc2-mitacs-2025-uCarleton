import os, sys, inspect
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import PaddingWrapper, TransformerWrapper

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from scenario_shuffler import update_yaml_file

# ========== CONFIGURATION ==========
ALGORITHMS = [DQN, PPO]  # Evaluate both
transformer = True
extended = True

RUN_ID = "CLS[1,2*64] perhost + SPLIT BACKPROP + save weights"
TOTAL_TIMESTEPS = 500_000

weights_path = "CLS[1,2*64] perhost + SPLIT BACKPROP + save weights + 1.0"

def create_cyborg_env(yaml_path: str):
    """
    Create fresh CybORG environment from YAML
    This will be called every reset() to reload topology
    """
    cyborg = CybORG(yaml_path, 'sim', agents={'Red': RedMeanderAgent})
    cyborg.reset()
    return cyborg

# ========== HELPER FUNCTIONS ==========
def reload_env(path, transformer, shuffle=False, algorithm=None):
    if shuffle:
        new_assignments = {
            'Enterprise': ['User0', 'Enterprise2', 'Enterprise1', 'Op_Host2'],
            'Operational': ['Op_Host1', 'Enterprise0', 'User4', 'Op_Server0'],
            'User': ['Op_Host0', 'User3', 'User1', 'Defender', 'User2']
        }
        update_yaml_file(path, mode='assign', new_assignments=new_assignments)
    
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    
    if transformer:
        gym_env = TransformerWrapper(
            raw_cyborg=cyborg, 
            agent_name='Blue', 
            max_steps=100, 
            weights_path=f"TRAIN.{algorithm}.{weights_path}.encoder.pth")
    else:
        gym_env = PaddingWrapper(
            env=cyborg, agent_name='Blue', 
            max_devices=100, 
            max_steps=100
            )
    
    gym_env.reset()
    return gym_env

def evaluate_model(algorithm_class, filename, gym_env, env_name):
    """Evaluate a model on a given environment"""
    try:
        model = algorithm_class.load(filename, env=gym_env)
        mean_reward, std_reward = evaluate_policy(
            model, gym_env, n_eval_episodes=10, 
            deterministic=True, return_episode_rewards=True
        )
        
        print(f"\n{algorithm_class.__name__} on {env_name}:")
        print(f"  Mean reward: {0.5 * np.max(mean_reward) + 0.5 * np.min(mean_reward):.2f}")
        print(f"  Max reward:  {np.max(mean_reward):.2f}")
        print(f"  Min reward:  {np.min(mean_reward):.2f}")
        print(f"  Std reward:  {np.mean(std_reward):.2f}")
        
        return {
            'algorithm': algorithm_class.__name__,
            'env': env_name,
            'mean': 0.5 * np.max(mean_reward) + 0.5 * np.min(mean_reward),
            'max': np.max(mean_reward),
            'min': np.min(mean_reward),
            'std': np.mean(std_reward)
        }
    except Exception as e:
        print(f"Error evaluating {algorithm_class.__name__} on {env_name}: {e}")
        return None

# ========== PATHS ==========
if extended:
    path_1 = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".playground/scenarios/Scenario2.yaml"
    )
else:
    path_1 = str(inspect.getfile(CybORG))
    path_1 = path_1[:-10] + '/Shared/Scenarios/Scenario2.yaml'

# ========== EVALUATION ==========
results = []

for algorithm in ALGORITHMS:
    algorithm_name = algorithm.__name__
    wrapper_type = "transformer" if transformer else "padding"
    filename = f"{algorithm_name}_{wrapper_type}_model_{TOTAL_TIMESTEPS}_{RUN_ID}.zip"
    
    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm_name} ({wrapper_type} wrapper)")
    print(f"{'='*60}")
    
    # Training environment
    gym_env_train = reload_env(path_1, transformer, shuffle=False, algorithm=algorithm_name)
    result = evaluate_model(algorithm, filename, gym_env_train, "Training Env")
    if result:
        results.append(result)
    

# ========== SUMMARY ==========
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for r in results:
    print(f"{r['algorithm']:8} | {r['env']:15} | Mean: {r['mean']:7.2f} | Max: {r['max']:7.2f} | Min: {r['min']:7.2f}")
