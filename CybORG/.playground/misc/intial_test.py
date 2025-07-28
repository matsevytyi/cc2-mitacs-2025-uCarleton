import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inspect
from CybORG import CybORG

from CybORG.Agents import B_lineAgent

extended = False

# Locate Scenario2.yaml path
if extended:

    #scenario_name = "Scenario2.yaml"
    #scenario_name = "Scenario2_Linear.yaml"
    scenario_name = "Scenario2_Extended.yaml"

    path = os.path.dirname(__file__) + "/scenarios/" + scenario_name

else:

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

# Create the environment
cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})

# Reset the environment and take a step
cyborg.reset()

for _ in range(1):
    cyborg.step(agent='Blue', action='Sleep')

results = cyborg.step(action='Sleep', agent='Red')

# Inspect the result

for i in range(20):
    results = cyborg.step(agent='Red')
    print(f"\nStep {i+1} Observation:")
    print(results.observation)
    print("Action taken:", results.action)
    print("Reward:", results.reward)
    print("Done:", results.done)


