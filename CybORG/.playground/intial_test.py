import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inspect
from CybORG import CybORG

extended = True

# Locate Scenario2.yaml path
if extended:
    # path = "Scenario2.yaml"
    # path = "Scenario2_Linear.yaml"
    # path = "Scenario2_Extended.yaml"
    path = os.path.dirname(__file__) + "/scenarios/Scenario2_Linear.yaml"
else:
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

# Create the environment
cyborg = CybORG(path, 'sim')

# Reset the environment and take a step
cyborg.reset()
results = cyborg.step(action='Sleep', agent='Red')

# Inspect the result
print(results.observation)
