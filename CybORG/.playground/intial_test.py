import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inspect
from CybORG import CybORG
path = str(inspect.getfile(CybORG))
 path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

# Create the environment
cyborg = CybORG(path, 'sim')

# Reset the environment and take a step
cyborg.reset()
results = cyborg.step(action='Sleep', agent='Red')

# Inspect the result
print(results.observation)
