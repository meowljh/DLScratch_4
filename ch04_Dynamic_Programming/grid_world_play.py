import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.grid_world import GridWorld
import numpy as np

if __name__ == "__main__":
    env = GridWorld()
    V = {}
    for state in env.states(): # grid world위, 즉 격자 위에서의 좌표 값 #
        V[state] = np.random.randn()
    env.render_v(v=V, policy=None, print_value=True)    