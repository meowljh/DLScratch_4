if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from common.grid_world import GridWorld
from policy_iter import greedy_policy


def value_iter_one_step(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            cur_reward = env.reward(state, action, next_state)
            value = cur_reward + gamma * V[next_state]
            action_values.append(value)
        V[state] = max(action_values)
    
    return V


# def value_iter(V, env, gamma, threshold=0.001, is_render=True):
def value_iter(V, env, gamma, threshold=0.001):
    
    num_iter = 0
    while True:
        # if is_render:
            # fpath = f"{os.path.dirname(os.path.abspath(__file__))}/value_iter_{num_iter}.png"
            # env.render_v(V, fpath=fpath)
        
        old_v = V.copy()
        V = value_iter_one_step(V, env, gamma)
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_v[state])
            delta =  max(delta, t)
        if delta < threshold:
            break
        num_iter += 1
    print(f"{num_iter} iterations using 가치 반복법")
    return V


if __name__ == "__main__":
    env = GridWorld()
    V = defaultdict(lambda : 0)
    gamma = 0.9
    value_iter(V, env, gamma)
    
    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi, fpath = f"{os.path.dirname(os.path.abspath(__file__))}/value_iter_best.png")