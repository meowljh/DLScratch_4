import numpy as np
from collections import defaultdict
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.grid_world import GridWorld
from policy_eval import policy_eval

'''policy_iter.py
- 정책 반복법
'''

def argmax(d):
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if max_value == value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    new_policy = {}
    for state in env.states():
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            cur_reward = env.reward(state, action, next_state)
            value = cur_reward + gamma * V[next_state]
            action_values[action] = value
            
        # max_action = argmax(action_values)
        max_action = list(action_values.values()).index(max(list(action_values.values())))
        # action_probs = {k : 0 for k in range(4)}
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1.0
        new_policy[state] = action_probs
    
    return new_policy


def policy_iter(env, gamma, threshold=0.001, is_render=False):
    policy = defaultdict(lambda : {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    # policy = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    # policy = {s:0.25 for s in env.states}
    # policy = {}
    # for state in env.states():
    #     policy[state] = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    V = defaultdict(lambda: 0)
    # V = defaultdict(int)
    fig_flag = False
    while True:
        V = policy_eval(policy, V, env, gamma)
        new_policy = greedy_policy(V, env, gamma)

        if is_render:
            fpath = f"{os.path.dirname(os.path.abspath(__file__))}/policy_iter_first.png"
            if fig_flag == False:
                fig_flag = True
                env.render_v(V, policy, fpath=fpath)
            else:
                env.render_v(V, policy, fpath=None)
        
        if new_policy == policy:
            break
        policy = new_policy
        
    fpath = f"{os.path.dirname(os.path.abspath(__file__))}/policy_iter_best.png"
    env.render_v(V, new_policy, fpath=fpath)

    return new_policy


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    policy_iter(env, gamma, is_render=True)
    
