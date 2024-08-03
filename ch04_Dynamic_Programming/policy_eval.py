import numpy as np
from collections import defaultdict
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.grid_world import GridWorld

def eval_one_step(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_probs = pi[state]
        new_v = 0 ## 현재 state의 새로운 가치로 update ##
        ## 목표 state를 제외하고 나머지 state에서 수행 가능한 action들에 대해서 현재 state를 기반으로 action을 취했을 때의 다음 state를 구할 수 있음. ##
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state=state, action=action, next_state=next_state)
            new_v += action_prob * (r + gamma * V[next_state])
            # print(f"Reward: {r}       newV: {new_v}        ActionProb: {action_prob}")
        
        V[state] = new_v
    
    
    return V


def policy_eval(pi, V, env, gamma, threshold:float=0.001):
    '''이 단계까지는 정책 평가를 수행한 것에 불과하다.'''
    while True:
        old_v = V.copy()
        V = eval_one_step(pi, V, env, gamma)
        delta = 0
        for state in V.keys():
            ## 이 부분에서 DP를 사용하였다고 생각할 수 있음. old_v, 즉 이전의 가치 함수를 전부 바로 업데이트를 하고,
            # 이전에 저장해 놨던 값으로부터 "update된 가치의 차이"를 통해서 policy evaluation을 멈출지 결정하게 된다. 
            t = abs(V[state] - old_v[state])
            if delta < t:
                delta = t
        if delta < threshold: # 임계값보다 이전 value function에서의 값이 변화가 없는 상황이라면 break #
            break
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25}) # 정책 #
    V = defaultdict(lambda: 0) # 가치 함수 #
    V = policy_eval(pi, V, env, gamma, 0.0001)
    fpath = f"{os.path.dirname(os.path.abspath(__file__))}/policy_eval_1.png"
    
    env.render_v(V, policy=pi, print_value=True, fpath=fpath)
    
    # env.fig.savefig("policy_eval_1.png")

