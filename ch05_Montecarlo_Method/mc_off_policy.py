from collections import defaultdict
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.grid_world import GridWorld
from common.grid_world_renderer import Renderer

from mc_eval import greedy_probs_q_eps

class McOffPolicyAgent:
    def __init__(self):
        self.alpha = 0.1
        self.eps = 0.1
        self.gamma = 0.9
        self.action_size = 4
        
        random_actions = {i:0.25 for i in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions) # 대상 정책 #
        self.b = defaultdict(lambda: random_actions) # 행동 정책 #
        
        self.Q = defaultdict(lambda: 0) # 현 시점에서 각 state에 대한 행동 가치 함수의 추정값 #
        self.memory = []
    
    def reset(self):
        self.memory.clear()
    
    def add(self, state, action, reward):
        self.memory.append((state, action, reward))
    
    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        return np.random.choice(actions, p=probs)

    def update(self):
        rho = 1 # 행동 정책과 대상 정책간의 분포를 맞추기 위한 가중치 #
        G = 0
        
        for data in reversed(self.memory):
            state, action, reward = data
            G = reward + rho * self.gamma * G
            self.Q[(state, action)] += (G - self.Q[(state, action)]) * self.alpha
            
            rho *= self.pi[state][action] / self.b[state][action]
            
            self.pi[state] = greedy_probs_q_eps(self.Q, state, eps=0)
            self.b[state] = greedy_probs_q_eps(self.Q, state, eps=self.eps)
                        

if __name__ == "__main__":
    agent = McOffPolicyAgent()
    env = GridWorld()
    renderer = Renderer(env.reward_map, env.goal_state, env.wall_state)
    episodes = 1000
    from tqdm import tqdm
    loop = tqdm(range(episodes))
    for epi in loop:
        state = env.reset() # 다시 agent를 초기 상태로 바꾸고 episode를 새로 시작 #
        agent.reset() # clear all the memory, 즉 경험 초기화 #
        
        while True:
            random_action = agent.get_action(state)
            next_state, reward, is_done = env.step(random_action)
            agent.add(state=state, action=random_action, reward=reward)
            
            if is_done:
                agent.update()
                break
            
            state = next_state
        
    
    renderer.render_q(agent.Q,\
        fpath = f"{os.path.dirname(os.path.abspath(__file__))}/off_policy_mc.png")
