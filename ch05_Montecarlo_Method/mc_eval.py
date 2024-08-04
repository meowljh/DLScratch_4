import numpy as np
from collections import defaultdict
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.grid_world import GridWorld
from common.grid_world_renderer import Renderer
# from ch04_Dynamic_Programming.policy_iter import greedy_policy

def greedy_probs_q(Q, state, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    action_probs = {i:0.0 for i in range(action_size)}
    action_probs[max_action] = 1.0
    
    return action_probs

def greedy_probs_q_eps(Q, state, eps:float=1e-6, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    base_prob = eps / action_size
    action_probs = {i : base_prob for i in range(action_size)}
    action_probs[max_action] += (1 - eps)
    return action_probs

# class RandomAgent:
class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        self.alpha = 0.1 # Q-함수 갱신 시의 alpha 값 #
        self.eps = 0.1

        random_actions = {i:0.25 for i in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions) # 각 state마다 특정 action을 할 확률 #
        # self.V = defaultdict(lambda: 0) # 가치 함수 #
        self.Q = defaultdict(lambda: 0) 
        # self.cnts = defaultdict(lambda: 0)
        self.memory = []
    
    def get_action(self, state):
        action_list, action_probs = list(self.pi[state].keys()), list(self.pi[state].values())
        return np.random.choice(action_list, p=action_probs)
    
    def add(self, state, action, reward):
        episode_data = (state, action, reward)
        self.memory.append(episode_data)
    
    def reset(self):
        self.memory.clear()
    
    # def eval(self):
    def update(self):
        G = 0
        for data in reversed(self.memory): # 오래된 순으로 #
            state, action, reward = data
            G = reward + self.gamma * G
            # self.cnts[state] += 1
            ## 증분 방식으로 지금까지의 평균을 계산함 ##
            # prev_state_q = self.Q[state]
            prev_state_q = self.Q[(state, action)]
            ###### 고정 값으로 업데이트 rate를 최근의 업데이트 결과에 더 높은 가중치를 부여할 수 있도록 한다. #####

            # new_state_q = prev_state_q + ((G - prev_state_q) / self.cnts[state])
            new_state_q = prev_state_q + ((G - prev_state_q) * self.alpha)
            self.Q[(state, action)] = new_state_q
            
            # self.pi[state] = greedy_probs_q(self.Q, state)
            self.pi[state] = greedy_probs_q_eps(self.Q, state, eps=self.eps)
            
            # self.V[state] += (G - self.V[state]) / self.cnts[state]
            
    
if __name__ == "__main__":
    env = GridWorld()
    # agent = RandomAgent()
    agent = McAgent()
    renderer = Renderer(reward_map = env.reward_map, goal_state=env.goal_state, wall_state = env.wall_state)
    
    episodes = 1000
    from tqdm import tqdm
    loop = tqdm(range(episodes))
    # for epi in range(episodes):
    for epi in loop:
        state = env.reset() # 초기화된 agent state #
        agent.reset()
        
        while True:
            random_action = agent.get_action(state)
            next_state, reward, is_done = env.step(random_action)
            agent.add(state, random_action, reward)
            
            if is_done: # goal state에 도달 하였을 때 #
                # print("DONE!!")
                # print(f"Policy is: {agent.pi}")
                agent.update()
                break
            state = next_state
    print(agent.Q)
    # env.render_v(agent.V, fpath = f"{os.path.dirname(os.path.abspath(__file__))}/simple_mc.png")
    renderer.render_q(agent.Q, fpath = f"{os.path.dirname(os.path.abspath(__file__))}/simple_mc.png")
    
    