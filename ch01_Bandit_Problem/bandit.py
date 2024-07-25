import numpy as np

class Agent:
    '''epsilon 탐욕 정책 사용하여 
    "어떤 slot machine을 선택할 지 결정"
    '''
    def __init__(self, epsilon, action_size:int=10):
        super().__init__()
        self.epsilon = epsilon # 탐색 확률 #
        self.Qs = np.zeros(action_size) # Quality, 즉 action들에 대한 기댓값 array #
        self.ns = np.zeros(action_size)
        
    def update(self, action, reward): # slot machine의 가치 추정 Q(R|A) #
        self.ns[action] += 1 # 무작위 값이긴 함. #
        # 앞서 계산한 증분 구현을 따라서 현재 reward와 이전 action의 quality를 사용해서 현재 action의 quality를 계산 #
        self.Qs[action] += (reward - self.Qs[action]) * (1 / self.ns[action])
    
    def get_action(self): # 행동 선택 - Epsilon greedy policy #
        # epsilon이 탐색할 확률이기 때문에 이보다 random 값이 작을 때만!
        if np.random.rand() < self.epsilon: # epsilon보다 큰 확률이기 때문에 탐색 단계 진행 #
            return np.random.randint(0, len(self.Qs)) # 무작위 action 선택 #
        return np.argmax(self.Qs) # 선택, 제일 가치 추정이 높은 action 선택 #
    

class AlphaAgent(Agent):
    def __init__(self, epsilon, action_size, alpha):
        super().__init__(epsilon, action_size)
        self.alpha = alpha
        self.epilon = epsilon
        self.Qs = np.zeros(action_size)
        
    '''get_action 함수는 Agent 객체와 동일하게 사용'''
    def update(self, action, reward):
        # alpha : 가중치 #
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha
    
class Bandit:
    def __init__(self, arms:int=10):
        super().__init__()
        self.arms = arms # 슬롯 머신의 개수 #
        self.rates = np.random.rand(arms) # 슬롯 머신 각각의 승률을 무작위로 지정 (그러나 한번 설정하면 더 이상 바뀌지 않음) #
    
    def play(self, arm_idx):
        rate = self.rates[arm_idx]
        if rate > np.random.rand():
            return 1 # arm_idx에 해당하는 슬롯머신의 승률보다 현재 이길 확률이 크면 1개의 코인 획득 #
        else:
            return 0
    

# class NotStatBandit:
#     def __init__(self, arms:int=10):
#         super().__init__()
#         self.arms = arms
#         self.rates = np.random.rand(arms)
        
#     def play(self, arm_idx):
#         rate = self.rates[arm_idx] 
#         # 기존 Bandit 객체와 달리 위의 부분만 바뀜 #
#         self.rates += np.random.rand(self.arms) * 0.1 # random noise added to the rates of all slot machine #
#         if rate > np.random.rand():
#             return 1
#         else:
#             return 0


class NonStatBandit(Bandit):
    def __init__(self, arms):
        super().__init__(arms)
        self.arms = arms
        self.rates = np.random.rand(arms)
        self.init_rates = self.rates.copy()
    
    def _reinit(self):
        self.rates = self.init_rates
        
    '''비정상 문제를 위한 bandit
    비정상 문제란 계속 slot machine, 즉 environment의 상태가 바뀌는 것을 의미한다.'''
    def play(self, arm_idx):
        rate = self.rates[arm_idx]
        self.rates += np.random.randn(self.arms) * 0.1
        if rate > np.random.rand():
            return 1
        return 0
    


def _simulate(agent, bandit, num_actions):
    total_reward = 0
    total_rewards = []
    rates = []
    for epoch in range(num_actions):
        selected_action = agent.get_action()
        cur_reward = bandit.play(arm_idx=selected_action)
        agent.update(action=selected_action, reward=cur_reward)
        
        total_reward += cur_reward
        total_rewards.append(total_reward)
        rates.append(total_reward / (epoch + 1))
    
    print(f"Total Reward: {total_reward}")
    return total_rewards, rates