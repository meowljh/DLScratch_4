import numpy as np
'''avg.py
- 평균 구하는 코드
- 하나의 slot machine으로 총 n번 play를 했을 때의 행동 가치를 추정하도록 함.
- Q1, Q2, Q3, ..  순서대로 하나씩 증가시키면서 구할 수 있기 떄문에 "증분구현, incremental implementation"이라고 함.
'''
np.random.seed(0) # seed 고정 #

def _quality_calc_compelx(num_plays:int=10):
    '''
    플레이의 수가 늘어날수록 reward의 원소의 수도 늘어나기 떄문에 당연히 메모리와 계산량이 모두 증가하게 될 것.
    '''
    rewards = []

    for n in range(1, num_plays + 1):
        single_reward = np.random.rand() # normal distribution에서 sampling #
        rewards.append(single_reward)
    
        Q = np.sum(rewards) / n
        print(f"Step #{n} Quality: {Q} ")

def _quality_calc_fast(num_plays:int=10):
    '''
    Q_n = Q_(n-1) + 1/n (R_n - Q_(n-1))
    '''
    q_prev = 0
    for n in range(1, num_plays+1):
        single_reward = np.random.rand() # R_n
        q_now = q_prev + ((1/n) * (single_reward - q_prev))
        print(f"Step #{n} Quality: {q_now}")
        q_prev = q_now
        
        
    


