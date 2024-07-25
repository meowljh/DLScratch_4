import numpy as np
import matplotlib.pyplot as plt


from bandit import Agent, Bandit, NonStatBandit, AlphaAgent, _simulate

'''
- bandit.py에서 정의한 Agent, Bandit 클래스를 사용해서 simulation을 해보자.
- 행동을 1000번 취했을 때 보상을 얼마나 얻는지 알아보고자 한다.
- epsilon이 0.1에서 커짐에 따라서 승률의 최댓겂이 점점 줄어들게 되는데, 이를 통해서 탐색을 너무 많이 하게 되면
최적의 slot machine을 선택하는 경우가 적어지기 때문으로 해석할 수 있다.
'''

NUM_ACTION=1000
NUM_ARMS=10
EPS=0.1 # epsilon의 확률로 무작위 선택, 0-1 사이의 확률 값 #
ALPHA = 0.8

bandit = Bandit(arms=NUM_ARMS) # play 객체 #
agent = Agent(epsilon=EPS, action_size=NUM_ARMS) # epsilon-greedy policy로 다음 action 정의 #
# nonstat_bandit = NonStatBandit(arms=NUM_ARMS)

alpha_agent = AlphaAgent(epsilon=EPS, action_size=NUM_ARMS, alpha=ALPHA)

'''
비정상 문제, 즉 NonStationary Bandit 상황에서 
각각의 Reward에 대한 가중치를 고정하여 alpha라는 값을 사용할지
무작위로 설정할지에 대한 test
'''
reward1, rate1 = _simulate(agent, NonStatBandit(NUM_ARMS), num_actions=NUM_ACTION)
# nonstat_bandit._reinit()
reward2, rate2 = _simulate(alpha_agent, NonStatBandit(NUM_ARMS), num_actions=NUM_ACTION)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
X = np.arange(1, NUM_ACTION+1)

ax[0].set_ylabel("Total Reward");ax[0].set_xlabel("Steps")
ax[0].plot(X, reward1, 'r', label='sample average');ax[0].plot(X, reward2, 'b', label='alpha const update')

ax[1].set_ylabel("Rates");ax[1].set_xlabel("Steps")
ax[1].plot(X, rate1, 'r', label='sample average');ax[1].plot(X, rate2, 'b', label='alpha const update')

plt.legend()
plt.show()
fig.savefig("run_2_비정상.png")

# print(f"Random bandit win-rates: {bandit.rates}")

# total_reward = 0
# total_rewards = []
# rates = []

# for epoch in range(NUM_ACTION):
#     selected_action = agent.get_action() # index #
#     cur_reward = bandit.play(arm_idx=selected_action)
#     # agent를 update하는 과정이 보상을 통해서 학습을 하는 것과 동일하다고 볼 수도 있다. #
#     agent.update(action=selected_action, reward=cur_reward) # 해당 action을 했을 때의 보상 업데이트 #


#     total_reward += cur_reward
#     total_rewards.append(total_reward) # 현재까지의 보상합 저장 #
#     rates.append(total_reward / (epoch + 1)) # 현재까지의 승률 저장 #
    

# print(f"Total Reward: {total_reward}")

# fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# X = np.arange(1, NUM_ACTION+1)

# ax[0].set_ylabel("Total Reward")
# ax[0].set_xlabel("Steps")
# ax[0].plot(X, total_rewards)

# ax[1].set_ylabel("Rates")
# ax[1].set_xlabel("Steps")
# ax[1].plot(X, rates)


# plt.grid()
# plt.show()

# fig.savefig("run_1.png")
