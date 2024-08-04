import numpy as np

x = np.array([1, 2, 3]) # 확률 변수 #
pi = np.array([0.1, 0.1, 0.8]) # 확률 분포 #

e = np.sum(x * pi)

n = 100
samples = []

for _ in range(n):
    s = np.random.choice(x, p=pi) # 정책 pi의 확률 값을 기반으로 sampling #
    samples.append(s)


print(f"진째 기댓값: {e}")
print(f"몬테카르로 샘플링 \n 평균: {np.mean(samples)}    분산: {np.var(samples)}")
# print(np.std(samples))