import numpy as np
import matplotlib.pyplot as plt
"""
这个实验的设计如下：

1.创建一个修改版的10臂赌博机测试平台,其中所有的真实行动价值都相等,并且以独立的随机游走方式变化。
    在每个步骤中,将所有的真实行动价值都增加一个均值为0、标准差为0.01的正态分布随机增量。
2.实现两种行动价值方法:
    a. 样本平均方法: 使用样本平均方法逐步更新行动价值,其中行动价值的估计值是该行动获得的所有奖励的平均值。 
    b. 常数步长方法: 使用常数步长参数(α=0.1)逐步更新行动价值。
3.运行每种方法进行长时间的实验(例如10000步),使用ε-greedy策略(ε=0.1)进行探索。
4.记录每个步骤获得的奖励,对于每种方法分别计算平均奖励。
5.绘制每种方法的平均奖励曲线,并进行比较。

该代码实现了一个简单的实验,其中包含了一个修改版的10臂赌博机测试平台、样本平均方法和常数步长方法。
在该实验中,所有的真实行动价值都相等,并且以独立的随机游走方式变化。
我们运行了2000次实验,每次实验进行10000步。最终,我们绘制了样本平均方法和常数步长方法的平均奖励曲线。

在这个实验中,我们期望看到样本平均方法的表现较差,因为它会给过去的奖励分配相等的权重,难以适应真实行动价值的变化。
相比之下,常数步长方法应该表现更好,因为它更加关注最近的奖励,能够更快地适应真实行动价值的变化。

"""
# Define the modified 10-armed testbed
class Testbed:
    def __init__(self, n_arms, q_star_mean, q_star_std, reward_std):
        self.n_arms = n_arms
        self.q_star_mean = q_star_mean
        self.q_star_std = q_star_std
        self.reward_std = reward_std
        self.q_star = np.zeros(n_arms)
        self.reset()
        
    def reset(self):
        self.q_star = np.random.normal(self.q_star_mean, self.q_star_std, self.n_arms)
        self.action_count = np.zeros(self.n_arms)
        
    def step(self, action):
        reward = np.random.normal(self.q_star[action], self.reward_std)
        self.action_count[action] += 1
        self.q_star += np.random.normal(0, 0.01, self.n_arms)
        return reward

# Define the sample-average method
class SampleAverage:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.reset()
        
    def reset(self):
        self.action_count = np.zeros(self.n_arms)
        self.action_value = np.zeros(self.n_arms)
        
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_arms)
        else:
            return np.argmax(self.action_value)
        
    def update(self, action, reward):
        self.action_count[action] += 1
        self.action_value[action] += (reward - self.action_value[action]) / self.action_count[action]

# Define the constant step-size method
class ConstantStepSize:
    def __init__(self, n_arms, alpha, epsilon):
        self.n_arms = n_arms
        self.alpha = alpha
        self.epsilon = epsilon
        self.reset()
        
    def reset(self):
        self.action_value = np.zeros(self.n_arms)
        
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_arms)
        else:
            return np.argmax(self.action_value)
        
    def update(self, action, reward):
        self.action_value[action] += self.alpha * (reward - self.action_value[action])

# Run the experiment
n_runs = 2000
n_steps = 10000
n_arms = 10
q_star_mean = 0
q_star_std = 1
reward_std = 1
epsilon = 0.1
alpha = 0.1

testbed = Testbed(n_arms, q_star_mean, q_star_std, reward_std)
sample_average = SampleAverage(n_arms, epsilon)
constant_step_size = ConstantStepSize(n_arms, alpha, epsilon)

sample_average_rewards = np.zeros(n_steps)
constant_step_size_rewards = np.zeros(n_steps)

for i in range(n_runs):
    testbed.reset()
    sample_average.reset()
    constant_step_size.reset()
    for j in range(n_steps):
        print(f'==step: {i} {j}')
        action_sa = sample_average.select_action()
        action_cs = constant_step_size.select_action()
        reward_sa = testbed.step(action_sa)
        reward_cs = testbed.step(action_cs)
        sample_average.update(action_sa, reward_sa)
        constant_step_size.update(action_cs, reward_cs)
        sample_average_rewards[j] += reward_sa
        constant_step_size_rewards[j] += reward_cs

sample_average_rewards /= n_runs
constant_step_size_rewards /= n_runs

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sample_average_rewards, label='Sample Average')
plt.plot(constant_step_size_rewards, label='Constant Step Size')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.show()