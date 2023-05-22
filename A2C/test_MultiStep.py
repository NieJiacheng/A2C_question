from AC import n_step_1_ac
import torch
import gym
import time
import numpy as np
import matplotlib.pyplot as plt

def query_environment(name):
    env = gym.make(name)
    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")

query_environment("FrozenLake-v1")


def train(n, env, gamma, lr):
    print('train start')
    agent = n_step_1_ac.ac_agent(np.random.randint(16), n, 16, 4)
    for i in range(2000):
        print("episode " + str(i))
        start, _ = env.reset()
        agent.s = start
        all_reward = []
        all_state = [agent.s]
        all_action = []
        is_done = False
        for j in range(n):
            action = int(agent.take_action().cpu())
            s_new, reward, done, _, _ = env.step(action)
            all_action.append(action)
            all_reward.append(reward)
            all_state.append(s_new)
            if all_state and s_new == all_state[-1]:
                done = True
            if done:
                if all_reward[-1] == 0:
                    all_reward[-1] = -1
                    is_done = True
                    break
            agent.s = s_new
        print(sum(all_reward))
        agent.update_para(all_state, is_done, all_reward, all_action, lr, gamma)
    print("train end")
    return agent.V, agent.policy

def test(n, env, gamma, lr, game):
    V, policy = train(n, env, gamma, lr)
    env = gym.make(game, render_mode= "human", is_slippery=False)
    t_reward = []
    for i in range(20):
        start, _ = env.reset()
        agent = n_step_1_ac.ac_agent(start, n, 16, 4)
        agent.V = V
        agent.policy = policy
        tot_reward = 0
        while True:
            s_new, reward, done, _, _ = env.step(int(agent.take_action()))
            tot_reward += reward
            if done:
                break
            agent.s = s_new
            t_reward.append(tot_reward)
    print("After train, average reward is " + str(sum(t_reward) / 20))


gamma = 0.99
lr = 1e-6
game = 'FrozenLake-v1'
env = gym.make(game, is_slippery=True)
test(20, env, gamma, lr, game)