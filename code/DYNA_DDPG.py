# dyna_ddpg_inverted_pendulum.py

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
import csv
import matplotlib
import matplotlib.pyplot as plt
from global_variables import EPISODES, DYNA_DDPG_LR, ENV
import time 

start_time = time.time()


# setup matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.model(state)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))

# Learned Dynamics Model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def push(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = map(np.stack, zip(*batch))
        return map(lambda x: torch.tensor(x, dtype=torch.float32), (s, a, r, s_, d))

    def __len__(self):
        return len(self.buffer)

# plot the learning process

def plot_rewards(rewards, show_result=False):
    plt.figure(1)
    rewards_t = np.array(rewards)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('DYNA DDPG Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t)
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = np.convolve(rewards_t, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(rewards_t)), means, label='100-episode average')


    means = np.cumsum(rewards_t) / np.arange(1, len(rewards_t) + 1)
    plt.plot(means, label='Cumulative average')
    
    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Environment Setup
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize Networks and Optimizers
actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
dynamics = DynamicsModel(state_dim, action_dim)

actor_target = Actor(state_dim, action_dim, max_action)
critic_target = Critic(state_dim, action_dim)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optim = optim.Adam(actor.parameters(), lr=DYNA_DDPG_LR)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
dyn_optim = optim.Adam(dynamics.parameters(), lr=1e-3)

replay = ReplayBuffer()

# Hyperparameters
gamma = 0.99
tau = 0.005
batch_size = 128
episodes = EPISODES
planning_steps = 10

def soft_update(target, source):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

rewards_per_episode = []

# Training Loop
for ep in range(episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    total_reward = 0

    for t in range(1000):
        with torch.no_grad():
            action = actor(state).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(state.numpy(), action, reward, next_state, float(done))
        state = torch.tensor(next_state, dtype=torch.float32)
        
        total_reward += reward

        if len(replay) < batch_size:
            continue

        # === Real Experience Update ===
        s, a, r, s_, d = replay.sample(batch_size)
        with torch.no_grad():
            target_q = r + gamma * (1 - d) * critic_target(s_, actor_target(s_)).squeeze()
        current_q = critic(s, a).squeeze()
        critic_loss = nn.MSELoss()(current_q, target_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # === Actor Update ===
        actor_loss = -critic(s, actor(s)).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # === Dynamics Model Update ===
        pred_next = dynamics(s, a)
        dyn_loss = nn.MSELoss()(pred_next, s_)
        dyn_optim.zero_grad()
        dyn_loss.backward()
        dyn_optim.step()

        # === Planning with Learned Model ===
        for _ in range(planning_steps):
            idx = np.random.randint(0, len(replay), size=batch_size)
            s_m, a_m, _, _, _ = zip(*[replay.buffer[i] for i in idx])
            s_m = torch.tensor(np.array(s_m), dtype=torch.float32)
            a_m = torch.tensor(np.array(a_m), dtype=torch.float32)
            with torch.no_grad():
                s_next_m = dynamics(s_m, a_m)
                r_m = -torch.norm(s_next_m, dim=1)
                done_m = torch.zeros_like(r_m)

            target_q_m = r_m + gamma * critic_target(s_next_m, actor_target(s_next_m)).squeeze()
            current_q_m = critic(s_m, a_m).squeeze()
            model_loss = nn.MSELoss()(current_q_m, target_q_m)

            critic_optim.zero_grad()
            model_loss.backward()
            critic_optim.step()

        soft_update(actor_target, actor)
        soft_update(critic_target, critic)
            


        if done:
            break
        
    # plot
    rewards_per_episode.append(total_reward)
    plot_rewards(rewards_per_episode)

env.close()

end_time = time.time()

    
plot_rewards(rewards_per_episode, show_result=True)
plt.ioff()
plt.show()

# export the data in csv file 

with open('DYNA_DDPG_rewards_LR:' + str(DYNA_DDPG_LR) + '.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'reward'])
    for i, reward in enumerate(rewards_per_episode):
        writer.writerow([i, reward])
        
elapsed_time = end_time - start_time
print(f"Experiment completed in {elapsed_time:.2f} seconds.")