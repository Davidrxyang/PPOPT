import gymnasium as gym 
import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
from global_variables import PT_EPISODES, PPO_PT_LR, PT_ENV

# setup matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()

# hyperparameters 
GAMMA = 0.99
LR_POLICY = PPO_PT_LR
LR_VALUE = 3e-4
EPISODES = PT_EPISODES
POLICY_HIDDEN_LAYER_SIZE = 128 
VALUE_HIDDEN_LAYER_SIZE = 128
CLIP_EPSILON = 0.2
GAE_LAMBDA = 0.95
PPO_EPOCHS = 10

# baseline batch_size
BATCH_SIZE = 64

# for scheduler 
STEP_SIZE = 100
GAMMA_LR = 0.9

# create the environment 
# env = gym.make("Ant-v4", render_mode="human")
# env = gym.make("Ant-v5")
env = gym.make(PT_ENV)

# define state and action spaces 
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# define the policy network 
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, HIDDEN_LAYER_SIZE):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.mean_layer = nn.Linear(HIDDEN_LAYER_SIZE, action_dim)
        self.log_std_layer = nn.Linear(HIDDEN_LAYER_SIZE, action_dim)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.mean_layer.weight, np.sqrt(0.01))
        nn.init.orthogonal_(self.log_std_layer.weight, np.sqrt(0.01))
        
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x) # log standard deviation helps stabilize 
        std = torch.exp(log_std) # convert back 
        return mean, std
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob
    
# define the value network to serve as the baseline 
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, HIDDEN_LAYER_SIZE):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.value_layer = nn.Linear(HIDDEN_LAYER_SIZE, 1) # outputs a single value 
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.value_layer.weight, np.sqrt(1))
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_layer(x)
        return value
    
# compute the generalized advantage estimate
def compute_gae(rewards, values, gamma, lam):
    advantages = []
    gae = 0
    values = values + [0] # terminal value 
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages
    
# function to compute discounted returns 
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# plot the learning process

def plot_rewards(rewards, show_result=False):
    plt.figure(1)
    rewards_t = np.array(rewards)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('PPO Training...')
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

    
# now we are ready, initialize policy, value network, and optimizers for both 

policy = PolicyNetwork(state_dim, action_dim, POLICY_HIDDEN_LAYER_SIZE)
policy_optimizer = optim.Adam(policy.parameters(), lr=LR_POLICY)
policy_scheduler = optim.lr_scheduler.StepLR(policy_optimizer, step_size=STEP_SIZE, gamma=GAMMA_LR)

value_network = ValueNetwork(state_dim, VALUE_HIDDEN_LAYER_SIZE)
value_optimizer = optim.Adam(value_network.parameters(), lr=LR_VALUE)
value_scheduler = optim.lr_scheduler.StepLR(policy_optimizer, step_size=STEP_SIZE, gamma=GAMMA_LR)

# for plotting 
rewards_per_episode = []

# main training loop 
for episode in range(EPISODES):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    values = []
    
    states = []
    actions = []
    
    done = False
    
    total_reward = 0
    
    # generate trajectory
    while not done:
        action, log_prob = policy.select_action(state)
        value = value_network(torch.tensor(state, dtype=torch.float32))
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value.item())
        states.append(state)
        actions.append(action)
        
        state = next_state
        
        done = terminated or truncated
        total_reward += reward
    
    # summarize our experience
    returns = compute_returns(rewards, GAMMA)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # compute advantages using GAE 
    advantages = compute_gae(rewards, values, GAMMA, GAE_LAMBDA)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # convert to tensors (but to arrays first for efficiency)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    
    # PPO update 
    for _ in range(PPO_EPOCHS):
        for i in range(0, len(states), BATCH_SIZE):
            # sample data 
            sampled_states = states[i:i+BATCH_SIZE]
            sampled_actions = actions[i:i+BATCH_SIZE]
            sampled_log_probs = log_probs[i:i+BATCH_SIZE]
            sampled_advantages = advantages[i:i+BATCH_SIZE]
            sampled_returns = returns[i:i+BATCH_SIZE]
            
            # recompute log probabilities
            mean, std = policy(sampled_states)

            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(sampled_actions).sum(dim=1)
            
            # compute ratio
            ratio = torch.exp(new_log_probs - sampled_log_probs.detach())
            
            # compute surrogate loss
            surr1 = ratio * sampled_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * sampled_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # update policy network
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # compute value loss
            value_loss = nn.functional.mse_loss(value_network(sampled_states), sampled_returns.unsqueeze(1))

            # update value network
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
    
    # step the learning rate scheduler 
    policy_scheduler.step()
    value_scheduler.step()
    
    # plot
    rewards_per_episode.append(total_reward)
    plot_rewards(rewards_per_episode)

    
env.close()

# export our pretrained policy networ 
torch.save(policy.state_dict(), 'PPOPT_PT-NET.pth')

    
plot_rewards(rewards_per_episode, show_result=True)
plt.ioff()
plt.show()

# export the data in csv file 

'''
with open('PPOrewards.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'reward'])
    for i, reward in enumerate(rewards_per_episode):
        writer.writerow([i, reward])
'''