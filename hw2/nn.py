import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.head = 0

    def put(self, state, action, reward, next_state, done):
        # Your environment uses these 5 components for every step
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.head] = (state, action, reward, next_state, done)
            self.head = (self.head + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)
    
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # input_dim will be the number of features in your observation
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim) # output_dim = 31 actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, action_size, lr=0.001):
        self.network = QNetwork(input_dim, action_size)
        self.target_network = QNetwork(input_dim, action_size)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.action_size = action_size

    def synchronize(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def act(self, state, epsilon):
        # Epsilon-greedy implementation from the notebook
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.network(state_t).argmax().item()

    def optimize(self, memory, batch_size, gamma):
        if memory.size() < batch_size: return
        
        batch = memory.sample(batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        # Bellman Equation update logic from the notebook
        current_q = self.network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + (gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()


def train(self, env):
    # Initialize Agent based on observation size
    input_dim = env.observation_space.shape[0]
    action_size = 31 # From your get_position_list
    agent = DQNAgent(input_dim, action_size)
    memory = ReplayMemory(10000)
    
    epsilon = 1.0
    gamma = 0.99
    batch_size = 64
    
    # Tutorial recommends multiple episodes for minimum training
    for episode in range(20): 
        observation, info = env.reset()
        done, truncated = False, False
        
        while not done and not truncated:
            action = agent.act(observation, epsilon)
            next_obs, reward, done, truncated, info = env.step(action)
            
            memory.put(observation, action, reward, next_obs, done)
            agent.optimize(memory, batch_size, gamma)
            
            observation = next_obs
            
        # Sync target network and decay epsilon as shown in the notebook
        agent.synchronize()
        epsilon = max(0.01, epsilon * 0.95)
        print(f"Episode {episode} finished.")