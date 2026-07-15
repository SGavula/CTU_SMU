import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.head = 0

    def put(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.head] = (state, action, reward, next_state, done)
            self.head = (self.head + 1) % self.capacity
    
    def sample(self, batch_size):
        # Returns classic python list [(state, action, reward, next_state, done), (), ...]
        return random.sample(self.memory, batch_size)
    
    def size(self):
        return len(self.memory)
    

# Defining architecture of network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, input_dim, action_size, lr):
        self.network = QNetwork(input_dim, action_size)
        self.target_network = QNetwork(input_dim, action_size)
        # Copy parameters from main network to target network
        self.target_network.load_state_dict(self.network.state_dict())

        # Defining optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.action_size = action_size

    def synchronize(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def act(self, state, epsilon):
        # Implementing epsilon greedy strategy
        if random.random() < epsilon:
            # Return random action
            return random.randint(0, self.action_size - 1)
        else:
            # Convert state in list format to tensor format
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # make interference
            with torch.no_grad():
                return self.network(state_t).argmax().item()
    
    def optimize(self, memory, batch_size, gamma):
        if memory.size() < batch_size: return 

        # DATA TRANSFORMATION
        # Get batch of data from replay memory
        batch = memory.sample(batch_size)
        # Convert data to tensors
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).unsqueeze(1)

        # BELLMAN EQUATION
        # Now we want to get current q value of state, something like gettting q-value from q-table in classic q-learning
        current_q_value = self.network(states).gather(1, actions)
        # Update rule
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            # (1 - dones): This is a clever trick. If the episode is over (done = 1), then 1 - 1 = 0. This wipes out the future value because there is no "next state" after the game ends. If the game is still going, it multiplies by 1 (no change).
            targets = rewards + (gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q_value, targets)

        self.optimizer.zero_grad()
        # Perform backtracking using gradients
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

class TradingAgent:
    def __init__(self):
        # Set parameters
        self.num_of_episodes = 20
        self.epsilon = 1.0
        self.gamma = 0.99
        self.batch_size = 64

        # Create replay memory
        self.memory = ReplayMemory(10000)

        # Create neural network
        self.agent = None
        
        self.update_every = 20

    def reward_function(self, history):
        # TODO feel free to change the reward function ...
        #  This is the default one used in the gym-trading-env library, however, there might be better ones
        #  @see https://gym-trading-env.readthedocs.io/en/latest/customization.html#custom-reward-function
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

    def make_features(self, df):
        # Micro-view
        # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
        # this is percentual change in time
        df["feature_close"] = df["close"].pct_change()
        # Create the feature : close[t] / open[t]
        df["feature_open"] = df["close"] / df["open"]
        # Create the feature : high[t] / close[t]
        df["feature_high"] = df["high"] / df["close"]
        # Create the feature : low[t] / close[t]
        df["feature_low"] = df["low"] / df["close"]
        # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
        df.dropna(inplace=True)
        # the library automatically adds two features - your position and
        return df

    def get_position_list(self):
        # TODO feel free to specify different set of actions
        #  here, the acceptable actions are positions -1.0, -0.9, ..., 2.0
        #  corresponding actions are integers 0, 1, ..., 30
        #  @see https://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#action-space
        #  value 1.2 means all my money I am investing in bitcoin + 20% I will borrow
        #  value -0.9 means I am borrowing bitcoin to sell it and wait if the bitcoin goes down to buy it cheaper
        return [x / 10.0 for x in range(-10, 21)]

    def train(self, env):
        input_dim = env.observation_space.shape[0]
        action_size = 31
        self.agent = DQNAgent(input_dim=input_dim, action_size=action_size,lr= 0.001)

        # Training loop
        for episode in range(self.num_of_episodes):
            if episode % 10 == 0:
                print(f"Episode number: {episode}")

            step = 0
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                # Perform action from neural network
                action = self.agent.act(observation, self.epsilon)
                next_observation, reward, done, truncated, info = env.step(action)

                # Update replay memory
                self.memory.put(observation, action, reward, next_observation, done)
                if step % self.update_every == 0:
                    # Optimize neural network == update Q-table
                    self.agent.optimize(self.memory, self.batch_size, self.gamma)
                    # Update epsilon
                    self.epsilon = max(0.01, self.epsilon * 0.999)
                observation = next_observation
                step += 1

            # Sync target network
            self.agent.synchronize()


    def get_test_position(self, observation):
        # TODO implement the method that will return position for testing
        #  In other words, this method will contain policy used for testing, not training.
        # return 20 if observation[1] > 1.05 else 10  # all in USD ... maps to position 0.0
        action_index = self.agent.act(observation, 0)
        return action_index

    def test(self, env, n_epochs):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        for _ in range(n_epochs):
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                new_position = self.get_test_position(observation)
                observation, reward, done, truncated, info = env.step(new_position)
