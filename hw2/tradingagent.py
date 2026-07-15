import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(np.array(dones)))
    
    def __len__(self):
        return len(self.buffer)


class TradingAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.position_list = self.get_position_list()
        self.n_actions = len(self.position_list)
        self.n_features = None  # Will be set during training
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.target_update = 100
        self.steps = 0
        self.update_every = 20
        
    def reward_function(self, history):
        # Improved reward function that considers risk-adjusted returns
        portfolio_vals = history["portfolio_valuation", -3:]
        if len(portfolio_vals) >= 2:
            current = portfolio_vals[-1]
            previous = portfolio_vals[-2]
            
            # Log return (scale-invariant)
            log_return = np.log(current / previous) if previous > 0 else 0
            
            # Add small penalty for large position changes to encourage stability
            if len(history["position", -3:]) >= 2:
                position_change = abs(history["position", -1] - history["position", -2])
                stability_penalty = 0.01 * position_change
                return log_return - stability_penalty
            
            return log_return
        return 0.0

    def make_features(self, df):
        """
        Create technical indicators as features for the trading agent.
        IMPORTANT: No look-ahead bias - all features use only current and past data.
        """
        # Price-based features
        df["feature_close_pct"] = df["close"].pct_change()
        df["feature_high_low_ratio"] = (df["high"] / df["low"]) - 1
        df["feature_close_open_ratio"] = df["close"] / df["open"] - 1
        
        # Moving averages
        df["feature_sma_5"] = df["close"].rolling(window=5).mean() / df["close"]
        df["feature_sma_10"] = df["close"].rolling(window=10).mean() / df["close"]
        df["feature_sma_20"] = df["close"].rolling(window=20).mean() / df["close"]
        
        # Price relative to moving averages
        df["feature_price_to_sma_5"] = df["close"] / df["feature_sma_5"] - 1
        df["feature_price_to_sma_20"] = df["close"] / df["feature_sma_20"] - 1
        
        # Volatility features
        df["feature_volatility_5"] = df["close"].pct_change().rolling(window=5).std()
        df["feature_volatility_10"] = df["close"].pct_change().rolling(window=10).std()
        
        # Volume features
        df["feature_volume_sma"] = df["volume"] / df["volume"].rolling(window=24).mean()
        df["feature_volume_max"] = df["volume"] / df["volume"].rolling(24).max()
        
        # Bollinger Bands
        sma_20 = df["close"].rolling(window=20).mean()
        std_20 = df["close"].rolling(window=20).std()
        df["feature_bb_upper"] = (sma_20 + 2 * std_20) / df["close"] - 1
        df["feature_bb_lower"] = (sma_20 - 2 * std_20) / df["close"] - 1
        df["feature_bb_position"] = (df["close"] - sma_20) / (2 * std_20)  # Position within bands
        
        # Momentum features
        df["feature_momentum_3"] = df["close"].pct_change(periods=3)
        df["feature_momentum_5"] = df["close"].pct_change(periods=5)
        df["feature_momentum_10"] = df["close"].pct_change(periods=10)
        
        # RSI-like feature
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["feature_rsi"] = 100 - (100 / (1 + rs))
        
        # MACD-like features
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["feature_macd"] = ema_12 - ema_26
        
        # Price extremes relative to recent window
        df["feature_high_5"] = df["high"].rolling(window=5).max() / df["close"] - 1
        df["feature_low_5"] = df["low"].rolling(window=5).min() / df["close"] - 1
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df

    def get_position_list(self):
        # Positions from -1.0 (short) to 2.0 (long with leverage)
        # More granular positions for better control
        positions = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     1.2, 1.5, 2.0]
        return positions

    def train(self, env):
        """
        Train the DQN agent on the trading environment.
        """
        # Get the number of features from the environment
        obs_sample, _ = env.reset()
        self.n_features = len(obs_sample)
        
        # Initialize networks
        self.policy_net = QNetwork(self.n_features, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_features, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        n_episodes = 15  # Number of available training datasets
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            state = self._process_observation(obs)
            done = False
            truncated = False
            total_reward = 0
            
            while not done and not truncated:
                # Select action using epsilon-greedy policy
                action = self._select_action(state)
                
                # Take action in environment
                next_obs, reward, done, truncated, info = env.step(action)
                next_state = self._process_observation(next_obs)
                
                # Store transition in replay buffer
                self.memory.push(state, action, reward, next_state, done or truncated)
                
                # Update state
                state = next_state
                total_reward += reward
                self.steps += 1
                
                # Train the network
                if len(self.memory) > self.batch_size:
                    if self.steps % self.update_every == 0:
                        self._train_step()
                
                # Update target network periodically
                if self.steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.4f}, Epsilon: {self.epsilon:.4f}")
        
        # Final target network update
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _process_observation(self, obs):
        """
        Process observation to handle any NaN or infinite values.
        """
        if isinstance(obs, np.ndarray):
            # Replace NaN and infinite values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            return obs.astype(np.float32)
        return np.array(obs, dtype=np.float32)

    def _select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def _train_step(self):
        """
        Perform one training step on the DQN.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values with target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def get_test_position(self, observation):
        """
        Return the best action according to the trained policy for testing.
        """
        if self.policy_net is None:
            # If not trained, return neutral position
            return self.position_list.index(0.0)
        
        state = self._process_observation(observation)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
        
        return action

    def test(self, env, n_epochs):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        for _ in range(n_epochs):
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                new_position = self.get_test_position(observation)
                observation, reward, done, truncated, info = env.step(new_position)