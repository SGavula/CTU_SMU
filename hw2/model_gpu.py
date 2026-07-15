import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# This will use the GPU (cuda) if you have an NVIDIA card, otherwise it stays on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Defining architecture of network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.fc1 = nn.Linear(input_dim, 1024)
        # self.fc2 = nn.Linear(1024, 516)
        # self.fc3 = nn.Linear(516, output_dim)
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, input_dim, action_size, lr):
        self.network = QNetwork(input_dim, action_size).to(device)
        self.target_network = QNetwork(input_dim, action_size).to(device)
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
            state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            # make interference
            with torch.no_grad():
                # .item() function convert the output from network and argmax which is tensor to integer
                return self.network(state_t).argmax(1).item()
    
    def optimize(self, memory, batch_size, gamma):
        if memory.size() < batch_size: return 

        # DATA TRANSFORMATION
        # Get batch of data from replay memory
        batch = memory.sample(batch_size)
        # Convert data to tensors
        # We must unsqueeze (add one dimension) actions, rewards, dones (1D lists) because states are 2D matrix
        # We need all data have the same dimension
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)

        # BELLMAN EQUATION
        # Now we want to get current q value of state, something like gettting q-value from q-table in classic q-learning
        # gather(1, actions) --> neural network predicted 31 values (for each action) but we want only value for action that we took, from q-table we get Q[current_state][action], and this logic do gather(1, actions)
        # gather(1, actions) --> 1 is there because we tell pytorch to go for one row through each column and find the action, if there will be 0 then pytorch go through columns
        current_q_value = self.network(states).gather(1, actions)
        # Update rule
        with torch.no_grad():
            # .max(1): In PyTorch, .max(1) returns two things: the values and the indices.
            # [0]: We only want the values (the best predicted future score).
            # .unsqueeze(1): Just like before, we turn this list of 64 best scores into a column so we can do math with it.
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            # (1 - dones): This is a clever trick. If the episode is over (done = 1), then 1 - 1 = 0. This wipes out the future value because there is no "next state" after the game ends. If the game is still going, it multiplies by 1 (no change).
            targets = rewards + (gamma * max_next_q * (1 - dones))

        # alpha(learning rate) and the subtraction - Q(s, a) are missing from that line.
        # The Subtraction: This happens in the very next line of your code: loss = F.mse_loss(current_q, targets). 
        # Mean Squared Error (MSE) literally subtracts them and squares the result.The alpha (Alpha): This is handled by the Optimizer (optimizer.step()). 
        # When you created the Adam optimizer, you gave it a lr (Learning Rate). That is your alpha
        loss = self.criterion(current_q_value, targets)

        # In PyTorch, gradients are cumulative. 
        # This means if you don't explicitly clear them, every time you call loss.backward(), the new gradients will be added to the old ones from the previous trade.
        self.optimizer.zero_grad()
        # Perform backtracking using gradients
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
