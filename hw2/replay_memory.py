import random

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
    

