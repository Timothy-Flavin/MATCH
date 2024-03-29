import numpy as np
import torch

# Format following Machine learning with phil memory buffer 
class memory_buffer():
  def __init__(self, num_steps, action_size, state_size):
    self.mem_size = num_steps
    self.states = np.zeros((num_steps,state_size),dtype=np.float32)
    self.states_ = np.zeros((num_steps,state_size),dtype=np.float32)
    self.actions = np.zeros((num_steps,action_size),dtype=np.float32)
    self.rewards = np.zeros((num_steps),dtype=np.float32)
    self.done = np.zeros((num_steps),dtype=np.float32)
    self.idx = 0
  
  def save_transition(self, state, action, reward, state_, done):
    idx = self.idx % self.mem_size
    self.states[idx] = state
    self.states_[idx] = state_
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.idx+=1 
    self.done[idx] = done

  def sample_memory(self, batch_size):
    size = min(self.idx, batch_size)
    idx = np.random.choice(min(self.idx,self.mem_size), size, replace=False)
    return self.states[idx],self.actions[idx],self.rewards[idx],self.states_[idx], self.done[idx]
  def sample_memory_to_cuda(self, batch_size, device):
    size = min(self.idx, batch_size)
    idx = np.random.choice(min(self.idx,self.mem_size), size, replace=False)
    return torch.from_numpy(self.states[idx]).to(device), \
           torch.from_numpy(self.actions[idx]).to(device), \
           torch.from_numpy(self.rewards[idx]).to(device), \
           torch.from_numpy(self.states_[idx]).to(device), \
           torch.from_numpy(self.done[idx]).to(device) 
  
  