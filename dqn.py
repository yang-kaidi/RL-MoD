import numpy as np
import random

from collections import namedtuple


# DQN imports 
import torch
import torch.nn as nn
import torch.optim as optim
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.lin_input_to_hidden = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.lin_hidden_to_hidden = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32, output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.lin_input_to_hidden(x)))
        x = self.relu(self.bn2(self.lin_hidden_to_hidden(x)))
        return self.head(x.view(x.size(0), -1))
    

class DQN_Agent():
    # initialization
    def __init__(self, env, Ma=5):
        # book-keeping variables
        self.L = 1 # number of levels
        self.Ma = Ma # action-space discretization        
        self.env = env # simulated environment
        self.K = 1 # number of time-intervals
        self.region = env.region # list of cells
        self.nS = self.K * (len(self.region)+len(env.demand)) # number of states (+1 represents extra state'lambda')        
        
        self.n_vehicles = env.G.nodes[0]['accInit']*len(env.region) # total number of vehicles
        
        # initialize state/action spaces
        # state_space: list with elements (k, vleft, vright) -> observed distribution of AVs (at time k)
        # action_space: list with elements (vleft, vright) -> desired distribution of AVs 
        self.action_space = self._get_action_space()
        self.nA = len(self.action_space)
        self.policy_net = DQN(input_dim=self.nS, output_dim=self.nA).to(device)
        self.target_net = DQN(input_dim=self.nS, output_dim=self.nA).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
    
    def _get_action_space(self):
        actions = self._get_action_value([],len(self.region),self.Ma-1)
        action_space = np.array(actions)/(self.Ma-1) # enumerate all possible x_i^d values
        return action_space
    
    def _get_action_value(self, partial_action,n,m):
        value = []
        if n >1:        
            for k in range(0,m+1):
                value += self._get_action_value(partial_action+[k],n-1,m-k)
        elif n == 1:
            value += [partial_action + [m]]
        return value       
    
    def decode_state(self, sraw):
        acc, dacc, demand, t = sraw
        return [acc[i][t] for i in acc] + [dacc[i][tt] for i in acc for tt in range(t,t+self.K-1)] + [demand[i,j][tt] for i,j in demand for tt in range(t,t+self.K)]

    def get_desired_distribution(self, action_rl):
        """
        Given a RL action, returns the desired number of vehicles in each area.
        """
        return self.action_space[action_rl]*self.get_available_vehicles()
    
    def get_available_vehicles(self):
        """
        Count total number of available vehicles.
        """
        return np.sum([self.env.acc[region][self.env.time] for region in self.env.region])
