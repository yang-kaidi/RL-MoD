import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Dirichlet

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, env):
        super(Policy, self).__init__()
        self.env = env
        
        # initialize affine transformations for neural net
        self.affine1 = nn.Linear(56+16+1, 128)
        self.affine2 = nn.Linear(128, 64)
        self.affine3 = nn.Linear(64, 64)
        
        # initialize GRU cell
        self.rnn = nn.GRU(input_size=56+8, hidden_size=32, batch_first=True)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, 32))
        
        # actor's layer
        self.action_head = nn.Linear(96, 8)

        # critic's layer
        self.value_head = nn.Linear(96, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x_ext, x_temp):
        """
        forward of both actor and critic
        """
        x_ext = F.relu(self.affine1(x_ext))
        x_ext = F.relu(self.affine2(x_ext))
        x_ext = F.relu(self.affine3(x_ext))
        
        _, x_temp = self.rnn(x_temp, self.h_0)
        
        h = torch.cat((x_ext.view(1, 64), x_temp.view(1, 32)), dim=1)
        
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softplus(self.action_head(h)).reshape(-1) + 1e-20

        # critic: evaluates being in the state s_t
        state_values = self.value_head(h).view(-1,)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values
    
    def get_available_vehicles(self):
        """
        Count total number of available vehicles.
        """
        return np.sum([self.env.acc[region][self.env.time] for region in self.env.region])
    
    def get_desired_distribution(self, action_rl):
        """
        Given a RL action, returns the desired number of vehicles in each area.
        """
        v_d = action_rl*self.get_available_vehicles()
        return list(v_d.numpy())
