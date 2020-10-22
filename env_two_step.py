# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:09:46 2020
@author: yangk
"""
from collections import defaultdict
import numpy as np
import subprocess
import os
import networkx as nx
from util import mat2str
from copy import deepcopy

class CascadedQLearning():
    # initialization
    def __init__(self, env):
        # book-keeping variables
        self.L = 3 # number of levels
        self.Ms = 5 # state-space discretization
        self.Ma = 5 # action-space discretization
        self.K = 2 # number of time-intervals
        self.nA = self.Ma # number of actions
        self.nS = self.K * (self.Ms + 1) # number of states (+1 represents extra state'lambda')
        self.env = env # simulated environment
        self.region = env.region # list of cells
        self.n_vehicles = env.G.nodes[0]['accInit']*len(env.region) # total number of vehicles
        self.cascaded_regions = self._get_cascaded_regions() # regions for each level of cascade, key: l - 'level', r: - 'region'
        
        # initialize state/action spaces
        # state_space: list with elements (k, vleft, vright) -> observed distribution of AVs (at time k)
        # action_space: list with elements (vleft, vright) -> desired distribution of AVs 
        self.state_space, self.action_space = self._get_state_action_space()
        self.Q = self._get_q_tables() # initialize Q-tables, key: l - 'level', n: - 'node'
        self.nodes = list(self.Q.keys())
    
    def _get_cascaded_regions(self):
        cascaded_regions = dict()
        for l in range(1, self.L+1):
            n_l = 2**(l) # number of nodes in level 'l'
            splits = np.array_split(self.region, n_l)
            for node, split in enumerate(splits):
                cascaded_regions[l-1, node+1] = split
        return cascaded_regions
    
    def _get_state_action_space(self):
        vs_state = np.linspace(0, 1, self.Ms) # enumerate all possible v_i values 
        vs_action = np.linspace(0, 1, self.Ma) # enumerate all possible x_i^d values
        state_space, action_space = [], []
        for vleft in vs_state:
            for k in range(self.K):
                state_space.append((k, vleft, 1-vleft))
            action_space.append((vleft, 1-vleft))
        state_space.append('lambda')
        return state_space, action_space
    
    def _get_q_tables(self):
        Q_set = dict()
        for l in range(1, self.L+1):
            n_l = 2**(l-1) # number of nodes in level 'l'
            for n in range(n_l):
                Q_set[l-1, n+1] = np.zeros((self.nS, self.nA)) # Q_i \in R^{self.nS, self.nA}
        return Q_set
    
    def decode_state(self, raw_s, t):
        """
        Decodes raw environment state into state representation for learning.
        
        Parameters
        ----------
        raw_s : array_like
                Availability of idle vehicles as in env.acc.
        t : int
            Current environemt time.
        Returns
        -------
        s : array_like
            Idle vehicle distribution for every node.
        """
        k = t%self.K
        s = []
        list_regions = list(self.cascaded_regions.keys())
        for i in range(0, len(list_regions)-1, 2):
            x_left, x_right = 0, 0
            for cell in self.cascaded_regions[list_regions[i]]:
                x_left += raw_s[cell][t]
            for cell in self.cascaded_regions[list_regions[i+1]]:
                x_right += raw_s[cell][t]
            if x_left==x_right==0.:
#                 print(f"lambda, xl {x_left}, xr {x_right}")
                s.append("lambda")
            else:
                v_left = min(np.linspace(0, 1, self.Ms), key=lambda x:abs(x - (x_left / (x_left + x_right))))
                v_right = 1 - v_left
                s.append((k, v_left, v_right))
        return s
    
    def encode_state(self, s):
        """
        Encodes state representation into index for state-space list.
        
        Parameters
        ----------
        s : list
            State representation.
        Returns
        -------
        idx : int
            Index value for s in state_space
        """
        idx = self.state_space.index(s)
        return idx
    
    def get_desired_distribution(self, action_rl):
        """
        Given a RL action, returns the desired number of vehicles in each area.
        """
        v_d = [[] for _ in range(2**self.L)]
        for i, region in enumerate(list(self.cascaded_regions.keys())):
            for cell in self.cascaded_regions[region]:
                v_d[cell].append(self.action_space[action_rl[i//2]][i%2])
        return np.prod(v_d, axis=1)*self.get_available_vehicles()
    
    def get_available_vehicles(self):
        """
        Count total number of available vehicles.
        """
        return np.sum([self.env.acc[region][self.env.time] for region in self.env.region])
    
    def policy(self, obs, params, train=True, isMatching=False, CPLEXPATH=None, res_path=None):
        """
        Apply the current policy.
        
        Parameters
        ----------
        params : dict
            Training settings for cascaded learning.
        
        Returns
        -------
        action: list
            List of action indexes for each Q-table in the cascade. 
            
        # STEP 1 - Select desired distribution of idle vehicles through RL
        # 1.1 Pick actions for all Q tables with the following logic:
        # (i) If Q table under training: either epsilon greedy or max_Q
        # (ii) for all untrained Q tables select default action (.5, .5)
        # (iii) for all trained Q tables select argmax action argmax Q(:, a)
        """
        num_nodes = len(self.nodes)
        action_rl = [] # RL action for all nodes
        if train: # Allow for epsilon-greedy exploration during training
            training_round_len = params["training_round_len"]
            epsilon = params["epsilon"]
            k = params["k"]
            default_action = params["default_action"]
            idx = (k//training_round_len)%num_nodes # Q table index (initially, top-most node)
            for i in range(num_nodes):
                state_i = self.encode_state(self.decode_state(obs[0], obs[1])[i])
                if i==idx:
                    if np.random.rand() < epsilon: # Epsilon-greedy policy for current policy
                        action_rl.append(np.random.randint(low=0, high=self.nA))
                    else: # Apply current policy
                        action_rl.append(np.argmax(self.Q[self.nodes[i]][state_i, :]))
                else: # for all other nodes, select either default action or take argmax Q(:,a) 
                    if (k//(training_round_len*num_nodes) < 1) and (k//100 < i):
                        action_rl.append(default_action)
                    else:
                        action_rl.append(np.argmax(self.Q[self.nodes[i]][state_i, :]))
        else: # At test time, simply use the learnt argmax policy
            for i in range(num_nodes):
                state_i = self.encode_state(self.decode_state(obs[0], obs[1])[i])
                action_rl.append(np.argmax(self.Q[self.nodes[i]][state_i, :]))
        # 1.2 get actual vehicle distributions vi (i.e. (x1*x2*..*xn)*num_vehicles)
        v_d = self.get_desired_distribution(action_rl)

        # 1.3 Solve ILP - Minimal Distance Problem 
        # 1.3.1 collect inputs and build .dat file
        t = self.env.time
        accTuple = [(n,int(self.env.acc[n][t])) for n in self.env.acc]
        accRLTuple = [(n, int(v_d_n)) for n, v_d_n in enumerate(v_d)]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        OPTPath = os.getcwd().replace('\\','/')+'/OPT/CQL/'+res_path
        if not os.path.exists(OPTPath):
            os.makedirs(OPTPath)
        datafile = OPTPath + f'data_{t}.dat'
        resfile = OPTPath + f'res_{t}.dat'
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')

        # 2. execute .mod file and write result on file
        modfile = modPath+'minRebDistRebOnly.mod'
        if CPLEXPATH is None:
            CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file =  OPTPath + f'out_{t}.dat'
        with open(out_file,'w') as output_f:
            subprocess.check_call([CPLEXPATH+"oplrun", modfile, datafile], stdout=output_f, env=my_env)
        output_f.close()

        # 3. collect results from file
        flow = defaultdict(float)
        with open(resfile,'r', encoding="utf8") as file:
            for row in file:
                item = row.strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,j,f = v.split(',')
                        flow[int(i),int(j)] = float(f)
        action = [flow[i,j] for i,j in self.env.edges]
        return action, action_rl

class AMoD:
    # initialization
    def __init__(self, scenario, beta=0.2): # updated to take scenario and beta (cost for rebalancing) as input
        self.scenario = deepcopy(scenario) # I changed it to deep copy so that the scenario input is not modified by env 
        self.G = scenario.G # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.time = 0 # current time
        self.T = scenario.T # planning time
        self.tf = scenario.tf # final time
        self.demand = defaultdict(dict) # demand
        self.price = defaultdict(dict) # price
        for i,j,t,d,p in scenario.tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            self.price[i,j][t] = p
        self.acc = defaultdict(dict) # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(dict) # number of vehicles arriving at each region, key: i - region, t - time
        self.rebFlow = defaultdict(dict) # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict) # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = [] # set of rebalancing edges
        self.nregion = len(scenario.G) # number of regions
        for i in self.G:
            self.edges.append((i,i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
                self.region = list(self.G) # set of regions
        self.nedge = [len(self.G.out_edges(n))+1 for n in self.region] # number of edges leaving each region        
        for i,j in self.G.edges:
            self.rebFlow[i,j] = defaultdict(float)
            self.paxFlow[i,j] = defaultdict(float)            
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)   
        self.beta = beta
        t = self.time
        self.servedDemand = defaultdict(float)
        for i,j in self.demand:
            self.servedDemand[i,j] = defaultdict(float)
        
        self.N = len(self.region) # total number of cells
        
        # add the initialization of info here
        self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost', 'operating_cost'], 0)
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand        
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def matching(self, CPLEXPATH=None, PATH=''):
        t = self.time
        demandAttr = [(i,j,self.demand[i,j][t], self.price[i,j][t]) for i,j in self.demand \
                      if self.demand[i,j][t]>1e-3]
        accTuple = [(n,self.acc[n][t+1]) for n in self.acc]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        matchingPath = os.getcwd().replace('\\','/')+'/matching/'+PATH
        if not os.path.exists(matchingPath):
            os.makedirs(matchingPath)
        datafile = matchingPath + 'data_{}.dat'.format(t)
        resfile = matchingPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
        modfile = modPath+'matching.mod'
        if CPLEXPATH is None:
            CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file =  matchingPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
        output_f.close()
        flow = defaultdict(float)
        with open(resfile,'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)',')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,j,f = v.split(',')
                        flow[int(i),int(j)] = float(f)
        paxAction = [flow[i,j] if (i,j) in flow else 0 for i,j in self.edges]
        return paxAction

    # pax step
    def pax_step(self, paxAction=None, CPLEXPATH=None, PATH=''):
        t = self.time
        for i in self.region:
            self.acc[i][t+1] = self.acc[i][t]
        self.info['served_demand'] = 0 # initialize served demand
        self.info["operating_cost"] = 0 # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        if paxAction is None:  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
            paxAction = self.matching(CPLEXPATH=CPLEXPATH, PATH=PATH)
        self.paxAction = paxAction
        # serving passengers
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            self.paxAction[k] = min(self.acc[i][t+1], paxAction[k])            
            self.servedDemand[i,j][t] = self.paxAction[k]
            self.paxFlow[i,j][t+self.G.edges[i,j]['time']] = self.paxAction[k]
            self.info["operating_cost"] += self.G.edges[i,j]['time']*self.beta*self.paxAction[k]
            self.acc[i][t+1] -= self.paxAction[k]
            self.info['served_demand'] += self.servedDemand[i,j][t]            
            self.dacc[j][t+self.G.edges[i,j]['time']] += self.paxFlow[i,j][t+self.G.edges[i,j]['time']]
            self.reward += self.paxAction[k]*(self.price[i,j][t] - self.G.edges[i,j]['time']*self.beta)            
            self.info['revenue'] += self.paxAction[k]*(self.price[i,j][t] - self.G.edges[i,j]['time']*self.beta)  
        
        self.obs = (self.acc, self.time, self.dacc, self.demand) # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False # if passenger matching is executed first
        return self.obs, max(0,self.reward), done, self.info
    
    # reb step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0 # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.rebAction = rebAction      
        # rebalancing
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.rebAction[k] = min(self.acc[i][t+1], rebAction[k]) 
            self.rebFlow[i,j][t+self.G.edges[i,j]['time']] = self.rebAction[k]     
            self.acc[i][t+1] -= self.rebAction[k] 
            self.dacc[j][t+self.G.edges[i,j]['time']] += self.rebFlow[i,j][t+self.G.edges[i,j]['time']]   
            self.info['rebalancing_cost'] += self.G.edges[i,j]['time']*self.beta*self.rebAction[k]
            self.info["operating_cost"] += self.G.edges[i,j]['time']*self.beta*self.rebAction[k]
            self.reward -= self.G.edges[i,j]['time']*self.beta*self.rebAction[k]
            self.info['revenue'] -= self.G.edges[i,j]['time']*self.beta*self.rebAction[k]
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing        
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            self.acc[j][t+1] += self.rebFlow[i,j][t]
            self.acc[j][t+1] += self.paxFlow[i,j][t] # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion
            
        self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.demand) # use self.time to index the next time step
        
        done = (self.tf == t+1) # if the episode is completed
        return self.obs, self.reward, done, self.info
            
            
    # # simulation step
    # def step(self, action, isMatching = False, CPLEXPATH=None, PATH=None): 
    #     # rebAction/paxAction: np.array, where the kth element represents the number of vehicles going from region i to region j, (i,j) = self.edges[k]
    #     # paxAction is None if not provided (default matching algorithm will be used)
    #     self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost', 'operating_cost'], 0)
    #     t = self.time
    #     reward = 0
        
    #     # converting action - if isMatching is True, action is only for rebalancing, otherwise action includes both rebalancing and pax actions
    #     self.action = action             
    #     rebAction = np.zeros(len(action))
    #     paxAction = np.zeros(len(action))
    #     if not isMatching:
    #         for k in range(len(self.edges)):
    #             i,j = self.edges[k] 
    #             if (i,j) not in self.G.edges:
    #                 continue
    #             paxAction[k] = min([self.demand[i,j][t], action[k]])
    #             rebAction[k] = action[k] - paxAction[k] 
    #     else:
    #         rebAction = action
            
    #     for i in self.region:
    #         self.acc[i][t+1] = self.acc[i][t]
    #     self.rebAction = rebAction      
    #     # rebalancing
    #     for k in range(len(self.edges)):
    #         i,j = self.edges[k]    
    #         if (i,j) not in self.G.edges:
    #             continue
    #         # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
    #         # update the number of vehicles
            
    #         self.rebFlow[i,j][t+self.G.edges[i,j]['time']] = rebAction[k]     
    #         self.acc[i][t+1] -= rebAction[k] 
    #         self.dacc[j][t+self.G.edges[i,j]['time']] += self.rebFlow[i,j][t+self.G.edges[i,j]['time']]   
    #         self.info['rebalancing_cost'] += self.G.edges[i,j]['time']*self.beta*rebAction[k]
            
    #     if isMatching:  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
    #         paxAction = self.matching(CPLEXPATH=CPLEXPATH, PATH=PATH)
    #     self.paxAction = paxAction
    #     # serving passengers
    #     for k in range(len(self.edges)):
    #         i,j = self.edges[k]    
    #         if (i,j) not in self.G.edges:
    #             continue
    #         self.servedDemand[i,j][t] = paxAction[k]
    #         self.paxFlow[i,j][t+self.G.edges[i,j]['time']] = paxAction[k]
    #         if self.acc[i][t+1] - self.paxAction[k] < 0:
    #             self.paxAction[k] = self.acc[i][t+1]
    #         self.acc[i][t+1] -= paxAction[k]
    #         self.info['served_demand'] += self.servedDemand[i,j][t]
            
    #         self.dacc[j][t+self.G.edges[i,j]['time']] += self.paxFlow[i,j][t+self.G.edges[i,j]['time']]
            
    #     # arrival for the next time step
    #     for k in range(len(self.edges)):
    #         i,j = self.edges[k]    
    #         if (i,j) not in self.G.edges:
    #             continue
    #         self.acc[j][t+1] += self.paxFlow[i,j][t]                   
    #         self.acc[j][t+1] += self.rebFlow[i,j][t]
        
    #     for k in range(len(self.edges)):
    #         i,j = self.edges[k]    
    #         if (i,j) not in self.G.edges:
    #             continue
    #         # TODO: define reward here
    #         # defining the reward as: price * served demand - cost of rebalancing
    #         reward += (paxAction[k]*self.price[i,j][t] - self.G.edges[i,j]['time']*self.beta*(rebAction[k]+paxAction[k]))
    #         self.info["revenue"] += paxAction[k]*self.price[i,j][t]
    #         self.info["operating_cost"] += self.G.edges[i,j]['time']*self.beta*(rebAction[k]+paxAction[k])
    #     # observation: current vehicle distribution - for now, no notion of demand
    #     # TODO: define states here
    #     self.time += 1          
    #     self.obs = (self.acc, self.time)
    #     done = (self.tf == t+1) # if the episode is completed
    #     return self.obs, max(reward,0), done, self.info
    
    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i,i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        
        self.demand = defaultdict(dict) # demand
        self.price = defaultdict(dict) # price
        tripAttr = self.scenario.get_random_demand(reset=True)
        for i,j,t,d,p in tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            self.price[i,j][t] = p
            
        self.time = 0
        for i,j in self.G.edges:
            self.rebFlow[i,j] = defaultdict(float)
            self.paxFlow[i,j] = defaultdict(float)            
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float) 
        t = self.time
        for i,j in self.demand:
            self.servedDemand[i,j] = defaultdict(float)
         # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)      
        self.reward = 0
        return self.obs
    
    def MPC_exact(self, CPLEXPATH=None):
        t = self.time
        demandAttr = [(i,j,tt,self.demand[i,j][tt], self.price[i,j][tt]) for i,j in self.demand for tt in range(t,t+self.T) if self.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.acc[n][t]) for n in self.acc]
        daccTuple = [(n,tt,self.dacc[n][tt]) for n in self.acc for tt in range(t,t+self.T)]
        edgeAttr = [(i,j,self.G.edges[i,j]['time']) for i,j in self.G.edges]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        MPCPath = os.getcwd().replace('\\','/')+'/MPC/'
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.T)+';\r\n')
            file.write('beta='+str(self.beta)+';\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('daccAttr='+mat2str(daccTuple)+';\r\n')
            
        modfile = modPath+'MPC.mod'
        if CPLEXPATH is None:
            CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file =  MPCPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        with open(resfile,'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)',')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,j,f1,f2 = v.split(',')
                        paxFlow[int(i),int(j)] = float(f1)
                        rebFlow[int(i),int(j)] = float(f2)
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.edges]
        rebAction = [rebFlow[i,j] if (i,j) in rebFlow else 0 for i,j in self.edges]
        return paxAction,rebAction
    
    
    
class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, T=10, sd=None, ninit=5, tripAttr=None, demand_input=None,
                 trip_length_preference = 0.25, grid_travel_time = 1):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region, 
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        self.trip_length_preference = trip_length_preference
        self.grid_travel_time = grid_travel_time
        self.demand_input = demand_input
        self.N1 = N1
        self.N2 = N2
        self.G = nx.complete_graph(N1*N2)
        self.G = self.G.to_directed()
        for i,j in self.G.edges:
            self.G.edges[i,j]['time'] = (abs(i//N1-j//N1) + abs(i%N1-j%N1))*grid_travel_time
        for n in self.G.nodes:
            self.G.nodes[n]['accInit'] = ninit
        self.tf = tf
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        self.T = T
        if tripAttr != None: # given demand as a defaultdict(dict)
            self.tripAttr = deepcopy(tripAttr)
        else:
            self.tripAttr = self.get_random_demand() # randomly generated demand
    
    def get_random_demand(self, reset = False):        
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand
        
        demand = defaultdict(dict)
        price = defaultdict(dict)        
        tripAttr = []
        
        # default demand
        if self.demand_input == None:
            # generate demand, travel time, and price
            D = dict()
            for i,j in self.G.edges:
                D[i,j] = np.random.rand() * 0.5
                for t in range(0,self.tf+self.T):
                    if t%2 == 0:
                        if (i==0) and (j==7):
                            demand[i,j][t] = np.random.poisson(5)
                        elif (i==6) and (j==1):
                            demand[i,j][t] = np.random.poisson(5)
                        else:
                            demand[i,j][t] = np.random.poisson(D[i,j])
                    else:
                        if (i==7) and (j==0):
                            demand[i,j][t] = np.random.poisson(5)
                        elif (i==1) and (j==6):
                            demand[i,j][t] = np.random.poisson(5)
                        else:
                            demand[i,j][t] = np.random.poisson(D[i,j])
                    price[i,j][t] = min(3,np.random.exponential(2)+1) *self.G.edges[i,j]['time']
            tripAttr = []
            for i,j in demand:
                for t in demand[i,j]:
                    tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
            return tripAttr
        
        # converting demand_input to static_demand
        # skip this when resetting the demand
        if not reset:
            self.static_demand = dict()
            if type(self.demand_input) in [float, int, list, np.array]:
                if type(self.demand_input) in [float, int]:            
                    self.region_demand = np.random.rand(len(self.G)) * self.demand_input  
                else:
                    self.region_demand = self.demand_input            
                for i in self.G.nodes:
                    J = [j for _,j in self.G.out_edges(i)]
                    prob = np.array([np.math.exp(-self.G.edges[i,j]['time']*self.trip_length_preference) for j in J])
                    prob = prob/sum(prob)
                    for idx in range(len(J)):
                        self.static_demand[i,J[idx]] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i,j in self.G.edges:
                    self.static_demand[i,j] = self.demand_input[i,j] if (i,j) in self.demand_input else self.demand_input['default']
            else:
                raise Exception("demand_input should be number, array-like, or dictionary-like values")
        
        # generating demand and prices
        for t in range(0,self.tf+self.T):
            for i,j in self.G.edges:
                demand[i,j][t] = np.random.poisson(self.static_demand[i,j])
                price[i,j][t] = min(3,np.random.exponential(2)+1) * self.G.edges[i,j]['time']
                tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
        
        return tripAttr
    
                
if __name__=='__main__':
    # for training, put scenario inside the loop, for testing, put scenarios outside the loop and define sd
    scenario = Scenario(sd=10) # default one used in current training/testings    
    scenario = Scenario(sd=10,demand_input = {(1,6):2, (0,7):2, 'default':0.1}) # uni-directional 
    
    # only matching no rebalancing
    env1 = AMoD(scenario)
    opt_rew1 = []
    obs = env1.reset()
    done = False
    served1 = 0
    rebcost1 = 0
    opcost1 = 0
    revenue1 = 0
    while(not done):
        #print(env1.time)   
        
        obs, reward, done, info = env1.pax_step()
        opt_rew1.append(reward) # collect reward here to determine rebalancing actions
        rebAction = [0 for i,j in env1.edges]
        obs, reward, done, info = env1.reb_step(rebAction)
        served1 += info['served_demand']
        rebcost1 += info['rebalancing_cost']
        opcost1 += info['operating_cost']
        revenue1 += info['revenue']
        
    
    # MPC
    scenario = Scenario(sd=10,demand_input = {(1,6):2, (0,7):2, 'default':0.1}) # uni-directional
    env2 = AMoD(scenario)
    opt_rew2 = []
    obs = env2.reset()
    done = False
    served2 = 0
    rebcost2 = 0
    opcost2 = 0
    revenue2 = 0
    while(not done):
        #print(env2.time)         
        paxAction, rebAction = env2.MPC_exact()    
        obs, reward, done, info = env2.pax_step(paxAction)
        opt_rew2.append(reward) 
        obs, reward, done, info = env2.reb_step(rebAction)
        served2 += info['served_demand']
        rebcost2 += info['rebalancing_cost']
        opcost2 += info['operating_cost']
        revenue2 += info['revenue'] 