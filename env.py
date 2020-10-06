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
        self.L = int(round(np.log2(env.nregion))) # number of levels
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
            try:
                v_left = min(np.linspace(0, 1, self.Ms), key=lambda x:abs(x - (x_left / (x_left + x_right))))
                v_right = 1 - v_left
                s.append((k, v_left, v_right))
            except:
                s.append("lambda")
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
    



class AMoD:
    # initialization
        
    def __init__(self, scenario, beta): # updated to take scenario and beta (cost for rebalancing) as input
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

        # observation: current vehicle distribution, future arrivals, demand
								  
        self.obs = (self.acc, self.time)
													   
																							 
    def matching(self):
        t = self.time
        demandAttr = [(i,j,self.demand[i,j][t], self.price[i,j][t]) for i,j in self.demand \
                      if self.demand[i,j][t]>1e-3]
        accTuple = [(n,self.acc[n][t+1]) for n in self.acc]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        matchingPath = os.getcwd().replace('\\','/')+'/matching/'
        if not os.path.exists(matchingPath):
            os.makedirs(matchingPath)
        datafile = matchingPath + 'data_{}.dat'.format(t)
        resfile = matchingPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
        modfile = modPath+'matching.mod'
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

    # simulation step
    def step(self, rebAction, paxAction = None): 
        # rebAction/paxAction: np.array, where the kth element represents the number of vehicles going from region i to region j, (i,j) = self.edges[k]
        # paxAction is None if not provided (default matching algorithm will be used)
        self.info = dict.fromkeys(['revenue', 'served_demand', 'rebalancing_cost'], 0)
        t = self.time
        reward = 0
        
        self.rebAction = rebAction             
        
        for i in self.region:
            self.acc[i][t+1] = self.acc[i][t]
        
        # rebalancing
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            
            self.rebFlow[i,j][t+self.G.edges[i,j]['time']] = rebAction[k]     
            self.acc[i][t+1] -= rebAction[k] 
            self.dacc[j][t+self.G.edges[i,j]['time']] += self.rebFlow[i,j][t+self.G.edges[i,j]['time']]   
            self.info['rebalancing_cost'] += self.G.edges[i,j]['time']*self.beta*action[k]
            
        if paxAction == None:  # default matching algorithm used if paxAction is not provided
            paxAction = self.matching()
        self.paxAction = paxAction
        # serving passengers
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            self.servedDemand[i,j][t] = paxAction[k]
            self.paxFlow[i,j][t+self.G.edges[i,j]['time']] = paxAction[k]
            self.acc[i][t+1] -= paxAction[k]
            self.info['served_demand'] += self.servedDemand[i,j][t]
            
            self.dacc[j][t+self.G.edges[i,j]['time']] += self.paxFlow[i,j][t+self.G.edges[i,j]['time']]
            
        # arrival for the next time step
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            self.acc[j][t+1] += self.paxFlow[i,j][t]                   
            self.acc[j][t+1] += self.rebFlow[i,j][t]
        
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            # TODO: define reward here
            # defining the reward as: price * served demand - cost of rebalancing
            reward += (paxAction[k]*self.price[i,j][t] - self.G.edges[i,j]['time']*self.beta*(rebAction[k]+paxAction[k]))
 
        # observation: current vehicle distribution - for now, no notion of demand
        # TODO: define states here
        self.time += 1          
        self.obs = (self.acc, self.time)
																									
        done = (self.tf == t+1) # if the episode is completed
								
        return self.obs, max(reward,0), done, self.info
    
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
        self.obs = (self.acc, self.time)      
        return self.obs
    def MPC(self):
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
        return paxAction, rebAction
    
class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, T=10, sd=42, ninit=50, tripAttr=None, demand_scale=None,
                 trip_length_preference = 0.25, grid_travel_time = 2):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_scale： list - total demand out of each region, 
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_scale]
        #          dict/defaultdict - total demand between pairs of regions
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
        self.T = T
        if tripAttr != None: # given demand as a defaultdict(dict)
            self.tripAttr = deepcopy(tripAttr)
        else:
            self.tripAttr = self.get_random_demand(demand_scale,trip_length_preference) # randomly generated demand
    
    def get_random_demand(self, demand_scale,trip_length_preference):
        np.random.seed(self.sd)
        # generate demand, travel time, and price
        demand = defaultdict(dict)
        price = defaultdict(dict)
        
        tripAttr = []
        if type(demand_scale) in [float, int, list]:
            if type(demand_scale) in [float, int]:            
                Demand = np.random.rand(len(self.G)) * demand_scale
            elif type(demand_scale) == list:
                Demand = demand_scale
            for t in range(0,self.tf+self.T):
                for i in self.G.nodes:
                    J = [j for _,j in self.G.out_edges(i)]
                    prob = np.array([np.math.exp(-self.G.edges[i,j]['time']*trip_length_preference) for j in J])
                    prob = prob/sum(prob)
                    D = np.random.multinomial(np.random.poisson(Demand[i]),prob)
                    for idx,j in enumerate(J):            
                        demand[i,j][t] = D[idx]
                        price[i,j][t] = min(3,np.random.exponential(2)+1) * self.G.edges[i,j]['time']
                        tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
        elif type(demand_scale) in [dict, defaultdict]:
            for t in range(0,self.tf+self.T):
                for i,j in self.G.edges:    
                    demand[i,j][t] = np.random.poisson(demand_scale[i,j]) if (i,j) in demand_scale else 0                
                    price[i,j][t] = min(3,np.random.exponential(2)+1) * self.G.edges[i,j]['time']
                    tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
        else:
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
                    price[i,j][t] = min(3,np.random.exponential(2)+1) * self.G.edges[i,j]['time']
                    tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
        return tripAttr
    
                
if __name__=='__main__':
    scenario = Scenario(ninit=5,grid_travel_time = 1)
    env1 = AMoD(scenario, 0.2)
    opt_rew1 = []
    obs = env1.reset()
    done = False
    while(not done):
        print(env1.time)   
        rebAction = [0 for i,j in env1.edges]
        obs, reward, done, info = env1.step(rebAction)
        opt_rew1.append(reward) 
    
    env2 = AMoD(scenario, 0.2)
    opt_rew2 = []
    obs = env2.reset()
    done = False
    while(not done):
        print(env2.time)   
        rebAction = [0 for i,j in env2.edges]
        
        paxAction, rebAction = env2.MPC()    
        obs, reward, done, info = env2.step(rebAction, paxAction)
        # obs, reward, done = env2.step(rebAction)
        opt_rew2.append(reward) 
    