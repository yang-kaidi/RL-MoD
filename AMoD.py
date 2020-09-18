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
def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  
def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])


class AMoD:
    # initialization
    def __init__(self, G, tripAttr, parameters):
        self.G = G # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.time = 0 # current time
        self.T = parameters[0] # planning time
        self.tf = parameters[1] # final time
        self.demand = defaultdict(dict) # demand
        self.price = defaultdict(dict) # price
        for i,j,t,d,p in tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            self.price[i,j][t] = p
        self.acc = defaultdict(dict) # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(dict) # number of vehicles arriving at each region, key: i - region, t - time
        self.rebFlow = defaultdict(dict) # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict) # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = [] # set of rebalancing edges
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
        self.beta = parameters[2]
        t = self.time
        self.servedDemand = defaultdict(float)
        for i,j in self.demand:
            self.servedDemand[i,j] = defaultdict(float)
        # observation: current vehicle distribution, future arrivals, demand
        # TODO: define states here
        self.obs = np.array([self.acc[i][t] if tt ==t else self.dacc[i][t] for tt in range(t,t+self.T) for n in self.G])
        self.nact = len(self.edges) # number of actions
        self.nobs = len(self.G) * self.T + len(self.demand) * self.T # number of observations


    # simulation step
    def step(self, action): #action: np.array, where the kth element represents the number of vehicles going from region i to region j, (i,j) = self.edges[k]
        t = self.time
        reward = 0
        self.action = action
        for i in self.region:
            self.acc[i][t+1] = self.acc[i][t]
            
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            # update the number of vehicles
            self.acc[i][t+1] += self.paxFlow[i,j][t] + self.rebFlow[i,j][t] - action[k]
            self.paxFlow[i,j][t+self.G.edges[i,j]['time']] = min([self.demand[i,j][t],action[k]])
            self.rebFlow[i,j][t+self.G.edges[i,j]['time']] = action[k] - min([self.demand[i,j][t],action[k]])
            self.dacc[i][t+self.G.edges[i,j]['time']] += self.paxFlow[i,j][t+self.G.edges[i,j]['time']]
            self.dacc[i][t+self.G.edges[i,j]['time']] += self.rebFlow[i,j][t+self.G.edges[i,j]['time']]
            
            # TODO: define reward here
            # defining the reward as: price * served demand - cost of rebalancing - cost of serving the demand
            reward += (min([self.demand[i,j][t],action[k]])*self.price[i,j][t] - self.G.edges[i,j]['time']*self.beta*action[k])
        
            self.servedDemand[i,j][t] = min([self.demand[i,j][t],action[k]])
        # observation: current vehicle distribution, future arrivals, demand
        # TODO: define states here
        self.obs = np.array([self.acc[i][t+1] if tt ==t else self.dacc[i][t] for tt in range(t,t+self.T) for i in self.G]
                       + [self.demand[i,j][tt] for tt in range(t,t+self.T) for i,j in self.demand ])
        done = (self.tf == t+1) # if the episode is completed
        self.time += 1          
        return self.obs, max(reward,0), done
    
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
        self.obs = np.array([self.acc[i][t] if tt ==t else self.dacc[i][t] for tt in range(t,t+self.T) for i in self.G]
                       + [self.demand[i,j][tt] for tt in range(t,t+self.T) for i,j in self.demand ])
        self.nact = len(self.edges)
        self.nobs = len(self.G) * self.T + len(self.demand) * self.T
        
        return self.obs        
    
    def MPC(self):
        t = self.time
        demandAttr = [(i,j,tt,self.demand[i,j][tt], self.price[i,j][tt]) for i,j in self.demand for tt in range(t,t+self.T) if self.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.acc[n][t]) for n in self.acc]
        daccTuple = [(n,tt,self.dacc[n][tt]) for n in self.acc for tt in range(t+1,t+self.T)]
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
        action = [flow[i,j] for i,j in self.edges]
        return action
    
if __name__ == '__main__':
    N = 5 # sqrt root of the number of regions (the road map is a square)
    G = nx.complete_graph(N*N) # generate a complete graph
    G = G.to_directed() # to directed graph
     
    tf = 60 # length of an episode
    T= 10
    beta = 0.2 # cost for driving per unit time
    np.random.seed(42)
    
    # generate demand, travel time, and price
    D = dict()
    demand = defaultdict(dict)
    price = defaultdict(dict)
    for i,j in G.edges:
        G.edges[i,j]['time'] = (abs(i//N-j//N) + abs(i%N-j%N))*2
        D[i,j] = np.random.rand() * 0.5
        for t in range(0,tf+T):
            demand[i,j][t] = np.random.poisson(D[i,j])
            price[i,j][t] = min(3,np.random.exponential(2)+1) * G.edges[i,j]['time']
    tripAttr = []
    for i,j in demand:
        for t in demand[i,j]:
            tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
    for n in G.nodes:
        G.nodes[n]['accInit'] = 50
    
    # MPC-based control
    env = AMoD(G,tripAttr,[T,tf,beta])
    opt_rew = []
    obs = env.reset()
    done = False
    while(not done):
        print(env.time)        
        act = env.MPC()    
        obs, reward, done = env.step(act)
        opt_rew.append(reward) 
        
        
    
    
            
        
        
            
        
    