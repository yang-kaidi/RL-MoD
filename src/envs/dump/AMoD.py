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
            reward += (paxAction[k]*self.price[i,j][t] - self.G.edges[i,j]['time']*self.beta*rebAction[k])
 
        # observation: current vehicle distribution - for now, no notion of demand
        # TODO: define states here
        self.time += 1          
        self.obs = (self.acc, self.time)
																									
        done = (self.tf == t+1) # if the episode is completed
								
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
        self.obs = (self.acc, self.time)      
    
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
    
if __name__ == '__main__':
    N1 = 4 # sqrt root of the number of regions (the road map is a square)
    N2 = 2
    G = nx.complete_graph(N1*N2) # generate a complete graph
    G = G.to_directed() # to directed graph
     
    tf = 60 # length of an episode
    T= 10
    beta = 0.001 # cost for driving per unit time
    np.random.seed(42)
    
    # generate demand, travel time, and price
    demand = defaultdict(dict)
    price = defaultdict(dict)
    Demand = defaultdict(float)
    tripAttr = []
    for i,j in G.edges:
        G.edges[i,j]['time'] = (abs(i//N1-j//N1) + abs(i%N1-j%N1))*2
    
    
    for i in G.nodes:
        Demand[i] = np.random.rand() * N1*N2 * 3
    
    for t in range(0,tf+T):
        for i in G.nodes:
            J = [j for _,j in G.out_edges(i)]
            prob = np.array([np.math.exp(-G.edges[i,j]['time']/N1) for j in J])
            prob = prob/sum(prob)
            D = np.random.multinomial(np.random.poisson(Demand[i]),prob)
            for idx,j in enumerate(J):            
                demand[i,j][t] = D[idx]
                price[i,j][t] = min(3,np.random.exponential(2)+1) * G.edges[i,j]['time']
                tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))
            
    for n in G.nodes:
        G.nodes[n]['accInit'] = 30
    
    # MPC-based control
    env1 = AMoD(G,tripAttr,[T,tf,beta])
    opt_rew1 = []
    obs = env1.reset()
    done = False
    while(not done):
        print(env1.time)   
        rebAction = [0 for i,j in env1.edges]
        
        # paxAction, rebAction = env.MPC()    
        # obs, reward, done = env.step(rebAction, paxAction)
        obs, reward, done = env1.step(rebAction)
        opt_rew1.append(reward) 
    
    env2 = AMoD(G,tripAttr,[T,tf,beta])
    opt_rew2 = []
    obs = env2.reset()
    done = False
    while(not done):
        print(env2.time)   
        rebAction = [0 for i,j in env2.edges]
        
        paxAction, rebAction = env2.MPC()    
        obs, reward, done = env2.step(rebAction, paxAction)
        # obs, reward, done = env2.step(rebAction)
        opt_rew2.append(reward) 
        
        
    
    
            
        
        
            
        
    