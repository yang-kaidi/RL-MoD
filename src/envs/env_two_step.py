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
from src.misc.utils import mat2str
from copy import deepcopy

class AMoD:
    # initialization
    def __init__(self, scenario, beta=0.2): # updated to take scenario and beta (cost for rebalancing) as input 
        self.scenario = deepcopy(scenario) # I changed it to deep copy so that the scenario input is not modified by env 
        self.G = scenario.G # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.time = 0 # current time
        self.tf = scenario.tf # final time
        self.demand = defaultdict(dict) # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G) # set of regions
        for i in self.region:
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)
            
        self.price = defaultdict(dict) # price
        for i,j,t,d,p in scenario.tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            self.price[i,j][t] = p 
            self.depDemand[i][t] += d
            self.arrDemand[i][t+self.G.edges[i,j]['time']] += d
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

    def matching(self, CPLEXPATH=None, PATH='', platform = 'linux'):
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
        if platform == 'mac':
            my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
        else:
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
    def pax_step(self, paxAction=None, CPLEXPATH=None, PATH='', platform =  'linux'):
        t = self.time
        self.reward = 0
        for i in self.region:
            self.acc[i][t+1] = self.acc[i][t]
        self.info['served_demand'] = 0 # initialize served demand
        self.info["operating_cost"] = 0 # initialize operating cost
        self.info['revenue'] = 0
        self.info['rebalancing_cost'] = 0
        if paxAction is None:  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
            paxAction = self.matching(CPLEXPATH=CPLEXPATH, PATH=PATH, platform = platform)
        self.paxAction = paxAction
        # serving passengers
        for k in range(len(self.edges)):
            i,j = self.edges[k]    
            if (i,j) not in self.G.edges:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            assert paxAction[k] < self.acc[i][t+1] + 1e-3
            self.paxAction[k] = min(self.acc[i][t+1], paxAction[k])            
            self.servedDemand[i,j][t] = self.paxAction[k]
            self.paxFlow[i,j][t+self.G.edges[i,j]['time']] = self.paxAction[k]
            self.info["operating_cost"] += self.G.edges[i,j]['time']*self.beta*self.paxAction[k]
            self.acc[i][t+1] -= self.paxAction[k]
            self.info['served_demand'] += self.servedDemand[i,j][t]            
            self.dacc[j][t+self.G.edges[i,j]['time']] += self.paxFlow[i,j][t+self.G.edges[i,j]['time']]
            self.reward += self.paxAction[k]*(self.price[i,j][t] - self.G.edges[i,j]['time']*self.beta)            
            self.info['revenue'] += self.paxAction[k]*(self.price[i,j][t])  
        
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
        self.regionDemand= defaultdict(dict)
        for i,j,t,d,p in tripAttr: # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i,j][t] = d
            self.price[i,j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] +=d
            
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
   
    
    
class Scenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio = None,
                 trip_length_preference = 0.25, grid_travel_time = 1, fix_price=False, alpha = 0.2):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region, 
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        if demand_ratio != None:
            self.demand_ratio = list(np.interp(range(0,tf), np.arange(0,tf+1, tf/(len(demand_ratio)-1)), demand_ratio))+[1]*tf
        else:
            self.demand_ratio = [1]*(tf+tf)
            
        
        self.alpha = alpha
        self.trip_length_preference = trip_length_preference
        self.grid_travel_time = grid_travel_time
        self.demand_input = demand_input
        self.fix_price = fix_price
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
        if self.fix_price: # fix price
            self.p = defaultdict(dict)
            for i,j in self.G.edges:
                self.p[i,j] = (np.random.rand()*2+1)*self.G.edges[i,j]['time']
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
        
        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        self.static_demand = dict()            
        region_rand = (np.random.rand(len(self.G))*self.alpha*2+1-self.alpha) 
        if type(self.demand_input) in [float, int, list, np.array]:
            
            if type(self.demand_input) in [float, int]:            
                self.region_demand = region_rand * self.demand_input  
            else:
                self.region_demand = region_rand * np.array(self.demand_input)
            for i in self.G.nodes:
                J = [j for _,j in self.G.out_edges(i)]
                prob = np.array([np.math.exp(-self.G.edges[i,j]['time']*self.trip_length_preference) for j in J])
                prob = prob/sum(prob)
                for idx in range(len(J)):
                    self.static_demand[i,J[idx]] = self.region_demand[i] * prob[idx]
        elif type(self.demand_input) in [dict, defaultdict]:
            for i,j in self.G.edges:
                self.static_demand[i,j] = self.demand_input[i,j] if (i,j) in self.demand_input else self.demand_input['default']
                
                self.static_demand[i,j] *= region_rand[i]
        else:
            raise Exception("demand_input should be number, array-like, or dictionary-like values")
        
        # generating demand and prices
        if self.fix_price:
            p = self.p
        for i,j in self.G.edges:
            for t in range(0,self.tf*2):
                demand[i,j][t] = np.random.poisson(self.static_demand[i,j]*self.demand_ratio[t])
                if self.fix_price:
                    price[i,j][t] = p[i,j]
                else:
                    price[i,j][t] = min(3,np.random.exponential(2)+1)*self.G.edges[i,j]['time']
                tripAttr.append((i,j,t,demand[i,j][t],price[i,j][t]))

        return tripAttr

class Star2Complete(Scenario):
    def __init__(self, N1 = 4, N2 =4, sd = 10, star_demand = 20, complete_demand = 1, star_center = [5,6,9,10], grid_travel_time=3, ninit = 50, demand_ratio=[1,1.5,1.5,1], alpha=0.2, fix_price=False): 
        # beta - proportion of star network
        # alpha - parameter for uniform distribution of demand [1-alpha, 1+alpha]
        super(Star2Complete, self).__init__(N1=N1,N2=N2,sd=sd, ninit = ninit, 
                                            grid_travel_time=grid_travel_time, 
                                            fix_price = fix_price,
                                            alpha = alpha,
                                            demand_ratio = demand_ratio,
                                            demand_input = {(i,j): complete_demand + (star_demand if i in star_center and j not in star_center else 0) for i in range(0,N1*N2) for j in range(0,N1*N2) if i!=j}
                                            )
        
if __name__=='__main__':
    # for training, put scenario inside the loop, for testing, put scenarios outside the loop and define sd
    #scenario = Scenario(sd=10) # default one used in current training/testings    
    #scenario = Scenario(sd=10,demand_input = {(1,6):2, (0,7):2, 'default':0.1}) # uni-directional 
    #scenario = Scenario(sd=10,demand_input = {(1,6):20, (0,7):20, 'default':1}, ninit = 60, demand_ratio=[1,1.5,1])

    # only matching no rebalancing
    # env1 = AMoD(scenario)
    # opt_rew1 = []
    # obs = env1.reset()
    # done = False
    # served1 = 0
    # rebcost1 = 0
    # opcost1 = 0
    # revenue1 = 0
    # while(not done):
    #     #print(env1.time)   
        
    #     obs, reward, done, info = env1.pax_step()
    #     opt_rew1.append(reward) # collect reward here to determine rebalancing actions
    #     rebAction = [0 for i,j in env1.edges]
    #     obs, reward, done, info = env1.reb_step(rebAction)
    #     served1 += info['served_demand']
    #     rebcost1 += info['rebalancing_cost']
    #     opcost1 += info['operating_cost']
    #     revenue1 += info['revenue']
        
    
    # MPC
    #scenario = Scenario(sd=10,demand_input = {(1,6):20, (0,7):20, 'default':1}, ninit = 60, demand_ratio=[1,1.5,1], alpha = 0.2)
    scenario = Star2Complete(star_demand = 6, complete_demand=1.6, beta=0.7, ninit = 200)

    env2 = AMoD(scenario)
    
    for step in range(0,5):
    
    
        CPLEXPATH = '/opt/ibm/ILOG/CPLEX_Studio1210/opl/bin/x86-64_linux/'
        
        opt_rew2 = []
        obs = env2.reset()
        done = False
        served2 = 0
        rebcost2 = 0
        opcost2 = 0
        revenue2 = 0
        demand2 = sum([env2.demand[i,j][t] for i,j in env2.demand for t in range(0,60)])
        while(not done):
            #print(env2.time)         
            paxAction, rebAction = env2.MPC_exact(CPLEXPATH=CPLEXPATH)    
            obs, reward, done, info = env2.pax_step(paxAction,CPLEXPATH = CPLEXPATH)
            opt_rew2.append(reward) 
            obs, reward, done, info = env2.reb_step(rebAction)
            served2 += info['served_demand']
            rebcost2 += info['rebalancing_cost']
            opcost2 += info['operating_cost']
            revenue2 += info['revenue'] 
        print(demand2, served2/demand2)