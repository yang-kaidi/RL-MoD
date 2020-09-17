# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:05:54 2020

@author: yangk
"""

import os
from collections import defaultdict
import subprocess
import re
import csv
import numpy as np
from multiprocessing import Pool
import random
from get_input import adjust_time, get_regions, obtain_trip_info,get_demand,get_cruise_prob,get_acc,get_compensation
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dayDict = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}


WT = 1
dt = 1
platform = 'thinkpad'
CURPATH= os.getcwd().replace('\\','/')
SSPATH = '{}/steady_state/'.format(CURPATH)
RESPATH = CURPATH+'/res_{}{}-{}_{}_{}_{}_{}/'
MODPATH = '{}/mod/'.format(CURPATH)
SUMMARYPATH = '{}/summary/'.format(CURPATH)  
if not os.path.exists(SUMMARYPATH):
    os.makedirs(SUMMARYPATH)
if not os.path.exists(SSPATH):
    os.makedirs(SSPATH)
my_env = os.environ.copy()
if platform == 'mac':
    CPLEXPATH = "/Applications/CPLEX_Studio1210/opl/bin/x86-64_osx/"
elif platform == 'surface':
    CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
elif platform == 'thinkpad':
    CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
elif platform == 'google':
    CPLEXPATH = "/home/yangkaidi07/CPLEX_Studio1210/opl/bin/x86-64_linux/"    
if platform == 'surface':
    CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"

if platform == 'mac':
    my_env["DYLD_LIBRARY_PATH"]  = CPLEXPATH
else:
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH

def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  

def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])

# HV assignment
def assignment(puc,dmd,SE,demand,spl, parameters):
    day,t,beta,N,gamma,xi,method = parameters    
    
    # preparing data
    data_file = RESPATH+'assigndata_{}{}_{}_{}_{}_{}.dat'.format(day,t,N,round(gamma,2),round(xi,2),method)
    res_file = RESPATH+'assignout_{}{}_{}_{}_{}_{}.dat'.format(day,t,N,round(gamma,2),round(xi,2),method)
    result_file = RESPATH+'assignres_{}{}_{}_{}_{}_{}.txt'.format(day,t,N,round(gamma,2),round(xi,2),method)
    with open(data_file, 'w') as file:
        file.write('path="'+result_file+'";\r\n')
        file.write('PickupCost='+str(set(puc)).replace("'",'"').replace('(','<').replace(')','>')+';\r\n')
        file.write('Demand='+str(set(dmd)).replace("'",'"').replace('(','<').replace(')','>')+';\r\n')
        file.write('solutionEdge='+mat2str(SE)+';\r\n')
        file.write('Supply='+str(set(spl)).replace("'",'"').replace('(','<').replace(')','>')+';\r\n')
    
    # solve optimization problem
    with open(res_file,'w') as output_f:
        subprocess.check_call([CPLEXPATH+"oplrun", MODPATH+'assignment.mod',data_file],stdout=output_f,env=my_env)
    output_f.close()
    
    # get solutions
    flow = dict()
    rebflow = dict()
    with open(result_file,'r', encoding="utf8") as file:
        count = 0
        for row in file:
            items = row.strip().replace('e|','|').strip('|').split('||')
            if len(items)>1:                   
                count += 1
                for item in items:
                    flowItem = item.split(',')
                    keys = flowItem[0].strip().strip('<').strip('>').replace('"','').split()
                    if len(keys)==3:
                        s,o,d = keys
                        flow[int(s),int(o),int(d)] = float(re.sub('[^0-9e.-]','', flowItem[1]))     
                    else:
                        s,o = keys
                        rebflow[int(s),int(o)] = float(re.sub('[^0-9e.-]','', flowItem[1]))     
    return flow, rebflow


def HVParameters(v, gamma,dt, day, hr_s,hr_e,pickupTime, pickupDist, regions, demand, compensation,paxTime,paxDist,autoRebTime,rebDist):
    avgPaxDemand = defaultdict(dict)
    avgPaxTime = defaultdict(dict)
    avgPaxDist = defaultdict(dict)
    avgPaxFare = defaultdict(dict)
    avgRebTime = defaultdict(dict)   
    avgRebDist = defaultdict(dict)   
    X = defaultdict(dict)
    Y = defaultdict(dict)
    Phi = defaultdict(dict)
    Pi = defaultdict(dict)
    for hr in range(hr_s,hr_e):
        for t in range(int(hr*60//dt),int((hr+1)*60//dt)):
            for o,d in demand[t]:
                if demand[t][o,d]<1e-3:
                    continue
                if hr in avgPaxDemand[o,d]:
                    avgPaxDemand[o,d][hr] += demand[t][o,d]/60 
                    avgPaxTime[o,d][hr] += paxTime[o,d][t//60]*demand[t][o,d]/60 
                    avgPaxFare[o,d][hr] += compensation[o,d][t//60]*demand[t][o,d]/60 
                    avgPaxDist[o,d][hr] += paxDist[o,d][t//60]*demand[t][o,d]/60 
                else:
                    avgPaxDemand[o,d][hr] = demand[t][o,d]/60 
                    avgPaxTime[o,d][hr] = paxTime[o,d][t//60]*demand[t][o,d]/60 
                    avgPaxFare[o,d][hr] = compensation[o,d][t//60]*demand[t][o,d]/60 
                    avgPaxDist[o,d][hr] = paxDist[o,d][t//60]*demand[t][o,d]/60 
        for o,d in avgPaxDemand:
            if hr in avgPaxDemand[o,d] and avgPaxDemand[o,d][hr]>1e-4:
                avgPaxTime[o,d][hr] /= avgPaxDemand[o,d][hr]
                avgPaxFare[o,d][hr] /= avgPaxDemand[o,d][hr]
                avgPaxDist[o,d][hr] /= avgPaxDemand[o,d][hr]
        
        for o,d in autoRebTime:
            if o!=d:
                avgRebTime[o,d][hr] = np.mean([autoRebTime[o,d][hr] for hr in range(hr,hr+1)])
                avgRebDist[o,d][hr] = np.mean([rebDist[o,d][hr] for hr in range(hr,hr+1)])
                
        for o,d in pickupTime:
            if  hr not in avgRebTime[o,d]:
                avgRebTime[o,d][hr] = np.mean([pickupTime[o,d][hr] for hr in range(hr,hr+1)])
                avgRebDist[o,d][hr] = np.mean([pickupDist[o,d][hr] for hr in range(hr,hr+1)])
        for o,d in avgPaxTime:
            if hr in avgPaxTime[o,d]:
                avgPaxTime[o,d][hr] += avgRebTime[o,o][hr]
                avgPaxDist[o,d][hr] += avgRebDist[o,o][hr]
            
        data_file = SSPATH + 'data_{}_{}_{}.txt'.format(hr,hr+1,v)    
        result_file = SSPATH + 'result_{}_{}_{}.txt'.format(hr,hr+1,v)
        P = [(o,d,avgPaxTime[o,d][hr],avgPaxFare[o,d][hr]+1e-6-gamma*avgPaxDist[o,d][hr],avgPaxDemand[o,d][hr]) for o,d in avgPaxDemand if hr in avgPaxDemand[o,d] ]
        R = [(o,d,avgRebTime[o,d][hr],-gamma*avgRebDist[o,d][hr],0) for o,d in avgRebTime if hr in avgRebTime[o,d] if o!=d ]
        with open(data_file,'w') as file:
            file.write('path="'+result_file+'";')
            file.write('PT='+mat2str(P)+';\r\n')
            file.write('RT='+mat2str(R)+';\r\n')
            file.write('N='+mat2str(regions)+';\r\n')
            file.write('v='+str(v)+';\r\n')
        
        x,y,phi,pi = steady_state(day,hr,hr+1,v)
        for i,j in x:
            X[i,j][hr] = x[i,j]
            Pi[i,j][hr] = pi[i,j]
        for i,j in y:
            Y[i,j][hr] = y[i,j]
        for i in phi:
            Phi[i][hr] = phi[i]
        print(dictsum(X,hr),dictsum(avgPaxDemand,hr))
    return Phi,avgPaxDemand

def steady_state(day,hr_s,hr_e,v):
    data_file = SSPATH + 'data_{}_{}_{}.txt'.format(hr_s,hr_e,v)
    result_file = SSPATH + 'result_{}_{}_{}.txt'.format(hr_s,hr_e,v)
    out_file = SSPATH + 'out_{}_{}_{}.txt'.format(hr_s,hr_e,v)
    mod = MODPATH + 'steady_state.mod' # 'optimization_fast.mod'
    my_env = os.environ.copy()
    if platform == 'mac':
        my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
    else:
        my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    
    with open(out_file,'w') as output_f:
        subprocess.check_call([CPLEXPATH+"oplrun", mod,data_file],stdout=output_f,env=my_env)
    output_f.close()

    x = defaultdict(float)
    y = defaultdict(float)
    phi = defaultdict(float)
    pi = defaultdict(float)
    with open(result_file,'r', encoding="utf8") as file:
        for row in file:
            item = row.replace('e)',')').strip().strip(';').split('=')
            if item[0] == 'x':
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                       continue
                    i,j,f,pp = [vv for vv in v.split(',')]                    
                    x[int(i),int(j)] = float(f)
                    pi[int(i),int(j)] = float(pp) 
            elif item[0] == 'phi':                
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                       continue
                    i,f = [vv for vv in v.split(',')]
                    phi[int(i)] = float(f)
            elif item[0] == 'y':                
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                       continue
                    i,j,f = [vv for vv in v.split(',')]
                    y[int(i),int(j)] = float(f)
    return x,y,phi,pi

def simulation(Input):
    BETA = 0.25    
    
    (parameters, regions, pickupTime, paxTime,pickupDist, paxDist, \
     autoRebTime,rebDist, acc, cruise, demand, fare,waitTime, served) = Input
    method, N, Ntot, hr_s, hr_e, dt, T, day, gamma,beta,commission, tl,\
        waitRatio,ttnoise,dnoise,sd,vot,xi  = parameters 
    
    for o,d in pickupTime:
        if (o,d) not in pickupDist:
            pickupDist[o,d] = [pickupTime[o,d][hr] * 20/60 for hr in range(0,24)]
    pickupEdge = list(pickupTime) #pickup edges
    pickupDict = defaultdict(list) # pickup edges for origin o
    for s,o in pickupEdge:
        pickupDict[o].append(s)
        
    rebEdge = [(i,j) for i,j in pickupTime if i!=j]
    rebEdge = list(set(rebEdge).union([(i,j) for i,j in autoRebTime if i!=j]))
    for e in rebEdge:
        if e not in autoRebTime:
            autoRebTime[e] = pickupTime[e]
        if e not in rebDist:
            rebDist[e] = pickupDist[e]
            
    
    compensation = defaultdict(dict)
    for o,d in fare:
        for hr in range(hr_s,hr_e):
            compensation[o,d][hr] = fare[o,d][hr]*0.8
        compensation[o,d][hr_e] =  compensation[o,d][hr_e-1] 
    Phi,avgPaxDemand = HVParameters(BETA,gamma,dt, day, hr_s,hr_e,pickupTime, pickupDist, 
                                    regions, demand, compensation,paxTime,paxDist,autoRebTime,rebDist)

    avgRegionDemand = defaultdict(float)
    for o,d in avgPaxDemand:
        if hr_s in avgPaxDemand[o,d]:
            avgRegionDemand[o] += avgPaxDemand[o,d][hr_s]
    nonAccInit = defaultdict(float)
    autoAccInit = dict()
    for o in regions:
        nonAccInit[o] = avgRegionDemand[o]/sum(avgRegionDemand.values()) * max(Ntot - N,0)    
        autoAccInit[o] = round(acc[hr_s*60//dt][o] * N/sum(acc[hr_s*60//dt].values()),1)
                              
    # define variables
    nonAcc = defaultdict(dict) # number of HVs ready for assignment
    autoAcc = defaultdict(dict) # number of AVs ready for assignment
    autoRebFlow = defaultdict(dict) # number of HVs in rebalancing
    nonRebFlow = defaultdict(dict) # number of AVs in rebalancing
    waitAssign = defaultdict(dict) # number of passengers waiting to be assigned
    autoWaitTime = defaultdict(dict) # waiting time of AV passengers
    nonWaitTime = defaultdict(dict) # waiting time of HV passengers
    autoReq = defaultdict(dict) # number of requests taken by AVs
    nonReq = defaultdict(dict) # number of requests taken by HVs
    autoFlow = defaultdict(dict) # number of AVs for passenger pickup and delivery
    nonFlow = defaultdict(dict) # number of HVs for passenger pickup and delivery
    totNonNum = Ntot-N # total number of HVs 
    totNonTime = (Ntot-N)*(hr_e-hr_s)*60 # time of HVs to be in the system
     
    for hr in range(hr_s,hr_e): 
        phi = dict()
        for s in Phi:
            if hr in Phi[s]:
                phi[s] = Phi[s][hr]
                
        for mi in range(0,60//dt):
            t = hr * 60//dt + mi
                
            if t == hr_s * 60//dt:    
                # initialize the accummulation and flows
                for n in regions:
                    autoAcc[n] = defaultdict(float)
                    nonAcc[n] = defaultdict(float)                                   
                    nonAcc[n][t] = nonAccInit[n]
                    autoAcc[n][t] = autoAccInit[n]               

                for e in rebEdge:
                    autoRebFlow[e] = defaultdict(float)
                    nonRebFlow[e] = defaultdict(float)
                    
            # set the active demand edges with positive demand

            demandEdge = [e for e in waitAssign if waitAssign[e][t]>1e-3]  
            demandEdge = list(set(demandEdge).union(set([(o,d) for tt in range(t,t+T) 
                                                         for o,d in demand[tt] if demand[tt][o,d]>1e-3])))
            for e in demandEdge:
                if e not in waitAssign:
                    waitAssign[e] = defaultdict(float)

            nonDemand = defaultdict(float)
            
            for o,d in demandEdge:
                if (o,d) in demand[t]:
                    nonDemand[o,d] = waitAssign[o,d][t] + demand[t][o,d]
                else:
                    nonDemand[o,d] = waitAssign[o,d][t]
                waitAssign[o,d][t+1] = nonDemand[o,d]
            # set the solution edges                        
            solutionEdge = [(s,o,d) for o,d in demandEdge for s in pickupDict[o]] 
            
            # initialize nonFlow and autoFlow if e not in autoFlow
            for e in solutionEdge:
                if e not in autoFlow:
                    nonFlow[e] = defaultdict(float)
                    autoFlow[e] = defaultdict(float)
            
            # calculate the expected arrivial non-AV flow for each region 
            # (e.g., the number of vehicles entering the system, 
            # the number of vehicles that will arrive at the region after delivering passengers)
            dacc = defaultdict(dict)                
            for r in regions:
                for tt in range(t,t+T):
                    if tt not in dacc[r]:
                        dacc[r][tt] = defaultdict(float)
               
            for s,o,d in nonFlow:
                for tt in nonFlow[s,o,d]:  
                    h = int(tt*dt//60)                      
                    if t<=tt + paxTime[o,d][h] + pickupTime[s,o][h] < t+T:
                        if tt + paxTime[o,d][h] + pickupTime[s,o][h] not in dacc[d]:
                            dacc[d][tt + paxTime[o,d][h] + pickupTime[s,o][h]] = defaultdict(float)
                        dacc[d][tt + paxTime[o,d][h] + pickupTime[s,o][h]]['non-auto'] += nonFlow[s,o,d][tt]
            
            # initializing nonAcc at the next time setp
            for i in regions:                
                nonAcc[i][t+1] = nonAcc[i][t] + dacc[i][t]['non-auto']
                
            # calculating rebalancing probabilities of HVs
            avgRegionDemand = defaultdict(float)
            for o,d in avgPaxDemand:
                if hr_s in avgPaxDemand[o,d]:
                    avgRegionDemand[o] += avgPaxDemand[o,d][hr_s]
            delta = dict()
            for o in regions:
                if avgRegionDemand[o] < 1e-3:
                    delta[o] = 60
                else:                                      
                    delta[o] = min(nonAcc[o][t+1]/(avgRegionDemand[o]),60)
                    
            rebProb = defaultdict(dict) # probability of rebalancing HVs arriving at an adjacent region
            
            for i,j in rebEdge: # logit model
                rebProb[i][j] = np.exp(-(autoRebTime[i,j][hr]*BETA+rebDist[i,j][hr]*gamma-phi[i]+phi[j]-delta[i]*BETA+delta[j]*BETA)*1)
            
            for i in rebProb:
                sumprob = sum(rebProb[i].values())
                for j in rebProb[i]:
                    rebProb[i][j] = rebProb[i][j]/(sumprob+1)/autoRebTime[i,j][hr] 
                    
                    
            if N>0:
                rebflow, flow = control()
                for i in regions:
                    autoAcc[i][t+1] += autoAcc[i][t]
                for i,j in rebflow:
                    autoAcc[i][t+1]-= rebflow[i,j]
                    autoRebFlow[i,j][t] =  rebflow[i,j] 
            else:
                rebflow, flow = {}, {}             
                            
            # dispatching AVs
            for s,o,d in flow:
                autoFlow[s,o,d][t] = flow[s,o,d]
                nonDemand[o,d] -= autoFlow[s,o,d][t]
                autoAcc[s][t+1] -= autoFlow[s,o,d][t] 
                waitAssign[o,d][t+1] -= autoFlow[s,o,d][t]
                if (o,d) in autoWaitTime and t in autoWaitTime[o,d] and autoFlow[s,o,d][t]>1e-3:
                    autoWaitTime[o,d][t] += autoFlow[s,o,d][t] * pickupTime[s,o][hr]
                    autoReq[o,d][t] += autoFlow[s,o,d][t]
                elif autoFlow[s,o,d][t]>1e-3:
                    autoWaitTime[o,d][t] = autoFlow[s,o,d][t] * pickupTime[s,o][hr]
                    autoReq[o,d][t] = autoFlow[s,o,d][t]
            
            # rebalancing of HVs
                
            for i,j in rebEdge:
                nonRebFlow[i,j][t] = nonAcc[i][t+1]* rebProb[i][j] #rebflow[i,j]/pickupTime[i,j][hr]

                        
            for i,j in rebEdge:
                if nonRebFlow[i,j][t]>0:
                    nonAcc[i][t+1] -= nonRebFlow[i,j][t]
            
            for i,j in rebEdge:
                if nonRebFlow[i,j][t]>0:
                    nonAcc[j][t+1] += nonRebFlow[i,j][t]
            
            # assigning HVs
            SO = list(pickupTime)
            puc = [(s,o,pickupTime[s,o][hr]*BETA+pickupDist[s,o][hr]*gamma-phi[s]+phi[o],pickupTime[s,o][hr]) for s,o in SO if nonAcc[s][t+1]>1e-3 ]            
            if len(puc) > 0:               
                dmd = [(o,d,nonDemand[o,d],
                        compensation[o,d][hr]+1e-3-phi[d]+phi[o]-BETA*(paxTime[o,d][hr])-gamma*paxDist[o,d][hr]) 
                for o,d in nonDemand if nonDemand[o,d]>1e-3]
                SE = [(s,o,d) for o,d in nonDemand for s in pickupDict[o] if nonDemand[o,d]>1e-3 and nonAcc[s][t+1]>1e-3]
                S = list(set([s for s,o, c,t in puc]))
                spl = [ (s,max(nonAcc[s][t+1],0)) for s in S ]
                
                flow,rebflow = assignment(puc,dmd,SE,nonDemand,spl,(day,t,BETA,N,gamma,xi,method))
            else:
                flow,rebflow = {},{}
            flowEdge = list(flow)

            for s,o,d in flowEdge:
                if (s,o,d) not in nonFlow:
                    nonFlow[s,o,d] = defaultdict(float)
                nonFlow[s,o,d][t] = flow[s,o,d] 
                nonDemand[o,d] -= flow[s,o,d]
                waitAssign[o,d][t+1] -= nonFlow[s,o,d][t]
                nonAcc[s][t+1] -= nonFlow[s,o,d][t] 
                if t in nonWaitTime[o,d]:
                    nonWaitTime[o,d][t] += nonFlow[s,o,d][t] * pickupTime[s,o][hr]
                    nonReq[o,d][t] += nonFlow[s,o,d][t]
                else:
                    nonWaitTime[o,d][t] = nonFlow[s,o,d][t] * pickupTime[s,o][hr]
                    nonReq[o,d][t] = nonFlow[s,o,d][t]
                    
            for e in nonDemand:
                waitAssign[e][t+1] = waitRatio*min(waitAssign[e][t+1], sum([demand[tt][e] for tt in range(t-WT+1,t+1) if e in demand[tt]]))
                # waitRatio: percentage of passengers that would stay
                # WT: maximum time that passengers would wait
    
    # Values to record
    totAutoReq = sum([autoReq[o,d][t] for o,d in autoReq for t in autoReq[o,d]])
    totNonReq = sum([nonReq[o,d][t] for o,d in nonReq for t in nonReq[o,d]])
    totDemand = sum([sum(demand[tt].values()) for tt in range(hr_s*60//dt,hr_e*60//dt)])
    totNonWait = sum([nonWaitTime[o,d][t] for o,d in nonWaitTime for t in nonWaitTime[o,d]])
    totAutoWait = sum([autoWaitTime[o,d][t] for o,d in autoWaitTime for t in autoWaitTime[o,d]])
    totAutoTravelDist = sum([autoFlow[s,o,d][t]*(paxDist[o,d][int(t*dt//60)]+pickupDist[s,o][int(t*dt//60)]) for s,o,d in autoFlow  for t in autoFlow[s,o,d]])
    totNonTravelDist = sum([nonFlow[s,o,d][t]*(paxDist[o,d][int(t*dt//60)]+pickupDist[s,o][int(t*dt//60)]) for s,o,d in nonFlow for t in nonFlow[s,o,d]]) 
    totCompensation =  sum([nonFlow[s,o,d][t]*compensation[o,d][int(t*dt//60)] for s,o,d in nonFlow for t in nonFlow[s,o,d]]) 
    totAutoFare = sum([autoFlow[s,o,d][t]*fare[o,d][int(t*dt//60)] for s,o,d in autoFlow  for t in autoFlow[s,o,d]])
    totNonFare = sum([nonFlow[s,o,d][t]*fare[o,d][int(t*dt//60)] for s,o,d in nonFlow for t in nonFlow[s,o,d]]) 
    totAutoTravel = sum([autoFlow[s,o,d][t]*paxTime[o,d][int(t*dt//60)] for s,o,d in autoFlow  for t in autoFlow[s,o,d]])
    totNonTravel = sum([nonFlow[s,o,d][t]*paxTime[o,d][int(t*dt//60)] for s,o,d in nonFlow for t in nonFlow[s,o,d]]) 
    totAutoReb = sum([autoRebFlow[i,j][t]*autoRebTime[i,j][int(t*dt//60)] for i,j in autoRebFlow for t in autoRebFlow[i,j]])
    totNonReb = sum([nonRebFlow[i,j][t]*autoRebTime[i,j][int(t*dt//60)] for i,j in nonRebFlow for t in nonRebFlow[i,j]])
    totAutoRebDist = sum([autoRebFlow[i,j][t]*rebDist[i,j][int(t*dt//60)] for i,j in autoRebFlow for t in autoRebFlow[i,j]])
    totNonRebDist = sum([nonRebFlow[i,j][t]*rebDist[i,j][int(t*dt//60)] for i,j in nonRebFlow for t in nonRebFlow[i,j]])
    operatorProfit =  totNonFare + totAutoFare - totCompensation - gamma * (totAutoTravelDist+totAutoRebDist)
    nonEarning = ((1-commission[1]) * totNonFare-gamma * (totNonTravelDist+totNonRebDist))/totNonTime*60 if totNonTime >1e-3 else None
    paxUti = totAutoFare + totNonFare - vot * (totNonWait + totAutoWait)
    SocialWelfare = totNonFare + totAutoFare - vot * (totNonWait + totAutoWait)  \
        - gamma * (totAutoTravelDist+totNonTravelDist+totAutoRebDist+totNonRebDist) 
    results = [N,totNonNum,totNonTime,totAutoReq,totNonReq,totDemand,totNonWait,
               totAutoWait,totAutoFare,totNonFare,totAutoTravel,totNonTravel,totAutoTravelDist,totNonTravelDist,
               totAutoReb,totNonReb,totAutoRebDist,totNonRebDist,operatorProfit,nonEarning,
               (totNonWait+totAutoWait)/(totNonReq+totAutoReq),
               (totNonReq+totAutoReq)/totDemand,paxUti,SocialWelfare]
    with open(SUMMARYPATH+'{}_{}{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(method,day,hr_s,hr_e,round(beta,2),round(gamma,2),N,T,round(vot,2),round(xi,2)),'w') as file:
        csv.writer(file).writerow(['N','totNonNum','totNonTime','totAutoReq','totNonReq','totDemand',
                                   'totNonWait','totAutoWait','totAutoFare','totNonFare','totAutoTravel',
                                   'totNonTravel','totAutoTravelDist',
                                   'totNonTravelDist','totAutoReb','totNonReb','totAutoRebDist','totNonRebDist','operatorProfit','nonEarning',
                                   'paxWaitTime','paxAcceptance','paxUtil','SocialWelfare'])
        csv.writer(file).writerow(results)

    return


if __name__=='__main__':   
    
    hr_s = 8
    hr_e = 10
    dt = 1  
    T = int(15//dt)
    day = 'Wednesday'
    commission = [1,0.2]
    tl = 1    
    gamma = 0.15
    waitRatio = 0
    dl = 2
    beta = 0.25
    vot = 0.45
    xi = 0
    
    RESPATH = RESPATH.format(day,hr_s,hr_e,round(beta,2),round(vot,2),round(xi,1),dl)
    if not os.path.exists(RESPATH):
        os.makedirs(RESPATH)
    
    Ntot = 14000
    ttnoise = 0.1
    dnoise = 0.1
    sd = 42

    regions,r2n = get_regions(dl=dl)
    pickupTime, paxTime, pickupDist, paxDist, autoRebTime, rebDist, fare = obtain_trip_info(['pickupTime', 'paxTime', 'pickupDist', 'paxDist', 'rebTime', 'rebDist','fare'], day,r2n,dl=dl)
    pickupTime, paxTime, autoRebTime,pickupDist = adjust_time(['pickupTime', 'paxTime','autoRebTime','pickupDist'], [pickupTime, paxTime,autoRebTime,pickupDist], 1,regions, r2n, day)
    
    cruise = get_cruise_prob(day,r2n,dl=dl)    
    acc, occupied, dispatch,idle = get_acc(day,r2n,dl=dl)
    request,demand,served,waitTime = get_demand(day,r2n,dl=dl)
    
    # sd = 0
    # random.seed(sd)
    # np.random.seed(sd)
    # station = [0,1,2,3,4,5,6,7]
    # tt = defaultdict(dict)
    # ds = defaultdict(dict)
    # dd = defaultdict(dict)
    # ff = defaultdict(dict)
    # cc = defaultdict(dict)
    
    # for r1 in station:
    #     for r2 in station:
    #         xx = random.randint(3,30)
    #         for hr in range(8,11):
    #             if r1 == r2:
    #                 tt[r1,r2][hr] = 3
    #             else:
    #                 tt[r1,r2][hr] = xx
    #             ff[r1,r2][hr] = tt[r1,r2][hr]
    #             ds[r1,r2][hr] = tt[r1,r2][hr]

    
    #     for r1 in station:
    #         for r2 in station:
    #             if r1<r2:
    #                 x = random.randint(0,20)
    #                 for t in range(480,615):
    #                     if x !=0:
    #                         dd[t][r1,r2] = np.random.poisson(x)
                            
    # aa = defaultdict(dict)
    # for r in station:
    #     aa[480][r] = 10
            
    NSet = [0]
    inputs = []
    for N in NSet:
        for method in ['social']:
            for xi in [6]:
                parameters = (method,N,Ntot,hr_s,hr_e,dt,T,day,gamma, beta,commission,tl,waitRatio,ttnoise,dnoise,sd,vot,xi)  
                inputs.append((parameters, regions, pickupTime, paxTime,pickupDist, paxDist, autoRebTime,rebDist, acc, cruise, demand, fare,waitTime, served))
                # inputs.append((parameters, station, tt, tt,ds, ds, tt,ds, aa, {}, dd, ff,tt, dd))

    for Input in inputs:
        simulation(Input)
 
    
