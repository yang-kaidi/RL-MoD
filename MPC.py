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


class MPC:
    def __init__(self, env, CPLEXPATH=None):
        self.env = env 
        self.CPLEXPATH = CPLEXPATH
    def MPC_exact(self, CPLEXPATH=None):
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,t+self.env.T) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,t+self.env.T)]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        MPCPath = os.getcwd().replace('\\','/')+'/MPC/exact/'
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.env.T)+';\r\n')
            file.write('beta='+str(self.env.beta)+';\r\n')
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
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        rebAction = [rebFlow[i,j] if (i,j) in rebFlow else 0 for i,j in self.env.edges]
        return paxAction,rebAction
    
    def bi_level_matching(self):
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,t+self.env.T) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,t+self.env.T)]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        MPCPath = os.getcwd().replace('\\','/')+'/MPC/bi-level-matching/'
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.env.T)+';\r\n')
            file.write('beta='+str(self.env.beta)+';\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('daccAttr='+mat2str(daccTuple)+';\r\n')
           
        modfile = modPath+'bi-level-matching.mod'
        if self.CPLEXPATH is None:
            self.CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = self.CPLEXPATH
        out_file =  MPCPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([self.CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
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
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        rebAction = [rebFlow[i,j] if (i,j) in rebFlow else 0 for i,j in self.env.edges]
        return rebAction
        
    def bi_level_rebalancing(self):
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,t+self.env.T) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,t+self.env.T)]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        MPCPath = os.getcwd().replace('\\','/')+'/MPC/bi-level-rebalancing/'
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.env.T)+';\r\n')
            file.write('beta='+str(self.env.beta)+';\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('daccAttr='+mat2str(daccTuple)+';\r\n')
           
        modfile = modPath+'bi-level-rebalancing.mod'
        if self.CPLEXPATH is None:
            self.CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = self.CPLEXPATH
        out_file =  MPCPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([self.CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        desiredAcc = defaultdict(float)
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
                elif item[0] == 'desiredAcc':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,f = v.split(',')
                        desiredAcc[int(i)] = float(f)
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        return paxAction,desiredAcc
    
    def bi_level_rebalancing_2_actions(self):
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,t+self.env.T) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,t+self.env.T)]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        MPCPath = os.getcwd().replace('\\','/')+'/MPC/bi-level-rebalancing-2-actions/'
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.env.T)+';\r\n')
            file.write('beta='+str(self.env.beta)+';\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('daccAttr='+mat2str(daccTuple)+';\r\n')
           
        modfile = modPath+'bi-level-rebalancing-2-actions.mod'
        if self.CPLEXPATH is None:
            self.CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = self.CPLEXPATH
        out_file =  MPCPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([self.CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        departure = defaultdict(float)
        arrival = defaultdict(float)
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
                elif item[0] == 'acc':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,f1,f2 = v.split(',')
                        departure[int(i)] = float(f1)
                        arrival[int(i)] = float(f2) 
                       
                        
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        return paxAction,departure,arrival
        
    def tri_level(self):
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,t+self.env.T) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,t+self.env.T)]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\','/')+'/mod/'
        MPCPath = os.getcwd().replace('\\','/')+'/MPC/tri-level/'
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.env.T)+';\r\n')
            file.write('beta='+str(self.env.beta)+';\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('daccAttr='+mat2str(daccTuple)+';\r\n')
           
        modfile = modPath+'tri-level.mod'
        if self.CPLEXPATH is None:
            self.CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = self.CPLEXPATH
        out_file =  MPCPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([self.CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        desiredAcc = defaultdict(float)
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
                elif item[0] == 'desiredAcc':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,f = v.split(',')
                        desiredAcc[int(i)] = float(f)
        return desiredAcc

def solveRebFlow(env,res_path,desiredAcc,CPLEXPATH):
    t = env.time
    accRLTuple = [(n,desiredAcc[n]) for n in desiredAcc]
    accTuple = [(n,int(env.acc[n][t])) for n in env.acc]
    edgeAttr = [(i,j,env.G.edges[i,j]['time']) for i,j in env.G.edges]
    modPath = os.getcwd().replace('\\','/')+'/mod/'
    OPTPath = os.getcwd().replace('\\','/')+'/MPC/'+res_path
    if not os.path.exists(OPTPath):
        os.makedirs(OPTPath)
    datafile = OPTPath + f'data_{t}.dat'
    resfile = OPTPath + f'res_{t}.dat'
    with open(datafile,'w') as file:
        file.write('path="'+resfile+'";\r\n')
        file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
        file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
        file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')
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
    action = [flow[i,j] for i,j in env.edges]
    return action