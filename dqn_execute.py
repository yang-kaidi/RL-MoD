import math
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
import os
import subprocess
from tqdm import trange
from copy import deepcopy
from util import mat2str
from env import Scenario, AMoD
 
# DQN imports
import torch
import torch.nn.functional as F
from dqn import DQN_Agent

platform = "mac"
plt.style.use('ggplot')
#CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
if platform == "windows":
    CPLEXPATH = "C:/Program Files/IBM/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
elif platform == "mac":
    CPLEXPATH = "/Applications/CPLEX_Studio1210/opl/bin/x86-64_osx/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
 
steps_done = 0
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.15
EPS_END = 0.01
EPS_DECAY = 60*100
TARGET_UPDATE = 10
 
def select_action(state, dqn, test=False):
    global steps_done
    sample = random.random()
    if test:
        eps_threshold = 0.
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return dqn.policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(dqn.nA)]], device=device, dtype=torch.long)
   
        
def optimize_model(dqn):
    if len(dqn.memory) < BATCH_SIZE:
        return
    transitions = dqn.memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
 
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).float()
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
 
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    dqn.policy_net.train()
    state_action_values = dqn.policy_net(state_batch).gather(1, action_batch)
 
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = dqn.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
 
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
 
    # Optimize the model
    dqn.optimizer.zero_grad()
    loss.backward()
    for param in dqn.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    dqn.optimizer.step()
 
def training(scenario, env, dqn,sidx):
    # book-keeping variables
    training_rewards = []
    training_revenue = []
    training_served_demand = []
    training_rebalancing_cost = []
    training_operating_cost = []
   
    last_t_update = 0
    train_episodes = 200 # num_of_episodes_with_same_epsilon x num_of_q_tables x num_epsilons         
    max_steps = 100 # maximum length of episode
    epochs = trange(train_episodes) # build tqdm iterator for loop visualization
 
    for i_episode in epochs:
        obs = env.reset()
        state = torch.tensor(dqn.decode_state(obs)).to(device).view(1,-1).float()
        episode_reward = 0
        episode_revenue = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        episode_operating_cost = 0
        for step in range(max_steps):
            # Select and perform an RL action
            dqn.policy_net.eval()
            action_rl = select_action(state,dqn)
 
            # 1.2 get actual vehicle distributions vi (i.e. (x1*x2*..*xn)*num_vehicles)
            v_d = dqn.get_desired_distribution(action_rl)
 
            # 1.3 Solve ILP - Minimal Distance Problem
            # 1.3.1 collect inputs and build .dat file
            t = dqn.env.time
            accTuple = [(n,int(dqn.env.acc[n][t])) for n in dqn.env.acc]
            accRLTuple = [(n, int(v_d_n)) for n, v_d_n in enumerate(v_d)]
            edgeAttr = [(i,j,dqn.env.G.edges[i,j]['time']) for i,j in dqn.env.G.edges]
            modPath = os.getcwd().replace('\\','/')+'/mod/'
            OPTPath = os.getcwd().replace('\\','/')+'/OPT/DQN/Train/'
            if not os.path.exists(OPTPath):
                os.makedirs(OPTPath)
            datafile = OPTPath + f'data_{t}_{sidx}_training.dat'
            resfile = OPTPath + f'res_{t}_{sidx}_training.dat'
            with open(datafile,'w') as file:
                file.write('path="'+resfile+'";\r\n')
                file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
                file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
                file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')
 
            # 2. execute .mod file and write result on file
            modfile = modPath+'minRebDistRebOnly.mod'
            my_env = os.environ.copy()
            if platform == 'mac':
                my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
            else:
                my_env["LD_LIBRARY_PATH"] = CPLEXPATH
            out_file =  OPTPath + f'out_{t}_{sidx}_training.dat'
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
            rebAction = [flow[i,j] for i,j in dqn.env.edges]
 
            # Take step
            new_obs, reward, done, info = env.step(rebAction, isMatching=True, CPLEXPATH=CPLEXPATH, PATH="DQN/Train/")
            new_state = torch.tensor(dqn.decode_state(new_obs)).to(device).view(1,-1).float()
 
            reward = torch.tensor([reward], device=device).float()
 
            # Store the transition in memory
            dqn.memory.push(state, action_rl, new_state, reward)
 
            # Move to the next state
            # track performance over episode
            episode_reward += reward.item()
            episode_revenue += info['revenue']
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            episode_operating_cost += info['operating_cost']
            obs, state = deepcopy(new_obs), deepcopy(new_state)
 
            # Perform one step of the optimization (on the target network)
            optimize_model(dqn)
            if done:
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            dqn.target_net.load_state_dict(dqn.policy_net.state_dict())
            last_t_update = i_episode
        epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Revenue: {episode_revenue:.2f} | ServedDemand: {episode_served_demand:.2f} \
| Reb. Cost: {episode_rebalancing_cost:.2f} | Oper. Cost: {episode_operating_cost:.2f}| Epsilon: {EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)},\
Idx: {0} | Last Target Update {last_t_update}")
        #Adding the total reward and reduced epsilon values
        training_rewards.append(episode_reward)
        training_revenue.append(episode_revenue)
        training_served_demand.append(episode_served_demand)
        training_rebalancing_cost.append(episode_rebalancing_cost)
        training_operating_cost.append(episode_operating_cost)
 
   
    torch.save(dqn.policy_net.state_dict(), f"policy_net_{sidx}_training")
    # Plot results
    fig = plt.figure(figsize=(12,32))
    fig.add_subplot(411)
    plt.plot(training_rewards, label="Reward")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("J")
    plt.legend()
   
    fig.add_subplot(412)
    plt.plot(training_revenue, label="Revenue")
    plt.title("Episode Revenue")
    plt.xlabel("Episode")
    plt.ylabel("Revenue")
    plt.legend()
   
    fig.add_subplot(413)
    plt.plot(training_served_demand, label="Served Demand")
    plt.title("Episode Served Demand")
    plt.xlabel("Episode")
    plt.ylabel("Served Demand")
    plt.legend()
   
    fig.add_subplot(414)
    plt.plot(training_rebalancing_cost, label="Reb. Cost")
    plt.title("Episode Reb. Cost")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
    fig.savefig(f'{sidx}_training.png')
 
    print("Average Performance: \n")
    print(f'Avg Reward: {np.mean(training_rewards):.2f}')
    print(f'Total Revenue: {np.mean(training_revenue):.2f}')
    print(f'Total Served Demand: {np.mean(training_served_demand):.2f}')
    print(f'Total Rebalancing Cost: {np.mean(training_rebalancing_cost):.2f}')
   
def testing(scenario, env, dqn, sidx):
    # Test Episodes
    test_episodes = 100
    epochs = trange(test_episodes) # build tqdm iterator for loop visualization
    np.random.seed(10)
    max_steps = 100 # maximum length of episode
 
 
    # book-keeping variables
    test_rewards = []
    test_revenue = []
    test_served_demand = []
    test_rebalancing_cost = []
    test_operating_cost = []
   
    for episode in epochs:
        try:
            obs = env.reset()
            state = torch.tensor(dqn.decode_state(obs)).to(device).view(1,-1).float()
            episode_reward = 0
            episode_revenue = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            episode_operating_cost = 0
            for step in range(max_steps):
                dqn.policy_net.eval()
                action_rl = select_action(state, dqn, test=True)
   
                # 1.2 get actual vehicle distributions vi (i.e. (x1*x2*..*xn)*num_vehicles)
                v_d = dqn.get_desired_distribution(action_rl)
   
                # 1.3 Solve ILP - Minimal Distance Problem
                # 1.3.1 collect inputs and build .dat file
                t = dqn.env.time
                accTuple = [(n,int(dqn.env.acc[n][t])) for n in dqn.env.acc]
                accRLTuple = [(n, int(v_d_n)) for n, v_d_n in enumerate(v_d)]
                edgeAttr = [(i,j,dqn.env.G.edges[i,j]['time']) for i,j in dqn.env.G.edges]
                modPath = os.getcwd().replace('\\','/')+'/mod/'
                OPTPath = os.getcwd().replace('\\','/')+'/OPT/DQN/Test/'
                if not os.path.exists(OPTPath):
                    os.makedirs(OPTPath)
                datafile = OPTPath + f'data_{t}_{sidx}_testing.dat'
                resfile = OPTPath + f'res_{t}_{sidx}_testing.dat'
                with open(datafile,'w') as file:
                    file.write('path="'+resfile+'";\r\n')
                    file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
                    file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
                    file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')
   
                # 2. execute .mod file and write result on file
                modfile = modPath+'minRebDistRebOnly.mod'
                my_env = os.environ.copy()
                if platform == 'mac':
                    my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
                else:
                    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
                out_file =  OPTPath + f'out_{t}_{sidx}_testing.dat'
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
                rebAction = [flow[i,j] for i,j in dqn.env.edges]
       
                # Take step
                new_obs, reward, done, info = env.step(rebAction, isMatching=True, CPLEXPATH=CPLEXPATH, PATH="DQN/Test/")
                new_state = torch.tensor(dqn.decode_state(new_obs)).to(device).view(1,-1).float()
               
                # track performance over episode
                episode_reward += reward
                episode_revenue += info['revenue']
                episode_served_demand += info['served_demand']
                episode_rebalancing_cost += info['rebalancing_cost']
                episode_operating_cost += info['operating_cost']
                obs, state = deepcopy(new_obs), deepcopy(new_state)
   
                # end episode if conditions reached
                if done:
                    break
               
            epochs.set_description(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Revenue: {episode_revenue:.2f} | ServedDemand: {episode_served_demand:.2f} \
    | Oper. Cost: {episode_operating_cost:.2f}")
            #Adding the total reward and reduced epsilon values
            test_rewards.append(episode_reward)
            test_revenue.append(episode_revenue)
            test_served_demand.append(episode_served_demand)
            test_rebalancing_cost.append(episode_rebalancing_cost)
            test_operating_cost.append(episode_operating_cost)
        except KeyboardInterrupt:
            break
    fig = plt.figure(figsize=(12,32))
    fig.add_subplot(411)
    plt.plot(test_rewards, label="Reward")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("J")
    plt.legend()
   
    fig.add_subplot(412)
    plt.plot(test_revenue, label="Revenue")
    plt.title("Episode Revenue")
    plt.xlabel("Episode")
    plt.ylabel("Revenue")
    plt.legend()
   
    fig.add_subplot(413)
    plt.plot(test_served_demand, label="Served Demand")
    plt.title("Episode Served Demand")
    plt.xlabel("Episode")
    plt.ylabel("Served Demand")
    plt.legend()
   
    fig.add_subplot(414)
    plt.plot(test_rebalancing_cost, label="Reb. Cost")
    plt.title("Episode Reb. Cost")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
    fig.savefig(f'{sidx}_testing.png')
 
    print("Average Performance: \n")
    print(f'Avg Reward: {np.mean(test_rewards):.2f}')
    print(f'Total Revenue: {np.mean(test_revenue):.2f}')
    print(f'Total Served Demand: {np.mean(test_served_demand):.2f}')
    print(f'Total Rebalancing Cost: {np.mean(test_rebalancing_cost):.2f}')
   
if __name__=='__main__':
   
    scenarios = [Scenario(), Scenario(demand_input = {(1,6):2, (0,7):2, 'default':0.1}) ]
    MA = [5,10]
    sidx = 0
    for scenario in scenarios[0:1]:
        for Ma in [5]: 
            env = AMoD(scenario, sidx=sidx)
            dqn = DQN_Agent(env)
#            training(scenario, env, dqn,sidx)
            dqn.policy_net.load_state_dict(torch.load(f"policy_net_{sidx}_training"))           
            testing(scenario, env, dqn,sidx)
            sidx += 1