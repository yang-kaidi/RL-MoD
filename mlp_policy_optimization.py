import torch
import torch.nn as nn
from torch.distributions.multinomial import Multinomial
from torch.optim import Adam
import numpy as np
from env_policy_optimization import Scenario, AMoD
import networkx as nx
from collections import defaultdict
from collections.abc import Iterable

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env, hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=600, render=False):
    
    obs_dim = len(env.obs)# sum([len(item) if isinstance(item, Iterable) else 1 for item in env.obs])
    n_acts = len(env.edges)

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    print(obs_dim,n_acts)
    nidx = np.concatenate(([0],np.cumsum(env.nedge)))
    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(torch.tensor(obs))
        return [Multinomial(int(obs[k]),logits=logits[nidx[k]:nidx[k+1]]) for k in range(len(nidx)-1)]

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return torch.cat([policy.sample() if obs[k]>0 else torch.tensor([0.]*(nidx[k+1]-nidx[k])) for k,policy in enumerate(get_policy(obs))])

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        loss = 0
        # print(obs.size(),act.size(),weights.size())
        for m in range(obs.size()[0]):
            policy = get_policy(obs[m,:])
            logp = sum([policy[k].log_prob(act[m,nidx[k]:nidx[k+1]].type(torch.FloatTensor)) for k in range(len(nidx)-1)])
            # print(logp)
            loss -= logp* weights[m]/obs.size()[0]
        return loss

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            # act = env.MPC()
            obs, rew, done, _ = env.step(act)
            # print([env.acc[n][env.time] for n in env.acc],env.action,rew)
            # save action, reward
            batch_acts.append(np.array(act))
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                # print(len(batch_obs))
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens,batch_acts

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens,batch_acts = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    return batch_acts
if __name__ == '__main__':
    scenario = Scenario() # default one used in current training/testings    
    # scenario = Scenario(demand_input = {(1,6):2, (0,7):2, 'default':0.1}) # uni-directional 
    
    env = AMoD(scenario)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument('--lr', type=float, default=1e-2)
    # args = parser.parse_args()
    # print('\nUsing reward-to-go formulation of policy gradient.\n')
    batch_acts = train(env)