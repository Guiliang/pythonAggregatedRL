"""
Functions that set up and perform complete experiments

David J 2015
"""

import pandas as pd
import Evaluation
import Random_MDP
import numpy as np
import pylab as pl
import Agents
import Problems
import Aggregation
import pickle
from datetime import datetime

path = 'Data/'

def gen_param_tuples(n_min=32,n_max=None,
                  a_min=16,a_max=None,
                  b_min=16,b_max=None,
                  act_min=1,act_max=None,
                  e_min=0.,e_max=None):
    """
    Returns a list of parameter tuples with minimum and maximum values given in arguments
    Passing None for a maximum holds that value constant
    Parameters are spaced in powers of 2
    """

    min_array = np.array([n_min,a_min,b_min,act_min,e_min+1.])
    max_array = np.copy(min_array)
    for i, m in enumerate([n_max,a_max,b_max,act_max,e_max]):
        if m:
            max_array[i] = m

    assert (min_array>0).all(), "minimum values must be > 0,"\
                                +"except epsilon which must be >= 0"
    assert (min_array[0]>=min_array[1:3]).all(), "agg and b must be at most n"
    
    min_pows = map(int,np.floor(np.log2(min_array)))
    max_pows = map(lambda x: int(x+1),np.floor(np.log2(max_array)))
        
    return [(2**i,2**j,2**k,2**l,(2**m)-1) for i in range(min_pows[0],max_pows[0])
                                       for j in range(min_pows[1],min(i,max_pows[1]))
                                       for k in range(min_pows[2],min(i+1,max_pows[2]))
                                       for l in range(min_pows[3],max_pows[3])
                                       for m in range(min_pows[4],max_pows[4])]



def test_convergence_ql(param_tuple,timeout=100,interval=100,gamma=0.9,aggtype='q'):
    """
    Generates a random problem from (single) parameter tuple, runs qlearning 
    and records value function deviation at regular intervals. 
    """
    n, n_agg, b, acts, e_noise = param_tuple
    p_r, p_a, _ = problems.genRandomProblems(n,n_agg,acts,b,gamma=gamma,e_noise=e_noise)
    
    agent_r = Agents.QAgent(p_r,alpha=1e-1)
    agent_a = Agents.QAgent(p_a,alpha=1e-1)
    
    dmat = np.zeros((intervals,3))
    
    for i in range(intervals):
        agent_r.episode(timeout=timeout)
        agent_a.episode(timeout=timeout)
        
        delta_a = Evaluation.getDeltas(agent_a,p_a)
        delta_r = Evaluation.getDeltas(agent_r,p_r)
        
        dmat[i] = (i*interval,delta_a,delta_r)
        
    data = pd.DataFrame(data=dmat,columns=['n','d_a','d_r'])
    return data


def compare_raw_agg_ql(param_tuples,timeout=1000,gamma=0.5,aggtype='q',log_prob=False,rep=1,alpha=0.005,decayAlpha=False):
    """
    Generates random problems from parameter lists, runs qlearning 
    and records value function deviations for each problem after timeout steps
    log optionally records problems so that they can be retrieved later
    """

    dmat = np.zeros((len(param_tuples)*rep,9))
    
    if log_prob:
        problem_dict = {}
        d = datetime.today().strftime('%d-%m-%Y--%H_%M_%S')
        filename = 'problems' + d + '.pkl'
           
    for i, (n,n_agg,b,acts,e_noise) in enumerate(param_tuples):
        for j in range(rep):           
            p_r, p_a, _ = Problems.genRandomProblems(n,n_agg,acts,b,gamma=gamma,e_noise=e_noise)
                    
            pid = hash(str(p_r.transitions))
        
            if log_prob:
                problem_dict[pid] = {'raw':p_r,'agg':p_a}
                    
            agent_r = Agents.QAgent(p_r,alpha=alpha)
            agent_a = Agents.QAgent(p_a,alpha=alpha)
                    
            agent_r.episode(timeout=timeout,decayAlpha=decayAlpha)
            agent_a.episode(timeout=timeout,decayAlpha=decayAlpha)
        
            delta_r = Evaluation.getDeltas(agent_r,p_r)
            delta_a = Evaluation.getDeltas(agent_a,p_a,agg=p_a.aggregation)
        
            dtilde = Evaluation.nonMarkovianity(p_a.transitions[0], p_a.aggregation)

            dmat[i*rep+j] = (pid,n,n_agg,b,acts,e_noise,dtilde,np.average(delta_r),np.average(delta_a))
            
                    
    if log_prob:
        with open(path+filename,'wb') as f:
            pickle.dump(problem_dict,f)
                    
    data = pd.DataFrame(data=dmat,columns=['pid','n','n_agg','b','acts','e_noise','nonmarkovianity','d_r','d_a'])
    return data


def test_mountain_car(n_states,qhat_file='Data/mc_qhat.pkl',max_eps = 10000):
    """
    Compares aggregated mountain car with n_states with a simple discretised version of the same.
    Supplying a precalculated qhat file will drastically speed computation
    """

    divs = int(np.floor(np.sqrt(n_states)))

    if qhat_file:
        with open(qhat_file,'rb') as f:
            qhat = pickle.load(f)
    else:
        qhat = evaluate_MC_qhat()

    agg = Aggregation.generateAggregation(qhat,target_divisions=n_states)

    mc_d = Problems.MountainCar(representation='disc',divisions = divs)
    mc_a = Problems.MountainCar(representation='aggr',aggregation=agg,divisions=100)
    
    ag_d = Agents.QAgent(mc_d,alpha=1e-3)
    ag_a = Agents.QAgent(mc_a,alpha=1e-3)

    d_eps = ag_d.run_n_episodes(max_eps)
    a_eps = ag_a.run_n_episodes(max_eps)

    n_eps = np.array([2**i for i in range(1,int(np.log2(max_eps))+1)])

    data = pd.DataFrame()
    data['n_eps'] = n_eps
    data['disc'] = d_eps
    data['aggr'] = a_eps

    return data

def evaluate_MC_qhat():
    p = Problems.MountainCar(representation='tile')
    a = Agents.QAgent(p,alpha=0.01,tiles=True)
    a.run_n_episodes(35000)
    qv = agent.approxQValues(100)
    return qv
