"""
Functions for aggregating mountain car states
"""

import numpy as np
import math
import random
    

def generateAggregation(qhat,target_divisions=64,start_epsilon=0.1):
    """
    Takes an existing estimate qhat and generates a dictionary mapping from raw states
    to "close to" target_divisions aggregated states
    """
    epsilon = start_epsilon

    while True:
        
        agg, n = aggregateToEpsilon(qhat,epsilon=epsilon,stop_at=2*target_divisions)
        if abs(target_divisions/float(n))>0.66: break
        epsilon*=2
            
    return agg

def aggregateToEpsilon(qhat,epsilon=0.1,stop_at=float('inf'),aggType='q'):
    """
    Generates an aggregation phi of states which are phi-uniform within epsilon
    """
    agg = {}
    central_class_members = []

    for xdot, xs in enumerate(qhat):
        for x, q in enumerate(xs):
            for i, (cxdot,cx) in enumerate(central_class_members):
                cqval = qhat[cxdot][cx]
                if dInf(q,cqval,aggType)<epsilon:
                    agg[(xdot,x)] = i
            if (xdot,x) not in agg:
                central_class_members.append((xdot,x))
                l = len(central_class_members)
                if l>stop_at:
                    return agg, l
                agg[(xdot,x)] = central_class_members.index((xdot,x))

    agg['term'] = l
    agg['n'] = l+1
    print("epsilon {}, len {}".format(epsilon,l+1))
    return agg, l+1



def dInf(x,y,aggType='q'):
    if np.argmax(x) != np.argmax(y):
        return float('inf')
    elif aggType == 'q':
        return max(abs(x -y))
    else:
        return abs(max(x)-max(y))

def kMeans(qhat,k,aggType='q'):
    """
    Aggregate by k-means clustering, return aggregation and maximum class width
    """
    x_max, xdot_max, a = qhat.shape
    qhat = qhat.reshape(x_max*xdot_max,a)
    mi = np.random.choice(np.arange(x_max*xdot_max),k)
    converged = False
    agg = {'n':k+1}
    agg['term']=k 
    classes = {j:[qhat[v],[]] for j, v in enumerate(mi)}
    unc = 0
    
    while not converged:
        converged = True
        d_min_max = 0
        for i, q in enumerate(qhat):
            xdot = i/100
            x = i % 100

            d_min = float('inf')
            for c, (mu,l) in classes.items():
                d = dInf(q,mu,aggType)
                if d < d_min:
                    d_min = d
                    tmp_class = c
            if d_min == float('inf'):
                unc += 1
                if unc > 10:
                    mi = np.random.choice(np.arange(x_max*xdot_max),k)
                    classes = {j:[qhat[v],[]] for j, v in enumerate(mi)}
                continue
            d_min_max = max(d_min_max,d_min)
            classes[tmp_class][1].append(q)
            agg[(xdot,x)] = tmp_class
        
        for c, (mu, l) in classes.items():
            nu_mu = np.average(np.array(l),axis=0)
            if max(abs(mu-nu_mu)) > 1e-4:
                converged = False
                classes[c][0] = nu_mu
                classes[c][1] = []


    return (agg, d_min_max)
