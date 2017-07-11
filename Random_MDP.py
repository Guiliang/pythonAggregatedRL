'''
Functions for generating random aggregatable MDPs

David Johnston 2015
'''

import numpy as np
import random

def getValues(n,aggStates,actions,aggType='q',e_noise=0):
    """
    Generates state values given a number of states and aggregated states
    """    
    vv = np.zeros((actions,n))
    
    for a in range(actions):
        if a == 0:
            vv[a] = getVStar(n,aggStates,e_noise=e_noise) 
        else:
            vv[a] = getVSub(n,aggStates,vv[0],aggType=aggType,e_noise=e_noise) 
    
    agg = { (i+j*n/aggStates): j for i in range(n/aggStates) for j in range(aggStates)}
    agg['n'] = aggStates
        
    return vv, agg

def getVStar(n,aggStates,e_noise=0):
    """
    Generates state values given a number of states and aggregated states
    """    
    if n%aggStates:
        raise ValueError("number of aggregated states must"+\
                         "be divisor of number of states")

    vp = random.sample(range(0,10*aggStates),aggStates)
    v = [(vp[i] + random.random()*e_noise)  for i in range(aggStates) for j in range(int(n/aggStates))]

    return np.array(v)


def getVSub(n,aggStates,vstar,aggType='q',e_noise=0):
    """
    Generate a set of values for suboptimal actions
    """


    if aggType == 'v':
        v = [random.randint(i-40,i-1) for i in vstar]
    elif aggType == 'q':
        v = []
        for i in range(aggStates):
            vs = vstar[int(n*i/aggStates)]
            val = random.randint(vs - 40, vs - 1)
            v = v + [(val + random.random()*e_noise) for i in range(int(n/aggStates))]

    #print(vstar,v)
    return np.array(v)

def getTrns(n,b):
    """
    Generates a random invertible nxn matrix with b non-zero entries per row.
    Row entries sum to 1
    """
    valid = 0
    count = 0
    count_max = 1.e8/(n**2)

    assert b>= 2, "branching must be at least 2"

    while not valid:
        unreachableStates = set(range(n))
        count += 1
        m = np.random.rand(n,n)

        for row in m:
            zeros = random.sample(range(n),(n-b))
            for i in zeros:
                row[i] = 0
            row /= sum(row)

        diag = 0

        for i, row in enumerate(m):
            if row[i]:
                diag = 1

        if diag and isIrreducible(m) and np.linalg.det(m):
            valid = 1

        if count >= count_max:
            raise ValueError("Could not find a valid matrix for" + 
                             "n={},b={}".format(n,b))

    return m


def isIrreducible(matrix):
    """
    returns t if matrix is irreducible
    returns f if it might not be
    """
    for i in range(20):
        if not 0 in np.linalg.matrix_power(matrix,i):
            return True

    return False


def optRewardMatrix(TMatrix,values,gamma=0.5):
    """
    Calculates a reward matrix given a transition matrix and desired value function
    """
    n = len(TMatrix)
    rv = np.linalg.solve(TMatrix,(np.identity(n)-gamma*TMatrix).dot(values))

    rM = np.vstack([rv for i in rv])

    return rM

def subRewardMatrix(TMatrix,vstar,subvalues,gamma=0.5):
    """
    Calculates a reward matrix for suboptimal actions
    """
    rv = np.linalg.solve(TMatrix,subvalues-gamma*TMatrix.dot(vstar))
    rM = np.vstack([rv for i in rv])

    return rM

def getRewards(TMatrices,value_vecs,gamma = 0.5):
    """
    Sets up rewards. Transitions, values for optimal action are in the first entry of 
    TMatrices, value_vecs
    """
    n = len(value_vecs[0])
    m = len(value_vecs)
    rewards = np.zeros((m,n,n))

    vstar = value_vecs[0]



    for i, (TM, vv) in enumerate(zip(TMatrices,value_vecs)):
        if i == 0:
            rewards[i]  = optRewardMatrix(TM,vv,gamma)
        else:
            rewards[i] = subRewardMatrix(TM,vstar,vv,gamma)

    return rewards

def getAggregation(values,e_noise=0):
    """
    Produces a dictionary mapping raw states to aggregated states in the form 
    {raw_state:aggregated_state}
    """
    assert e_noise<1

    rounded_values = map(int,values)

    unique_values = list(set(rounded_values))

    aggregation = {i:unique_values.index(v) for i, v in enumerate(rounded_values)}

    aggregation['n'] = len(unique_values)

    return aggregation
