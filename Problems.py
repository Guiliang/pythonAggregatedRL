'''
Reinforcement Learning Problems

David Johnston 2015
'''


import math
import numpy as np
import random
from datetime import datetime
from Tiles import tiles
import Random_MDP
#import Aggregation

def genRandomProblems(n,n_agg,actions,b,gamma,aggType='q',e_noise=0):
    '''
    Returns an aggregated and raw problem instance with 
    same underlying MDP and n, n_agg states respectively
    '''
    transitions = np.zeros((actions,n,n))
    
    values, aggregation = Random_MDP.getValues(n,n_agg,actions,aggType=aggType,e_noise=e_noise)
    for i in range(len(transitions)):
        transitions[i] =  Random_MDP.getTrns(n,b)
    rewards = Random_MDP.getRewards(transitions,values,gamma)

    p_raw = MDP(0,transitions,rewards,gamma,qValues=values)
    p_agg = MDP(0,transitions,rewards,gamma,aggregation=aggregation,qValues=values)
    p_vec = MDP(0,transitions,rewards,gamma,aggregation=aggregation,qValues=values,vec_output=1)

    return p_raw, p_agg, p_vec
    
class MDP:
    """
    An MDP. Contains methods for initialisation, state transition. 
    Can be aggregated or unaggregated.
    """

    def __init__(self, initial, transitions, rewards, gamma, aggregation = None,
                 qValues = None, vec_output = 0):       
        self.actions = range(len(transitions))
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma

        self.qValues = qValues

        self.reset = None
        self.isEpisodic = 0
        self.aggregation = aggregation

        self.rawStates = len(transitions[0])
        if isinstance(aggregation,dict):
            self.nStates = aggregation['n']
        else:
            self.nStates = self.rawStates
            
        self.vec_output = vec_output
        if vec_output:
            self.dimSizes = [self.nStates,self.rawStates]
        else:
            self.dimSizes = [self.nStates]

        self.problemState = initial

        d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
        b = sum([1 for x in transitions[0][0] if x>0])
        self.probName = "{}_n={}_b={}_gamma={}_agg={}".format(d,len(transitions[0]),
                                                              b,gamma,self.nStates)

        
    def getZeroQTable(self):
        return np.zeros((self.nStates,len(self.actions)))
        
    def setAlpha(self, alpha):
        return alpha
    
    def getActions(self):
        return self.actions
        
    def getAgentState(self):
        if self.aggregation:
            return self.aggregation[self.problemState]
        else:
            return self.problemState
 
            
    def result(self, action):

        stateVec = np.zeros(self.rawStates)
        stateVec[self.problemState] = 1
        
        successorVec = stateVec.dot(self.transitions[action])

        successor = np.random.choice(self.rawStates,1,p=successorVec)[0]
        
        reward = self.rewards[action][self.problemState][successor]

        self.problemState = successor           

        if self.aggregation:
            agg_successor = self.aggregation[successor]
            if self.vec_output:
                return [agg_successor,successor], reward
            else:
                return agg_successor, reward
        else:
            return successor, reward         




class MountainCar():
    """
    The mountain car problem, with parameters as given by Sutton at
    http://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.lisp.

    The state can be represented in a variety of ways; the raw representation is
    for agents that create their own state representations for continuous values 
    while the other three work with table-driven agents.
    """

    def __init__(self,randomStart=0,representation='raw',
                 aggregation=None,
                 numtilings=16,
                 divisions=10):

        assert representation in {'raw','disc','aggr','tile','qtree'}, "invalid representation"
        assert isinstance(randomStart,int), "randomStart must be int"
        assert isinstance(divisions, int), "divisions must be int"


        self.x = -0.5
        self.xMin = -1.2
        self.xGoal = 0.5

        self.xdot = 0
        self.xDotMax = 0.07
        
        self.dt = 1e-3
        self.grav = 2.5

        self.isEpisodic = 1
        self.gamma = 1
        self.randomStart = randomStart

        self.representation = representation
        self.divisions = divisions
        self.dimSizes = [divisions**2]

        self.actions = range(3) #this is converted to [-1,0,1] in the actual updates

        self.isEpisodic = True

        if representation == 'tile':
            self.numtilings = numtilings
            minMem = numtilings*self.divisions*self.divisions*10
            i = 1
            while 2**i < minMem:
                i += 1
            self.nStates = 2**i

            self.ctable = tiles.CollisionTable(sizeval=self.nStates)

            self.getAgentState = self.getTileState
        
        elif representation == 'disc':
            self.nStates = self.divisions**2
            self.getAgentState = self.getDiscState

        elif representation == 'aggr':
            assert isinstance(aggregation,dict), "invalid aggregation supplied"
            self.aggregation = aggregation
            self.nStates = self.aggregation['n']
            self.getAgentState = self.getAggrState

        elif representation == 'raw':
            self.getAgentState = self.getRawState

        elif representation == 'qtree':
            self.getAgentState = self.getQTree
            self.depth = int(math.floor(np.log2(divisions)))
            self.divisions = 2**self.depth
            self.dimSizes = [4 for i in range(self.depth)]

    def reset(self):
        if self.randomStart:
            self.x = random.uniform(self.xMin,self.xGoal)
            self.xdot = random.uniform(-self.xDotMax, self.xDotMax)
        else:
            self.x = -0.5
            self.xdot = 0
    
    def getRawState(self):
        return self.x, self.xdot

    def getQTree(self):
        v = np.zeros(self.depth)
        x = self.x
        xdot = self.xdot
        xBound = np.array([self.xMin,self.xGoal])
        xdBound = np.array([-self.xDotMax, self.xDotMax])
        for i, vi in enumerate(v):
            assert x<=xBound[1]
            assert x>=xBound[0]
            assert xdot<=xdBound[1], "xdB={} {},xdot={}".format(xdBound[0],xdBound[1],xdot,)
            assert xdot>=xdBound[0]
            xh, xdh = 0.5*np.sum(xBound), 0.5*np.sum(xdBound)
            x, xdot = x-xh, xdot-xdh
            xs, xds = rescale(x,1), rescale(xdot,2)
            v[i] = xs + xds
            xBound -= xh
            xdBound -= xdh
            xBound[1-xs]=0
            xdBound[1-xds/2]=0
        return v

    def moreThanHalf(self,a,aBound):
        """
        Returns 1 if a is halfway or more to the upper aBound
        and rescales the bounds to half their original size,
        keeping a between them.
        """
        assert a<=aBound[1]
        assert a>=aBound[0], "min = {}, a={}".format(aBound[0],a)
        h = 0.5*(aBound[0]+aBound[1])
        if a-h < 0:
            aBound[0] = aBound[0] - h
            aBound[1] = 0
            return 0
        else:
            aBound[1] = aBound[1] - h
            aBound[0] = 0
            return 1

    def getDiscState(self):
        x_index = (self.x-self.xMin)*(self.divisions-1)//(self.xGoal-self.xMin)
        xdot_index = (self.xdot + self.xDotMax)*(self.divisions-1)//(2*self.xDotMax)
        return int(x_index + self.divisions*xdot_index)

    def getAggrState(self):
        if self.x >= self.xGoal:
            return self.aggregation['term']
        x_index = math.floor((self.x-self.xMin)*(self.divisions-1)/(self.xGoal-self.xMin))
        xdot_index = math.floor((self.xdot + self.xDotMax)*(self.divisions-1)/(2*self.xDotMax))
        return self.aggregation[(xdot_index,x_index)]
 
    def getTileState(self):
        coords = (self.divisions/(self.xGoal-self.xMin)*self.x, self.divisions/(2*self.xDotMax)*self.xdot)
        return np.array(tiles.tiles(self.numtilings,self.ctable,coords))

    def getZeroQTable(self):
        if self.representation == 'raw':
            raise TypeError("Requesting state count for continuous valued representation")
        return np.zeros((self.nStates,len(self.actions)))

    def setAlpha(self,alpha):
        if self.representation == 'tile':
            return alpha/self.numtilings
        return alpha

    def bound_xdot(self):
        if abs(self.xdot)>self.xDotMax:
            self.xdot = np.copysign(self.xDotMax,self.xdot)
        
    def bound_x(self):
        if self.x<=self.xMin:
            self.xdot = 0
            self.x = self.xMin
        if self.x>=self.xGoal:
            self.xdot = 0
            self.x = self.xGoal


    def result(self,action):
        self.compute_result(action)
        if self.x >= self.xGoal:
            return 1, self.getAgentState(), 0
        return 0, self.getAgentState(), -1

    def compute_result(self,action):
        real_action = action-1

        grav = - self.grav * np.cos(self.x*3)
        
        self.xdot += (real_action + grav)*self.dt
        self.bound_xdot()
        self.x += self.xdot
        self.bound_x()


def rescale(n,i):
    if n == 0:
        return 1*i
    else:
        return 0.5*(n/abs(n)+1)*i
