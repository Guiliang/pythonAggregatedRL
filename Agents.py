'''
Reinforcement learning agents.



David Johnston 2015
'''

import Problems
import numpy as np
import collections
import numbers
import random
import UTree

random.seed(1)

class OnlineAgent:
    """
    Generic online agent class; executes e-greedy policy, looks up values
    """
    
    def __init__(self,problem,epsilon=1e-1,tiles=False):
        self.epsilon = epsilon
        self.problem = problem
        self.qValues = problem.getZeroQTable()
        self.reset = self.problem.reset
        if tiles:
            self.getQValue = self.getQTile
        else:
            self.getQValue = self.getQDisc
        
        
    def executePolicy(self, state ,tiebreak='first'):

        qs = self.getQArray(state)

        test = random.random()
        if test < self.epsilon:
            return random.choice(range(len(qs)))
        elif tiebreak == 'first':
            return np.where(qs==max(qs))[0][0]
        elif tiebreak == 'random':
            return random.choice(np.where(qs==max(qs))[0])        

    
    def episode(self,deltaMin=1e-3,timeout=int(1e5),decayAlpha=True):
        '''
        Runs an episode, updates q-values and returns the length of the episode.
        '''
        
        for i in range(timeout):

            currentState = self.problem.getAgentState()
            
            action = self.executePolicy(currentState)

            self.preUpdate(currentState,action)
                
            if self.problem.isEpisodic:
                terminal, nextState, reward = self.problem.result(action)
                
                if terminal:

                    self.update(currentState,nextState,action,reward,decayAlpha,
                                terminal=1)
                    self.problem.reset()
                    return i
            else:
                
                nextState, reward = self.problem.result(action)
                

            self.update(currentState,nextState,action,reward,decayAlpha)

        return timeout
    
    def run_n_episodes(self,n,decayAlpha=False,timeout=int(1e5)):
        e_lengths = []
        e_avgs = np.zeros(int(np.log2(n)))
        j = 1

        for i in range(n):
            l = self.episode(timeout=timeout,decayAlpha=decayAlpha)
            if l<timeout:
                e_lengths.append(l)
                if i == 2**j:
                    s = min(1000,(len(e_lengths)+1)/2)
                    e_avgs[j-1]= np.average(e_lengths[-s:-1])
                    print(np.average(e_lengths[-s:-1]))
                    j += 1

            else:
                e_lengths.append(timeout)
                self.reset()
                print("Episode timed out {}".format(l))
        return e_avgs

    def getQDisc(self,state,action):
        return self.qValues[state,action]
        
    def getQTile(self,state,action):
        return sum(self.qValues[state,action])

    
    def getValue(self,state):
        qValues = self.getQArray(state)
        return max(qValues)

    def getQArray(self,state):
        return np.array([self.getQValue(state,a) for a in self.problem.actions])

    def approxQValues(self,n):
        """
        Hack for getting q-value estimates in an array
	from the mountain car problem with tiles representation
        """
	assert isinstance(self.problem, Problems.MountainCar)
	assert self.problem.representation == 'tile'
        qv = np.zeros((n,n,3))
        self.problem.xdot = -self.problem.xDotMax
        dx = abs(self.problem.xGoal-self.problem.xMin)/(n-1)
        dxdot = abs(2*self.problem.xDotMax)/(n-1)
        for i, vi in enumerate(qv):
            self.problem.x = self.problem.xMin
            for j, vj in enumerate(vi):
                t = self.problem.getAgentState()
                qv[i] = self.getQArray(t)
                self.problem.x += dx
            self.problem.xdot += dxdot

        self.problem.reset()
        return qv

        
class QAgent(OnlineAgent):
    """
    Q-learning agent 
    """
    def __init__(self,problem,alpha=1e-1,
                 epsilon=1e-1,tiles=False):
        OnlineAgent.__init__(self,problem,epsilon=epsilon,tiles=tiles)
        self.alpha = problem.setAlpha(alpha)
        self.counter = problem.getZeroQTable()
        
    def update(self,state,nextState,action,reward,decayAlpha,terminal=0):
        '''
        Q-learning update. State is either an integer or list(array) of integers
        '''
        if terminal:
            nextV = 0
        else:
            nextV = self.getValue(nextState)

        currentQV = self.getQValue(state,action)

        delta = reward - currentQV + self.problem.gamma*nextV
        #print "delta={}, reward={}, currentQV = {}, nextV = {}".format(delta,reward,currentQV, nextV)
        #print nextV, self.getQValue(nextState,0)
        if decayAlpha:
            alpha =  self.alpha/(self.counter[state,action]+1)
        else:
            alpha = self.alpha

        #print "a*d={}".format(alpha*delta)

        self.qValues[state,action] += alpha * delta
        self.counter[state,action] += 1

    def preUpdate(self,state,action):
        return

        
class SarsaLambda(OnlineAgent):
    """
    SARSA with eligibility traces
    """
    def __init__(self,problem,alpha,lamda=0.5,policy='e-greedy',
                 epsilon=1e-1,tiles=False):
        OnlineAgent.__init__(self,problem,epsilon=epsilon,tiles=tiles)
        self.alpha = problem.setAlpha(alpha)
        self.e = problem.getZeroQTable()
        self.counter = problem.getZeroQTable()
        self.lamda = lamda

    def reset(self):
        self.problem.reset
        self.e = problem.getZeroQTable()

    def preUpdate(self,state,action):
        self.e *= self.problem.gamma*self.lamda

        for a in self.problem.actions:
            if a == action:
                self.e[state,a] = 1
            else:
                self.e[state,a] = 0

    def update(self,state,nextState,action,reward,decayAlpha,terminal=0):
        '''
        Sarsa(Lambda) update
        '''
        nextAction = self.executePolicy(nextState)
        if terminal:
            nextV=0
        else:
            nextV = self.getQValue(nextState,nextAction)
        
        delta = reward - self.getQValue(state,action)
        delta += self.problem.gamma*nextV


        if decayAlpha:
            alpha =  self.alpha*((self.counter[state]+1)**(-1))
        else:
            alpha = self.alpha

        
        self.counter[state,action] += 1
        self.qValues += delta*alpha*self.e



        

class VIAgent(): 
    """
    Offline value iteration agent
    """

    def __init__(self,problem, policy="e-greedy",epsilon=1e-1,timeout=int(1e6)):
        '''
        Must be initialised with a problem with known transition and reward matrices
        '''

        self.problem = problem
        self.epsilon = epsilon
        self.qValues = problem.getZeroQTable()
        self.transitionMatrix = problem.transitions
        self.rewardMatrix = problem.rewards
        self.timeout = timeout
        #if policy == "e-greedy":
        self.policyMatrix = np.zeros(self.qValues.shape) + 1/self.qValues.shape[0]

    def executePolicy(self, state, epsilon=1e-1,tiebreak='random'):

        qs = self.getQArray(state)
            
        test = random.random()
        if test < epsilon:
            return random.choice(range(len(qs)))
        elif tiebreak == 'first':
            return np.where(qs==max(qs))[0][0]
        elif tiebreak == 'random':
            return random.choice(np.where(qs==max(qs))[0])        

        
    def getQValue(self,state,action):
        '''
        Get Q(s,a). S may be either an integer of list of ints if 
        function approximation is used.
        '''

        if isinstance(state,collections.Container):
            state=np.array(state)
            return sum(self.qValues[state,action])
        return self.qValues[state,action]

    
    def getValue(self,state):
        qValues = self.getQArray(state)
        return max(qValues)

    def getQArray(self,state):
        return np.array([self.getQValue(state,a) for a in self.problem.actions])

            
    def greedifyPolicy(self,epsilon=1e-1):
        
        old_policy = self.policyMatrix

        self.policyMatrix = np.full_like(self.policyMatrix,epsilon/self.qValues.shape[0])

        for state, policy in enumerate(self.policyMatrix):
            policy_choice = self.executePolicy(state,epsilon=0)
            policy[policy_choice] += 1-epsilon
            
        if (self.policyMatrix == old_policy).all():
            return 1
        else:
            return 0

    def VISweep(self):
        
        while True:
            self.evalPolicy()

            if self.greedifyPolicy(): 
                break

    def evalPolicy(self, deltaMin=1e-5):
        
        delta = float('inf')
        counter = 0

        while delta>deltaMin and counter<self.timeout:
            delta = 0
            for state, aValues in enumerate(self.qValues):
                for action, action_value in enumerate(aValues):
                    temp = action_value
                    states = range(len(self.qValues))
                    new_values = [self.transitionMatrix[action,state,nstate]* 
                                  (self.rewardMatrix[action,state,nstate]+
                                  self.problem.gamma*self.getValue(nstate))
                                 for nstate in states ]
                    new_action_value = sum(new_values)
 
                    self.qValues[state,action] = new_action_value
                    delta = max(delta, abs(temp-new_action_value))
                    counter += 1
            if counter >= self.timeout-1:
                print("Value iteration did not converge, delta = {}".format(delta))




class UTreeAgent:
    """
    Agent that implements McCallum's U-Tree algorithm
    """
    
    def __init__(self, problem, max_hist=1e3, epsilon=1e-1, check_fringe_freq = 1000,
                 eps_decay=1,is_episodic=0):
        
        self.utree = UTree.UTree(problem.gamma,len(problem.actions),
                                 problem.dimSizes,max_hist,is_episodic=is_episodic)
        self.problem = problem
        self.cff = check_fringe_freq
        self.decayEpsilon = eps_decay
        self.epsilon = epsilon
    
    def update(self, nextObs, action, reward, terminal=0, check_fringe=0):
        t = self.utree.getTime()
        i = UTree.Instance(t,action,nextObs,reward)

        self.utree.updateCurrentNode(i)
        self.utree.sweepLeaves() # every timestep might be too slow?
        
        if check_fringe:
            self.utree.testFringe()

        
    def executePolicy(self, epsilon=1e-1):
        test = random.random()
        if test<epsilon:
            return random.choice(range(len(self.problem.actions)))
        return self.utree.getBestAction()

    def episode(self,timeout=int(1e5)):
        act_hist = np.zeros(len(self.problem.actions))

        for i in range(timeout):

            if self.decayEpsilon:
                epsilon = min(1,10./len(self.utree.nodes))
            else: epsilon = self.epsilon

            action = self.executePolicy(epsilon)
            
            act_hist[action] += 1
            if self.problem.isEpisodic:
                terminal, nextObs, reward = self.problem.result(action)

                if terminal:
                    nextObs = [-1]
                    self.update(nextObs,action,reward)
                    self.problem.reset()
                    return i


                if isinstance(nextObs, numbers.Number):
                    nextObs = [nextObs]

            else:
                nextObs, reward = self.problem.result(action)
                
                if isinstance(nextObs, numbers.Number):
                    nextObs = [nextObs]


            if i % self.cff == 0:
                #print("checking fringe {}, a={}".format(i,act_hist))
                self.update(nextObs,action,reward,check_fringe=1)

            self.update(nextObs,action,reward)
            
        
