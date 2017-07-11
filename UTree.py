"""
Implementation of McCallum's U-Tree algorithm

David Johnston 2015
"""

import numpy as np
from scipy.stats import ks_2samp
import random

NodeSplit = 0
NodeLeaf = 1
NodeFringe = 2
ActionDimension = -1


class UTree:

    def __init__(self,gamma,n_actions,dim_sizes,max_hist,max_back_depth=4,minSplitInstances=50,
                 significance_level=0.00005,is_episodic=0):

        self.node_id_count = 0
        self.root = UNode(self.genId(),NodeLeaf,None,n_actions)
        self.n_actions = n_actions
        self.max_hist = max_hist
        self.max_back_depth = max_back_depth
        self.gamma = gamma
        self.history = []
        self.n_dim = len(dim_sizes)
        self.dim_sizes = dim_sizes
        self.minSplitInstances = minSplitInstances
        self.significanceLevel = significance_level


        self.nodes = {self.root.idx : self.root}

        self.term = UNode(self.genId(),NodeLeaf,None,1) #dummy terminal node with 0 value
        self.nodes[self.term.idx] = self.term

            

    def print_tree(self):
        for i, node in self.nodes.items():
            if node.nodeType == NodeSplit:
                print("idx = {}, dis={}, back={}, par={}".format(node.idx,node.distinction.dimension,
                                                         node.distinction.back_idx,node.parent))
            else:
                print("idx={}, qvaules={}, par={}".format(node.idx,node.qValues,node.parent))

    def getTime(self):
        return len(self.history)

    def updateCurrentNode(self,instance):
        old_state = self.getLeaf()
        if old_state.idx == self.term.idx:
            return
        old_state.addInstance(instance,self.max_hist)

        self.insertInstance(instance)
        new_state = self.getLeaf()
        old_state.updateModel(new_state.idx,instance.action,instance.reward)

    def sweepLeaves(self):
        return self.sweepRecursive(self.root,self.gamma)

        
    def sweepRecursive(self,node,gamma):
        """ 
        Apply single step of value iteration to leaf node
        or recursively to children if it is a split node
        """
        tot = 0
        if node.nodeType == NodeLeaf:
            for a, r in enumerate(node.rewards):
                c = float(node.count[a])
                if c == 0: continue
                exp_utility = sum([self.nodes[node_to].utility()*t/c 
                                   for node_to, t in node.transitions[a].items()])

                node.qValues[a] = r/c + gamma*exp_utility
            return 1


        assert node.nodeType == NodeSplit
        
        for c in node.children:
            tot += self.sweepRecursive(c,gamma)
        return tot


    def insertInstance(self,instance):
        
        self.history.append(instance)
        #if len(self.history)>self.max_hist:
        #    self.history = self.history[1:]
 
    def nextInstance(self,instance):
        assert instance.timestep + 1 < len(self.history)
        return self.history[instance.timestep + 1]
       
    def transFromInstances(self, node, n_id, action):
        
        count = 0
        total = 0

        for inst in node.instances:
            if inst.action == action:
                leaf_to = getInstanceLeaf(ninst,previous=0)
                if leaf_to.idx == n_id:
                    c += 1
                    
                total += 1

        if total:
            return c/total
        else:
            return 0


    def rewardFromInstances(self, node, action):
        
        rtotal = 0
        total = 0

        for inst in node.instances:
            if inst.action == action:
                rew_total += inst.reward
                total += 1
        if total:
            return rew_total/total
        else:
            return 0

    def modelFromInstances(self, node):

        assert node.nodeType == NodeLeaf

        node.rewards = np.zeros(self.n_actions)
        node.count = np.zeros(self.n_actions)
        node.transitions = [{} for i in range(self.n_actions)]
        
        for inst in node.instances:
            leaf_from = self.getInstanceLeaf(inst)
            assert leaf_from == node
            leaf_to = self.getInstanceLeaf(inst,previous=0)
            node.updateModel(leaf_to.idx,inst.action,inst.reward)

    def getLeaf(self):
        """ Get leaf corresponding to current history """
        if len(self.history) == 0:
            return self.root
       
        idx = len(self.history) - 1
        node = self.root
        if self.history[idx].observation[0] == -1:
            #print "terminal state"
            return self.term

        while node.nodeType != NodeLeaf:
            assert node.nodeType == NodeSplit
            child = node.applyDistinction(self.history,idx)
            node = node.children[child]
        return node

    def getInstanceLeaf(self,inst,ntype=NodeLeaf,previous=1):
        """ Get leaf that inst records a transition from """

        idx = inst.timestep - previous

        if len(self.history)>0 and self.history[idx].observation[0] == -1 and idx >= 0:
            #print "terminal instance"
            return self.term


        node = self.root
        while node.nodeType != ntype:
            child = node.applyDistinction(self.history,idx)
            node = node.children[child]

        return node

    def genId(self):
        self.node_id_count += 1
        return self.node_id_count

    def split(self,node,distinction):
        assert node.nodeType == NodeLeaf
        assert distinction.back_idx >= 0


        node.nodeType = NodeSplit
        node.distinction = distinction

        # Add children
        if distinction.dimension == ActionDimension:
            for i in range(self.n_actions):
                idx = self.genId()
                n = UNode(idx,NodeLeaf,node,self.n_actions)
                n.qValues = np.copy(node.qValues)
                self.nodes[idx] = n
                node.children.append(n)

        else:
            for i in range(self.dim_sizes[distinction.dimension]):
                idx = self.genId()
                n = UNode(idx,NodeLeaf,node,self.n_actions)
                n.qValues = np.copy(node.qValues)
                self.nodes[idx] = n
                node.children.append(n)

        # Add instances to children
        for inst in node.instances:
            n = self.getInstanceLeaf(inst)
            assert n.parent.idx == node.idx, "node={}, par={}, n={}".format(node.idx,n.parent.idx,n.idx)
            n.addInstance(inst,self.max_hist)
        
        # Re-build model for all nodes
        for i, n in self.nodes.items():
            if n.nodeType == NodeLeaf:
                self.modelFromInstances(n)

        # Update Q-values for children
        for n in node.children:
            self.sweepRecursive(n,self.gamma)

    def splitToFringe(self,node,distinction):
        """
        Create fringe nodes instead of leaf nodes after splitting; these nodes 
        aren't used in the agent's model
        """
        assert distinction.back_idx >= 0

        node.distinction = distinction
                
        # Add children
        if distinction.dimension == ActionDimension:
            for i in range(self.n_actions):
                idx = self.genId()
                fringe_node = UNode(idx,NodeFringe,node,self.n_actions)
                node.children.append(fringe_node)

        else:
            for i in range(self.dim_sizes[distinction.dimension]):
                idx = self.genId()
                fringe_node = UNode(idx,NodeFringe,node,self.n_actions)
                node.children.append(fringe_node)

        # Add instances to children
        for inst in node.instances:
            n = self.getInstanceLeaf(inst,ntype=NodeFringe)
            assert n.parent.idx == node.idx, "idx={}".format(n.idx)
            n.addInstance(inst,self.max_hist)


    def unsplit(self,node):
        """
        Undo split operation; can delete leaf or fringe nodes.
        """
        if node.nodeType == NodeSplit:
            assert len(node.children) > 0
        
            node.nodeType = NodeLeaf
            node.distinction = None

            for c in node.children:
                del self.nodes[c.idx]

            
            # Re-build model for all nodes
            for i, n in self.nodes.items():
                if n.nodeType == NodeLeaf:
                    self.modelFromInstances(n)


        node.children = []
        
    def getBestAction(self):
        node = self.getLeaf()
        return random.choice(np.where(node.qValues == max(node.qValues))[0])

    def testFringe(self): # Tests fringe nodes for viable splits, splits nodes if they're found
        return self.testFringeRecursive(self.root)

    def testFringeRecursive(self,node):
        
        if len(node.instances) < self.minSplitInstances:
            return 0
        
        if node.nodeType == NodeLeaf or node.nodeType == NodeFringe:
            d = self.getUtileDistinction(node)

            if d:
                self.split(node,d)
                return 1 + self.testFringeRecursive(node)

            return 0


        assert node.nodeType == NodeSplit
        total = 0

        for c in node.children:
            total += self.testFringeRecursive(c)

        return total

    def getUtileDistinction(self,node):
        
        assert node.nodeType == NodeLeaf

        root_utils = self.getEFDRs(node)

        cds = self.getCandidateDistinctions(node)
        
        child_utils = []

        for cd in cds:
            self.splitToFringe(node,cd)
            #self.split(node,cd)
            for c in node.children:
                if len(c.instances) < self.minSplitInstances:
                    continue
                child_utils.append(self.getEFDRs(c))
            self.unsplit(node)
            
            for i, cu in enumerate(child_utils):
                k,p = ks_2samp(root_utils,cu)
                if p<self.significanceLevel:
                    print("KS passed, p={}, d = {}, back={}".format(p, cd.dimension, cd.back_idx))
                    #print(root_utils)
                    #print(cu)
                    return cd
                #elif p< 0.1:
                #    print("KS failed, p={}. d= {}, back={}".format(p,cd.dimension,cd.back_idx))


        return None

    def getEFDRs(self,node):
        """
        Get all expected future discounted returns for all instances in a node
        (q-value is just the average EFDRs)
        """
        efdrs = np.zeros(len(node.instances))
        for i, inst in enumerate(node.instances):
            next_state = self.getInstanceLeaf(inst,previous=0)
            next_state_util = next_state.utility()
            efdrs[i] = inst.reward + self.gamma*next_state_util
            
        return efdrs
        

    def getCandidateDistinctions(self,node):
        
        p = node.parent
        anc_distinctions = []

        while p:
            assert p.nodeType == NodeSplit
            anc_distinctions.append(p.distinction)
            p = p.parent


        candidates = []
        for i in range(self.max_back_depth):
            for j in range(-1,self.n_dim):
                d = Distinction(j,i)
                if d in anc_distinctions:
                    continue
                candidates.append(d)

        return candidates

                                



class UNode:

    def __init__(self,idx,nodeType,parent,n_actions):
        self.idx = idx
        self.nodeType = nodeType
        self.parent = parent

        self.children = []

        self.rewards = np.zeros(n_actions)
        self.count = np.zeros(n_actions)
        self.transitions = [{} for i in range(n_actions)]
        self.qValues = np.zeros(n_actions)
 

        self.instances = []
 
    def addInstance(self,instance,max_hist):
        assert(self.nodeType == NodeLeaf or self.nodeType == NodeFringe)
        self.instances.append(instance)
        if len(self.instances)>max_hist:
            self.instances=self.instances[1:]

    def updateModel(self,new_state,action,reward):
        self.rewards[action] += reward
        self.count[action] += 1
        if new_state not in self.transitions[action]:
            self.transitions[action][new_state] = 1
        else:
            self.transitions[action][new_state] += 1
      
    def applyDistinction(self,history,idx):
        assert self.nodeType != NodeFringe
        assert len(history) > self.distinction.back_idx
        assert len(history) > idx
        assert self.distinction.back_idx >= 0
        
        if idx == -1 or idx == 0:
            return 0

        # if back_idx is too far for idx, pick the first child
        if self.distinction.back_idx > idx:
            return 0
        
        inst = history[idx - self.distinction.back_idx]

        if self.distinction.dimension == ActionDimension:
            return inst.action
        
        assert self.distinction.dimension >= 0

        return inst.observation[self.distinction.dimension]

    def utility(self):
        return max(self.qValues)

class Instance:

    def __init__(self,timestep,action,observation,reward):
        self.timestep = int(timestep)
        self.action = int(action)
        self.observation = map(int,observation)
        self.reward = reward

class Distinction:

    def __init__(self,dimension,back_idx):
        self.dimension = dimension
        self.back_idx = back_idx
        
    def __eq__(self,distinction):
        return (self.dimension == distinction.dimension and self.back_idx == distinction.back_idx)



# def toEDF(samp,mini,maxi):
#     """
#     Convert list of samples to normalised empirical density function with 
#     10 divisions
#     """
#     edf = np.cumsum(np.histogram(samp,range=(mini,maxi))[0])
#     edf /= float(edf[-1])
#     return edf
    


# def KSTest(samp1, samp2):
#     """
#     Two-sample Kolmogorov-Smirnov test
#     """
#     mini = min(min(samp1),min(samp2))
#     maxi = max(max(samp1),max(samp2))
#     edf1, edf2 = toEDF(samp1,mini,maxi), toEDF(samp2,mini,maxi)

#     return max(np.abs(edf1-edf2))

