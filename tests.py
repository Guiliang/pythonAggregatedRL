"""
Test cases

David Johnston 2015
"""

import unittest
import numpy as np
import Agents
import Problems

def setupESA():
    """ Test problem from Hutter(2014) """

    gamma = 1/2.

    r0 = gamma/2/(1+gamma)
    r1 = (1+gamma/2)/(1+gamma)

    transitions = np.array([[[0   ,0.5 ,0.5 ,0],
                             [0.5 ,0   ,0   ,0.5],
                             [0   ,1.  ,0   ,0],
                             [1.  ,0   ,0   ,0]]])
    rewards = np.array([[[r0,r0,r0,r0],
                         [r1,r1,r1,r1],
                         [0 ,0 ,0 ,0],
                         [1 ,1 ,1 ,1]]])
    aggregation = {'n':2,0:0,1:1,2:0,3:1}

    qValues_raw = np.array([[gamma/(1-gamma**2),1/(1-gamma**2),
                             gamma/(1-gamma**2),1/(1-gamma**2)]]).reshape((4,1))
    qValues_agg = np.array([[gamma/(1-gamma**2),1/(1-gamma**2)]]).reshape((2,1))

    p_raw = Problems.MDP(0,transitions,rewards,gamma,qValues=qValues_raw)
    p_agg = Problems.MDP(0,transitions,rewards,gamma,qValues=qValues_raw,
                         aggregation=aggregation)

    return p_raw, p_agg

class TestMDPs(unittest.TestCase):

    def test_Q(self):

        p_raw, p_agg = setupESA()
        
        ql_raw = Agents.QAgent(p_raw,1)
        ql_agg = Agents.QAgent(p_agg,1)

        ql_raw.episode(timeout = 1000)
        ql_agg.episode(timeout = 1000)

        delta_r = sum(abs(ql_raw.qValues[0] - p_raw.qValues[0]))/4
        delta_a = sum(abs(ql_agg.qValues[0] - p_agg.qValues[0]))/2

        print("\nQ learning raw delta = {}, agg delta = {}".format(delta_r,delta_a))
        
        self.assertTrue(delta_r < 1e-1)
        self.assertTrue(delta_a < 1e-1)

    def test_SL(self):

        p_raw, p_agg = setupESA()

        sl_raw = Agents.SarsaLambda(p_raw,1)
        sl_agg = Agents.SarsaLambda(p_agg,1)

        sl_raw.episode(timeout = 1000)
        sl_agg.episode(timeout = 1000)

        delta_r = sum(sl_raw.qValues[0] - p_raw.qValues[0])/4
        delta_a = sum(sl_agg.qValues[0] - p_agg.qValues[0])/2

        print("\nSarsa(l) raw delta = {}, agg delta = {}".format(delta_r,delta_a))
        
        self.assertTrue(delta_r < 1e-1)
        self.assertTrue(delta_a < 1e-1)
                

    def test_VI(self):

        p_raw, _ = setupESA()

        vi_raw = Agents.VIAgent(p_raw,1)
        vi_raw.VISweep()

        delta_r = sum(vi_raw.qValues[0] - p_raw.qValues[0])/4

        print("\nValue Iteration raw delta = {}".format(delta_r))

        self.assertTrue(delta_r < 1e-4)


    def test_UT(self):

        p_raw, p_agg = setupESA()

        ut_raw = Agents.UTreeAgent(p_raw)

        ut_raw.episode(timeout = 1000)

        ut_raw.utree.print_tree()

if __name__ == '__main__':
    unittest.main()
