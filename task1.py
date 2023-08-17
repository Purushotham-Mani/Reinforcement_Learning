"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb=np.zeros(num_arms)
        self.time=1
    
    def give_pull(self):
        while self.time<=self.num_arms:
            return self.time-1
        else:
            return np.argmax(self.ucb)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        if self.time>self.num_arms:
            for i in range (0, self.num_arms):
                self.ucb[i]=self.values[i]+(math.sqrt(2*math.log(self.time)/self.counts[i]))
        self.time+=1

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.klucb=np.zeros(num_arms)
        self.time=1
    
    def give_pull(self):
        while self.time<=self.num_arms:
            return self.time-1
        else:
            return np.argmax(self.klucb)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        if self.time>self.num_arms:
            for i in range (0, self.num_arms):
                mx=1
                mn=self.values[i]
                p=mn
                q=p
                n=self.counts[i]
                while (mx-mn)>0.01:
                    q=(mn+mx)/2
                    if (p*math.log((p/q)+1e-9)+(1-p)*math.log(((1-p)/(1-q))+1e-9)-(math.log(self.time)+3*math.log(math.log(self.time)))/(n))<0:
                        mn=q
                    else:
                        mx=q        
                self.klucb[i]=q     
        self.time+=1


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.beta=np.zeros(num_arms)
        self.s=np.zeros(num_arms)
        self.f=np.zeros(num_arms)
        self.time=1
    
    def give_pull(self):
        if self.time==1:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.beta)

    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        if reward==1:
            self.s[arm_index]+=1
        else:
            self.f[arm_index]+=1
        for i in range (0, self.num_arms):
            self.beta[i]=np.random.beta(self.s[i]+1,self.f[i]+1)
        #print(self.beta)
        self.time+=1
