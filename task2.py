"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np
import math

# START EDITING HERE
# You can use this space to define any helper functions that you need.
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        #print(self.num_arms,self.batch_size)
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.beta=np.zeros(num_arms)
        self.s=np.zeros(num_arms)
        self.f=np.zeros(num_arms)
    
    def give_pull(self):
        arms_topull={}
        b=1
        tol=self.batch_size*0.1
        max_p=np.max(self.values)/tol
        batch_beta=[]
        for i in range(0, self.num_arms):
            if self.values[i]>=max_p:
                arms_topull[i]=0
                batch_beta.append(self.beta[i])
        for i in range (0,self.batch_size):
            max_beta=np.argmax(batch_beta)
            arms_topull[list(arms_topull.keys())[max_beta]]+=1                    
            for j in range (0, len(batch_beta)):
                if j==max_beta:
                    batch_beta[j]=np.random.beta(self.s[list(arms_topull.keys())[j]]+1,self.f[list(arms_topull.keys())[j]]+1)
                else:
                    batch_beta[j]=np.random.beta(self.s[list(arms_topull.keys())[j]]+math.log(b*2.71828182845),self.f[list(arms_topull.keys())[j]]+1)
            b+=1
        return (list(arms_topull.keys()),list(arms_topull.values()))

    def get_reward(self, arm_rewards):
        for arm_index in arm_rewards:
            for i in range(0,len(arm_rewards[arm_index])):
                self.counts[arm_index] += 1
                n = self.counts[arm_index]
                value = self.values[arm_index]
                new_value = ((n - 1) / n) * value + (1 / n) * arm_rewards[arm_index][i]
                self.values[arm_index] = new_value
                if arm_rewards[arm_index][i]==1:
                    self.s[arm_index]+=1
                else:
                    self.f[arm_index]+=1
        for i in range (0, self.num_arms):
            self.beta[i]=np.random.beta(self.s[i]+1,self.f[i]+1)
        