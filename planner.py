#! /usr/bin/python
import random,argparse
parser = argparse.ArgumentParser()
import numpy as np
from pulp import *

class planner():
    def __init__(self,S,R,T,gamma,algorithm,numStates,numActions,policy=np.array([])):
        if (policy.size != 0):
            self.policy_val(R,T,policy,gamma,numStates,truth=True)
        else:
            if algorithm=='vi':
                self.value_iteration(S,R,T,gamma,numStates,numActions)
            elif algorithm=='hpi':
                self.howard_policy(S,R,T,gamma,numStates,numActions)
            else :
                self.linear_program(R,T,gamma,numStates,numActions)

    def value_iteration(self,S,R,T,gamma,numStates,numActions):
        vf=np.random.rand(numStates)
        tol=1e-25
        vt=np.zeros(numActions)
        vi=np.zeros(numStates)
        A=np.zeros(numStates)
        while (np.max(np.absolute(vf-vi))>tol):
            for s in range(0,numStates):
                vi[s]=vf[s]
                for a in range(0,numActions):
                    vt[a]=np.sum(T[s][a]*(R[s][a]+(gamma*vi)))
                #print(vt)
                vf[s]=np.max(vt)
                A[s]=np.argmax(vt)
        
        for s in range(0,numStates):
            print(vf[s],' ', A[s])

    def policy_val(self,R,T,policy,gamma,numStates,truth=False):
        vf=np.random.rand(numStates)
        vi=np.zeros(numStates)
        tol=1e-20
        while (np.max(np.absolute(vf-vi))>tol):
            for s in range (0,numStates):
                vi[s]=vf[s]
                a=policy[s]
                vf[s]=np.sum(T[s][a]*(R[s][a]+(gamma*vi)))
        if (truth==True):
            for s in range(0,numStates):
                print(vf[s],' ', policy[s])
        else:
            return vf

    def howard_policy(self,S,R,T,gamma,numStates,numActions):
        policy=np.random.randint(numActions, size=numStates)
        #vpi=self.policy_val(R,T,policy,gamma,numStates,numActions)
        Q=np.zeros((numStates,numActions))
        #improv_states=np.zeros((numStates,numActions))
        while True:
            vpi=self.policy_val(R,T,policy,gamma,numStates)
            improv_states=np.zeros((numStates,numActions))
            #print(vpi)
            for s in range(0, numStates):
                for a in range(0, numActions):
                    Q[s][a]=np.sum(T[s][a]*(R[s][a]+(gamma*vpi)))
                improv_states[s][(Q[s]-vpi[s])>1e-6]=1
                #print(Q[s])
                #print(improv_states)
                a, = np.nonzero(improv_states[s])
                #print(a)
                if (a.size != 0):
                    policy[s]=np.random.choice(a)
                #print("policy", policy)
            #print(improv_states)
            if  not(improv_states.any()) :
                break
        vpi=self.policy_val(R,T,policy,gamma,numStates,truth=True)

    def linear_program(self,R,T,gamma,numStates,numActions):
        prob= LpProblem("myProb", LpMaximize)
        v=np.zeros(numStates,dtype=object)
        vpi=np.zeros(numStates)
        Q=np.zeros((numStates,numActions))
        for s in range(0, numStates):
            variable = str('x'+str(s))
            variable = pulp.LpVariable(variable)
            v[s]=variable

        prob += pulp.LpAffineExpression((v[s],-1) for s in range(0,numStates))
        for s in range(0, numStates):
            for a in range(0, numActions):
                #pulp.LpAffineExpression([(x, k) for x in xs] + [(y, 1.0) for y in ys])
                prob += v[s]>=np.sum(T[s][a]*R[s][a])+pulp.LpAffineExpression((v[s1],gamma*T[s][a][s1]) for s1 in range(0,numStates))
                #prob += (v[s]>=np.sum(T[s][a]*(R[s][a]+(gamma*v))))
        status =prob.solve(PULP_CBC_CMD(msg=False))

        for v in prob.variables():
            i= int(v.name[1:])
            vpi[i]=v.varValue
        for s in range(0, numStates):
                for a in range(0, numActions):
                    Q[s][a]=np.sum(T[s][a]*(R[s][a]+(gamma*vpi)))
        for s in range(0,numStates):
                print(vpi[s],' ', np.argmax(Q[s]))

if __name__ == "__main__":
    parser.add_argument("--mdp",required=True,help='mdp path')
    parser.add_argument("--algorithm",type=str,default='vi',help='vi, hpi, lp')
    parser.add_argument("--policy",default=False)
    
    args = parser.parse_args()
    myFile1 = args.mdp
    text = open(myFile1)
    t_line=text.readline().split(" ")
    numStates=int(t_line[1])
    t_line=text.readline().split(" ")
    numActions=int(t_line[1])
    transitions=np.zeros((numStates, numActions, numStates),dtype=object)
    rewards=np.zeros((numStates, numActions, numStates),dtype=object)
    states=np.zeros((numStates, numActions, numStates),dtype=object)
    t_line=text.readline().split(" ")
    endStates=list(map(int, t_line[1:]))
    i=''
    while i!='discount':
        t_line=text.readline().split(" ")
        if t_line[0]=='transition':
            raw_transitions=list(map(float, t_line[1:])) 
            transitions[int(raw_transitions[0])][int(raw_transitions[1])][int(raw_transitions[2])]=raw_transitions[4]
            rewards[int(raw_transitions[0])][int(raw_transitions[1])][int(raw_transitions[2])]=raw_transitions[3]
            states[int(raw_transitions[0])][int(raw_transitions[1])][int(raw_transitions[2])]=raw_transitions[2]
        elif t_line[0]=='discount':
            i='discount'
            discount=float(t_line[2])
        else:
            mdptype=t_line[1]

    #print(transitions[:][2][1])
    if (args.policy != False):
        myFile2 = args.policy
        text=open(myFile2)
        Policy=np.zeros(numStates, dtype=int)
        for s in range (0,numStates):
            Policy[s]=int(text.readline())

        algo = planner(states,rewards,transitions,discount,args.algorithm,numStates,numActions,policy=Policy)
    else:
        algo = planner(states,rewards,transitions,discount,args.algorithm,numStates,numActions)


