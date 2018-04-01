import random
import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from collections import defaultdict
from random import Random


class skeleton_agent(Agent):

    lastAction = Action()
    lastObservation = Observation()

    R = defaultdict(lambda: defaultdict(lambda: 0.0))
    T = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

    C_s_a_s = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    C_s_a = defaultdict(lambda: defaultdict(lambda: 0))
    rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
    rmax = 1.0

    m = 10
    Q = defaultdict(lambda: defaultdict(lambda:0.1))

    numS = 0
    numA = 0

    sarsa_epsilon=0.4
    sarsa_stepsize = 0.5
    sarsa_gamma = 0.9

    


    def agent_init(self, taskSpec):
        # See the sample_sarsa_agent in the mines-sarsa-example project for how to parse the task spec
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
        if TaskSpec.valid:
            assert len(TaskSpec.getIntObservations()) == 1, "expecting 1-dimensional discrete observations"

            assert len(TaskSpec.getDoubleObservations()) == 0, "expecting no continuous observations"

            assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][0]), " expecting min observation to be a number not a special value"

            assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][1]), " expecting max observation to be a number not a special value"

            self.numStates = TaskSpec.getIntObservations()[0][1] + 2;

            assert len(TaskSpec.getIntActions()) == 1, "expecting 1-dimensional discrete actions"

            assert len(TaskSpec.getDoubleActions()) == 0, "expecting no continuous actions"

            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"

            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"

            self.numActions = TaskSpec.getIntActions()[0][1] + 1;
            
            #self.value_function = [self.numActions * [0.0] for i in range(self.numStates)]

        else:
            print "Task Spec could not be parsed: " + taskSpec;

        self.lastAction = Action()
        self.lastObservation = Observation()

        S0 = TaskSpec.getIntObservations()[0][1]+1
        for a in range(self.numActions):
            self.R[S0][a] = self.rmax
            self.T[S0][a][S0] = 1.0





    def agent_start(self, observation):
        newState = observation.intArray[0]
        x = self.egreedy(newState)
        returnAction = Action()
        returnAction.intArray = [x]

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)

        return returnAction


    def agent_step(self, reward, observation):
        #x = agentPolicy()
        #y = adversaryPolicy()

        newState = observation.intArray[0]
        lastState = self.lastObservation.intArray[0]
        lastActionAgent = self.lastAction.intArray[0]


        self.C_s_a[lastState][lastActionAgent]+=1
        self.C_s_a_s[lastState][lastActionAgent][newState]+=1
        self.rsum[lastState][lastActionAgent]+= reward


        if self.C_s_a[lastState][lastActionAgent] >= self.m:
            self.R[lastState][lastActionAgent] = self.rsum[lastState][lastActionAgent] / self.C_s_a[lastState][lastActionAgent]
            for sr in range(self.numStates):
                if self.C_s_a_s[lastState][lastActionAgent][sr] > 0:
                    self.T[lastState][lastActionAgent][sr] = self.C_s_a_s[lastState][lastActionAgent][sr] / self.C_s_a[lastState][lastActionAgent]
        else:
            self.R[lastState][lastActionAgent] = self.rmax
            self.T[lastState][lastActionAgent][newState] = 1.0

        Q_sa=self.Q[lastState][lastActionAgent]
        Q_sprime_aprime=-500000
        for a in range(self.numActions):
        	if self.Q[newState][a]>Q_sprime_aprime:
        		Q_sprime_aprime=self.Q[newState][a]
        new_Q_sa=Q_sa + self.sarsa_stepsize * (self.R[lastState][lastActionAgent] + self.sarsa_gamma * Q_sprime_aprime - Q_sa)
        newIntAction=self.egreedy(newState)
        self.Q[lastState][lastActionAgent]=new_Q_sa


        x = self.egreedy(newState)
        returnAction = Action()
        returnAction.intArray = [x]
        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)
        return returnAction

    def agent_end(self,reward):
        lastState=self.lastObservation.intArray[0]
        lastAction=self.lastAction.intArray[0]
        Q_sa=self.Q[lastState][lastAction]
        new_Q_sa=Q_sa + self.sarsa_stepsize * (reward - Q_sa)
        self.Q[lastState][lastAction]=new_Q_sa

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        pass

    def egreedy(self, state):
        maxIndex=0
        a=1
        maxValue=-500000
        OptimalAction=0
        #if not self.randGenerator.random()<self.sarsa_epsilon:
        # #return self.randGenerator.randint(0,self.numActions-1)
        for a in range(self.numActions):
        	if maxValue<self.Q[state][a]:
                    maxValue=self.Q[state][a]
                    OptimalAction=a
        return OptimalAction

if __name__ == "__main__":
    AgentLoader.loadAgent(skeleton_agent())
