import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import networkx as nx
import matplotlib.pyplot as plt


class Qlearning_agent(Agent):
	randGenerator=Random()
	lastAction=Action()
	lastObservation=Observation()
	sarsa_stepsize = 0.1
	sarsa_epsilon = 0.1
	sarsa_gamma = 1.0
	numberOfStates = 0
	numberOfActions = 0
	value_function = None
	
	policyFrozen=False
	exploringFrozen=False
	
	def agent_init(self,taskSpecString):
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
		if TaskSpec.valid:
			assert len(TaskSpec.getIntObservations())==1, "expecting 1-dimensional discrete observations"
			assert len(TaskSpec.getDoubleObservations())==0, "expecting no continuous observations"
			assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][0]), " expecting min observation to be a number not a special value"
			assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][1]), " expecting max observation to be a number not a special value"
			self.numberOfStates=TaskSpec.getIntObservations()[0][1]+1;

			assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
			assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
			self.numberOfActions=TaskSpec.getIntActions()[0][1]+1;
			
			self.value_function=[self.numberOfActions*[0.0] for i in range(self.numberOfStates)]

		else:
			print "Task Spec could not be parsed: "+taskSpecString;
			
		self.lastAction=Action()
		self.lastObservation=Observation()
		
	def egreedy(self, state):
		maxIndex=0
		a=1
		if not self.exploringFrozen and self.randGenerator.random()<self.sarsa_epsilon:
			return self.randGenerator.randint(0,self.numberOfActions-1)

                
		return self.value_function[state].index(max(self.value_function[state]))
		
		
	
	def agent_start(self,observation):
		theState=observation.intArray[0]
		thisIntAction=self.egreedy(theState)
		returnAction=Action()
		returnAction.intArray=[thisIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_step(self,reward, observation):
		newState=observation.intArray[0]
		lastState=self.lastObservation.intArray[0]
		lastAction=self.lastAction.intArray[0]

		Q_sa=self.value_function[lastState][lastAction]
		Q_sprime_aprime=-500000
		for a in range(self.numberOfActions):
			if self.value_function[newState][a]>Q_sprime_aprime:
				Q_sprime_aprime=self.value_function[newState][a]
		#updating Q function
		new_Q_sa=Q_sa + self.sarsa_stepsize * (reward + self.sarsa_gamma * Q_sprime_aprime - Q_sa)

		newIntAction=self.egreedy(newState)
		if not self.policyFrozen:
			self.value_function[lastState][lastAction]=new_Q_sa

		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_end(self,reward):
		lastState=self.lastObservation.intArray[0]
		lastAction=self.lastAction.intArray[0]

		Q_sa=self.value_function[lastState][lastAction]

		new_Q_sa=Q_sa + self.sarsa_stepsize * (reward - Q_sa)

		if not self.policyFrozen:
			self.value_function[lastState][lastAction]=new_Q_sa

	
	def agent_cleanup(self):
		pass

	def save_value_function(self, fileName):
		theFile = open(fileName, "w")
		pickle.dump(self.value_function, theFile)
		theFile.close()

	def load_value_function(self, fileName):
		theFile = open(fileName, "r")
		self.value_function=pickle.load(theFile)
		theFile.close()
	
	def agent_message(self,inMessage):
		
		#	Message Description
	 	# 'freeze learning'
		# Action: Set flag to stop updating policy
		#
		if inMessage.startswith("freeze learning"):
			self.policyFrozen=True
			return "message understood, policy frozen"

		#	Message Description
	 	# unfreeze learning
	 	# Action: Set flag to resume updating policy
		#
		if inMessage.startswith("unfreeze learning"):
			self.policyFrozen=False
			return "message understood, policy unfrozen"

		#Message Description
	 	# freeze exploring
	 	# Action: Set flag to stop exploring (greedy actions only)
		#
		if inMessage.startswith("freeze exploring"):
			self.exploringFrozen=True
			return "message understood, exploring frozen"

		#Message Description
	 	# unfreeze exploring
	 	# Action: Set flag to resume exploring (e-greedy actions)
		#
		if inMessage.startswith("unfreeze exploring"):
			self.exploringFrozen=False
			return "message understood, exploring frozen"

		#Message Description
	 	# save_policy FILENAME
	 	# Action: Save current value function in binary format to 
		# file called FILENAME
		#
		if inMessage.startswith("save_policy"):
			splitString=inMessage.split(" ");
			self.save_value_function(splitString[1]);
			print "Saved.";
			return "message understood, saving policy"

		#Message Description
	 	# load_policy FILENAME
	 	# Action: Load value function in binary format from 
		# file called FILENAME
		#
		if inMessage.startswith("load_policy"):
			splitString=inMessage.split(" ")
			self.load_value_function(splitString[1])
			print "Loaded."
			return "message understood, loading policy"

		return "SampleSarsaAgent(Python) does not understand your message."



if __name__=="__main__":
	AgentLoader.loadAgent(Qlearning_agent())
