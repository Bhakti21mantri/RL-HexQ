import os
import time
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
sys.path.append('/home/bhakti/Study Time/Projects/RL-HEXQ/RL-HexQ')
from gym.envs.toy_text import taxi

class My_Environment(taxi.TaxiEnv):

	toprint = 0
	def env_init(self):
		taxi.TaxiEnv.__init__(self)
		return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1 OBSERVATIONS INTS (0 499) ACTIONS INTS (0 5) REWARDS (-10.0 20.0) EXTRA TaxiEnv by Us."
	
	# () -> Observation
 	def env_start(self):
		self.seed()
		self.reset()
		#self.seps=0
		
		returnObs=Observation()
		returnObs.intArray=[self.s]
		return returnObs
	
	# (Action) -> Reward_observation_terminal
	def env_step(self,action):
		
		assert len(action.intArray)<=2,"Expected 1 integer action."
		assert action.intArray[0]>=0, "Expected action to be in [0,5]"
		assert action.intArray[0]<6, "Expected action to be in [0,5]"
		s1, r1, d1, k1=self.step(action.intArray[0])
		returnRO=Reward_observation_terminal()
		returnRO.r=r1*1.0
		returnRO.o=Observation()
		returnRO.o.intArray=[s1]
		returnRO.terminal=d1
		if self.toprint == 1:
			self.clearscreen()
			x = taxi.TaxiEnv.render(self,taxi.TaxiEnv.metadata['render.modes'][0])
			print x
			time.sleep(0.08)
			
		#self.seps=self.seps+1
		#if self.seps >50:
		#	returnRO.terminal=TRUE
		return returnRO
	
	# () -> void
	def env_cleanup(self):
		pass
	
	# (string) -> string
	
	def env_message(self,inMessage):
		if inMessage.startswith("print"):
			self.toprint = 1
			return "message understood, print"
		if inMessage.startswith("stop print"):
			self.toprint = 0
			return "message understood, stop print"
		return "RmaxAgent(Python) does not understand your message."
	
	def clearscreen(self,numlines=100):
	  """Clear the console.
	numlines is an optional argument used only as a fall-back.
	"""
	# Thanks to Steven D'Aprano, http://www.velocityreviews.com/forums
	
	  if os.name == "posix":
	    # Unix/Linux/MacOS/BSD/etc
	    os.system('clear')
	  elif os.name in ("nt", "dos", "ce"):
	    # DOS/Windows
	    os.system('CLS')
	  else:
	    # Fallback for other operating systems.
	    print('\n' * numlines)	

if __name__=="__main__":
	EnvironmentLoader.loadEnvironment(My_Environment())
