import sys
import math
import os
import os.path
import rlglue.RLGlue as RLGlue

def offlineDemo():
	this_score=evaluateAgent();
	printScore(0,this_score);
	theFile = open("results.csv", "w");
	theFile.close();
	if os.path.isfile("Archive.csv"):
		os.remove('Archive.csv')
	for i in range(0,200):
		for j in range(0,50):
			RLGlue.RL_episode(0);
			RLGlue.RL_env_message("stop print")
			if j%20==0 and i>0 :
				RLGlue.RL_env_message("print");
			printScore((i+1)*50,this_score);
		this_score=evaluateAgent();
		printScore((i+1)*50,this_score);
		theFile = open("results.csv", "a");
		theFile.write("%d\t%.2f\t%.2f\n" % ((i)*50, this_score[0], this_score[1]))
		theFile.close();
	os.rename('results.csv', 'Archive.csv')	

def printScore(afterEpisodes, score_tuple):
	print "%d\t\t%.2f\t\t%.2f" % (afterEpisodes, score_tuple[0], score_tuple[1])

#
# Tell the agent to stop learning, then execute n episodes with his current
# policy.  Estimate the mean and variance of the return over these episodes.
#
def evaluateAgent():
	sum=0;
	sum_of_squares=0;
	this_return=0;
	mean=0;
	variance=0;
	n=10;
	
	RLGlue.RL_agent_message("freeze learning");
	for i in range(0,n):
		# We use a cutoff here in case the 
		#policy is bad and will never end an episode
		RLGlue.RL_episode(5000);
		this_return=RLGlue.RL_return();
		sum+=this_return;
		sum_of_squares+=this_return**2;
	
	mean=sum/n;
	variance = (sum_of_squares - n*mean*mean)/(n - 1.0);
	standard_dev=math.sqrt(variance);

	RLGlue.RL_agent_message("unfreeze learning");
	return mean,standard_dev;



#
# Just do a single evaluateAgent and print it
#
def	single_evaluation():
	this_score=evaluateAgent();
	printScore(0,this_score);



print "Starting offline demo\n----------------------------\nWill alternate learning for 25 episodes, then freeze policy and evaluate for 10 episodes.\n"
print "After Episode\tMean Return\tStandard Deviation\n-------------------------------------------------------------------------"
RLGlue.RL_init()
offlineDemo()

print "\nNow we will save the agent's learned value function to a file...."

RLGlue.RL_agent_message("save_policy results.dat");

print "\nCalling RL_cleanup and RL_init to clear the agent's memory..."

RLGlue.RL_cleanup();
RLGlue.RL_init();

print "Evaluating the agent's default policy:\n\t\tMean Return\tStandardDeviation\n------------------------------------------------------"
single_evaluation();

print "\nLoading up the value function we saved earlier."
RLGlue.RL_agent_message("load_policy results.dat");

print "Evaluating the agent after loading the value function:\n\t\tMean Return\tStandardDeviation\n------------------------------------------------------"
single_evaluation();

RLGlue.RL_cleanup();
print "\nProgram Complete."




