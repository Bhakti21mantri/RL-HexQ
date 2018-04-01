rl_glue &
pid1=$!
python My_environment.py &
python My_QLearning_Agent.py &
python My_experiment.py &
while [ ! -s ./results.csv ]; do sleep 1; done
gnuplot LivePlot.gnu


