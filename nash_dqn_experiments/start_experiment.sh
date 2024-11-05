#!/bin/bash

# Set the number of times you want to run the script
count=2

# Create a new tmux session
tmux new-session -d -s nash_dqn

# Run the Python script the specified number of times
for i in $(seq 1 $count); do
    echo "Running experiment set #$i"
    tmux send-keys -t nash_dqn "./run_nash_dqn.sh" C-m
    tmux wait-for script_finished
done

# Close the tmux session
tmux kill-session -t nash_dqn

echo "All experiments completed!"
