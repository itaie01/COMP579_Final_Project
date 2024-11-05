import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v3
from tqdm import tqdm
import json
from DQNAgent import *


def eps_greedy_act_selection(epsilon, action_space_size, q_values):
    if np.random.random() < epsilon:
        # randomly choose one action
        return np.random.randint(0, action_space_size)
    else:
        # all q values
        return np.argmax(q_values)

def normalization(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
    
        # Normalize the features using the mean and standard deviation
        X_norm = (X - mu) / sigma

        return X_norm
    
AGENT_NAMES = ['speaker_0', 'listener_0']
AGENT_INFOS = {
    name:
        {
            "state_space_size": 3 if name == 'speaker_0' else 11, 
            "action_space_size": 3 if name == 'speaker_0' else 5 
        } for name in AGENT_NAMES
}

def dqn_sl(epsilon, num_episode, max_cycles, env):    
    # initialize DQNAgents for the listener / speaker
    all_dqn_agents = {name: DQNAgent(AGENT_INFOS[name]["state_space_size"], 
                                     AGENT_INFOS[name]["action_space_size"]) for name in AGENT_NAMES}
    
    all_rewards = {agent_name:[] for agent_name in AGENT_NAMES}
    
    for episode in tqdm(range(1, num_episode + 1), ascii=True, unit='episodes'):
    # for episode in range(num_episode):
        # initialize environment & reward
        env.reset()
        episode_agent_reward = {agent_name:0 for agent_name in AGENT_NAMES}
        
        for agent in env.agent_iter():
            # do not do step if terminated 
            if env.truncations[agent] == True or env.terminations[agent] == True:
                    env.step(None)
                    continue    
            # storing size of action and state (metadata about agent)
            agent_info = AGENT_INFOS[agent]
            # store the corresponding dqn agent (not the game agent, the agent that does dqn stuff)
            dqn_agent = all_dqn_agents[agent]
            # actual agent living in environment
            game_agent_cur_state = env.observe(agent)
            
            game_agent_q_val = dqn_agent.get_qs(game_agent_cur_state)
            game_agent_action_size = agent_info["action_space_size"]
            action_taken = eps_greedy_act_selection(epsilon, game_agent_action_size, game_agent_q_val)
            
            # take the action choosen from eps greedy
            env.step(action_taken)
            
            # get reward and accumulate it
            _, R, termination, truncation, info = env.last()
            done = termination or truncation
            episode_agent_reward[agent] += R
            
            # get next state S'
            game_agent_next_state = env.observe(agent)
            
            # update replay memory, and train if we have enough replay memory
            dqn_agent.update_replay_memory((game_agent_cur_state, action_taken, R, game_agent_next_state, done))
            dqn_agent.train(done)
        
        # store the total rewards for last game play in one episode
        for name in AGENT_NAMES:
            all_rewards[name].append(episode_agent_reward[name]/max_cycles)
        
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
             
    return all_rewards

# run and plot dqn data
env = simple_speaker_listener_v3.env(max_cycles=MAX_CYCLES, continuous_actions=False)
R = dqn_sl(EPSILON, EPISODES, MAX_CYCLES, env)

print("Experiment done, saving data file...")
with open(f"dqn_data_ep{EPISODES}_cycles_{MAX_CYCLES}.json", "w") as f:
    # store the experiement result
    json.dump(R, f)