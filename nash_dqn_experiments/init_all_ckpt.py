import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from pettingzoo.mpe import simple_speaker_listener_v3
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
import os
import json
import pickle

#Global Variables
DISCOUNT = 0.99
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# Environments
EPISODES = 50
MAX_CYCLES = 25

REPLAY_MEMORY_SIZE = 3000  # How many last steps to keep for model training
CRITIC_MIN_REPLAY_MEMORY_SIZE = 500
AGENT_MIN_REPLAY_MEMORY_SIZE = 800 # Minimum number of steps in a memory to start training

# Exploration settings
EPSILON = 0.2  # decaying epsilon
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Checkpointing
checkpoint_path_sp = "./models/speaker"
checkpoint_path_ls = "./models/listener"
checkpoint_path_critic = "./models/critic"
checkpoint_dir_sp = os.path.dirname(checkpoint_path_sp)
checkpoint_dir_ls = os.path.dirname(checkpoint_path_ls)
checkpoint_dir_critic = os.path.dirname(checkpoint_path_critic)

cp_callback_sp = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_sp, 
    verbose=0, 
    save_weights_only=True,
    save_freq=50)

cp_callback_ls = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_ls, 
    verbose=0, 
    save_weights_only=True,
    save_freq=50)

cp_callback_critic = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_critic, 
    verbose=0, 
    save_weights_only=True,
    save_freq=50)


class ReplayMemory:
    def __init__(self, max_len = REPLAY_MEMORY_SIZE):
        self.replay_memory = deque(maxlen=max_len)
        # with open('./models/replay_buffer/my_deque.pickle', 'rb') as f:
        #     # use pickle.load to deserialize the deque object from the file
        #     self.replay_memory = pickle.load(f)

    def add_sample(self, sample):
        # sp = speaker, ls = listener, the format of a sample is:
        # (S_sp, S_ls, A_sp, A_ls, R_sp, R_ls, S_next_sp, S_next_ls, done_sp, done_ls)
        self.replay_memory.append(sample)

    def get_size(self):
        return len(self.replay_memory)
    
    def sample_minibatch(self, minibatch_size = MINIBATCH_SIZE):
        return random.sample(self.replay_memory, minibatch_size)
    
    def get_mem(self):
        return self.replay_memory
    

class NashDQNAgent:
    def __init__(self, input_layer_size, action_space_size, ReplayMemoryObject, is_speaker, critic):
        # Check if this agent is a speaker, if not then listener
        self.is_speaker = is_speaker

        # Main model which we use to train
        self.model = self.create_model(input_layer_size, action_space_size)

        # Target network to make sure the updating is stable
        self.target_model = self.create_model(input_layer_size, action_space_size)
        self.target_model.set_weights(self.model.get_weights())

        # The array to keep the memory for the last n steps for training
        self.replay_memory = ReplayMemoryObject

        # Count when to update target network with main network's weights
        self.target_update_counter = 0

        # Add the critic -- a centalized network to give Q values for joint actions by inputing joint observations
        self.critic = critic

    def create_model(self, input_layer_size, action_space_size):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_layer_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_space_size, activation = 'linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        # model.load_weights(checkpoint_dir_sp) if self.is_speaker else model.load_weights(checkpoint_dir_ls)
        return model

    def train(self, terminal_state):
        # Start training only if enough transition samples has been collected in the memory
        if self.replay_memory.get_size() < AGENT_MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch from memory replay table
        minibatch = self.replay_memory.sample_minibatch(minibatch_size = MINIBATCH_SIZE)

        # Get the current states and their corresponding q values for each sample in the minibatch
        current_states = np.array([transition[0] for transition in minibatch]) if self.is_speaker else np.array([transition[1] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)

        X = []
        y = []

        # for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
        for index, (S_sp, S_ls, A_sp, A_ls, R_sp, R_ls, S_next_sp, S_next_ls, done_sp, done_ls) in enumerate(minibatch):
            done = done_sp if self.is_speaker else done_ls
            reward = R_sp if self.is_speaker else R_ls
            action = A_sp if self.is_speaker else A_ls
            current_state = S_sp if self.is_speaker else S_ls

            if not done:
                # Calculate Nash Q using the centralized network
                joint_observation = np.concatenate((S_next_sp, S_next_ls), axis=None)
                joint_q_vals = self.critic.get_qs(joint_observation)
                nash_q = np.max(joint_q_vals)
                # Nash update
                new_q = reward + DISCOUNT * nash_q 
            else:
                new_q = reward
            
            # Update Q value for the given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Prepare training data
            X.append(current_state)
            y.append(current_qs)

        if self.is_speaker:
            self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, shuffle=False, verbose=0, callbacks=[cp_callback_sp])
        else:
            self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, shuffle=False, verbose=0, callbacks=[cp_callback_ls])


        if terminal_state:
            self.target_update_counter += 1
        
        # update target network with weights of main network if condition satisfied
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]

class JointCritic:
    def __init__(self, input_layer_size, action_space_size, ReplayMemoryObject):
        # Main model which we use to train
        self.model = self.create_model(input_layer_size, action_space_size)

        # Target network to make sure the updating is stable
        self.target_model = self.create_model(input_layer_size, action_space_size)
        self.target_model.set_weights(self.model.get_weights())

        # The array to keep the memory for the last n steps for training
        self.replay_memory = ReplayMemoryObject

        # Count when to update target network with main network's weights
        self.target_update_counter = 0


    def create_model(self, input_layer_size, action_space_size):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_layer_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_space_size, activation = 'linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        # model.load_weights(checkpoint_path_critic)
        return model
    
    def train(self, terminal_state):
        # Start training only if enough transition samples has been collected in the memory
        if self.replay_memory.get_size() < CRITIC_MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch from memory replay table
        minibatch = self.replay_memory.sample_minibatch(minibatch_size = MINIBATCH_SIZE)

        # Get the current states and their corresponding q values for each sample in the minibatch
        current_states = np.array([np.concatenate((transition[0], transition[1]), axis=None) for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)

        # Get the next states their corresponding q values for each sample in the minibatch
        new_current_states = np.array([np.concatenate((transition[6], transition[7]), axis=None) for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        # for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
        for index, (S_sp, S_ls, A_sp, A_ls, R_sp, R_ls, S_next_sp, S_next_ls, done_sp, done_ls) in enumerate(minibatch):
            done = done_sp or done_ls
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = R_sp + R_ls + DISCOUNT * max_future_q
            else:
                new_q = R_sp + R_ls
            
            # Update Q value for the given state
            current_qs = current_qs_list[index]
            action_idx = np.ravel_multi_index((A_sp, A_ls), dims=(3, 5))
            current_qs[action_idx] = new_q

            # Prepare training data
            X.append(current_states[index])
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, shuffle=False, verbose=0, callbacks=[cp_callback_critic])

        if terminal_state:
            self.target_update_counter += 1
        
        # update target network with weights of main network if condition satisfied
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]

def eps_greedy_act_selection(epsilon, action_space_size, q_values):
    if np.random.random() < epsilon:
        # randomly choose one action
        return np.random.randint(0, action_space_size)
    else:
        # all q values
        return np.argmax(q_values)

AGENT_NAMES = ['speaker_0', 'listener_0']
AGENT_INFOS = {name: {"agent_idx": 0 if name == 'speaker_0' else 1,
                        "action_space_size": 3 if name == 'speaker_0' else 5,
                        "input_layer_size": 3 if name == 'speaker_0' else 11,
                        "is_speaker": True if name == 'speaker_0' else False
                        } for name in AGENT_NAMES}

UPDATE_COUNTER = 0

ALL_REWARDS = {agent_name:[] for agent_name in AGENT_NAMES}
# Save the current reward dictionary
with open('./models/reward_dict/my_dict.json', 'w') as f:
    # use json.dump to serialize the dictionary object to JSON and write it to the file
    json.dump(ALL_REWARDS, f)


epsilon = EPSILON

# Create a replay buffer
replay_buff = ReplayMemory(max_len = REPLAY_MEMORY_SIZE)
with open('./models/replay_buffer/my_deque.pickle', 'wb') as f:
    pickle.dump(replay_buff.replay_memory, f)

# Create the critic DQN Agent
critic_dqn = JointCritic(input_layer_size=14, action_space_size=15, ReplayMemoryObject=replay_buff)


# Create the Nash DQN Agents
NASH_DQN_AGENTS = {name: NashDQNAgent(input_layer_size=AGENT_INFOS[name]["input_layer_size"],
                                      action_space_size=AGENT_INFOS[name]["action_space_size"],
                                      ReplayMemoryObject=replay_buff,
                                      is_speaker=AGENT_INFOS[name]["is_speaker"],
                                      critic=critic_dqn) for name in AGENT_NAMES}

##### Save the weights
critic_dqn.model.save_weights('./models/weights/critic_weights.h5')
NASH_DQN_AGENTS['speaker_0'].model.save_weights('./models/weights/speaker_weights.h5')
NASH_DQN_AGENTS['listener_0'].model.save_weights('./models/weights/listener_weights.h5')



os.system("tmux wait-for -S script_finished")



