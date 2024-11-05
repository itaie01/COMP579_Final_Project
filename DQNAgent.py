import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random

#Global Variables
DISCOUNT = 0.99
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# Environments
EPISODES = 1000
MAX_CYCLES = 25

REPLAY_MEMORY_SIZE = int(EPISODES * MAX_CYCLES / 5)  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = int(REPLAY_MEMORY_SIZE / 5)  # Minimum number of steps in a memory to start training

# Exploration settings
EPSILON = 0.1  # decaying epsilon
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001



class DQNAgent:
    def __init__(self, input_layer_size, action_space_size):
        # Main model which we use to train
        self.model = self.create_model(input_layer_size, action_space_size)

        # Target network to make sure the updating is stable
        self.target_model = self.create_model(input_layer_size, action_space_size)
        self.target_model.set_weights(self.model.get_weights())

        # The array to keep the memory for the last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Count when to update target network with main network's weights
        self.target_update_counter = 0


    def create_model(self, input_layer_size, action_space_size):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_layer_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_space_size, activation = 'linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        # transition = (s, a, r, s', done)
        self.replay_memory.append(transition)


    def train(self, terminal_state):
        # Start training only if enough transition samples has been collected in the memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get the current states and their corresponding q values for each sample in the minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        # current_states = StandardScaler().fit_transform(current_states)
        current_qs_list = self.model.predict(current_states, verbose=0)

        # Get the next states their corresponding q values for each sample in the minibatch
        new_current_states = np.array([transition[3] for transition in minibatch])
        # new_current_states = StandardScaler().fit_transform(new_current_states)
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            # Update Q value for the given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Prepare training data
            X.append(current_state)
            y.append(current_qs)

            # Perform normalization on X
            # X = StandardScaler().fit_transform(np.array(X))

        self.model.fit(np.array(X), np.array(y), shuffle=False, verbose=0)

        if terminal_state:
            self.target_update_counter += 1
        
        # update target network with weights of main network if condition satisfied
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]
