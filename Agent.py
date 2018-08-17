import numpy as np
import random, os, json

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

import skimage as skimage
from skimage import transform, color, exposure, io
from skimage.transform import rotate
from matplotlib import pyplot as plt
from django.urls.conf import path


class DeepQLearningAgent:
    
    def __init__(self, 
                 state_size, 
                 action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1     # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.01
        self.batch_size = 32
        self.filename = './data.h5'
        self.model = self._build_model()
        self.loss = 0


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), 'random'
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]), act_values


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
#         for state, action, reward, next_state, terminal in minibatch:
#             if reward > 0:
#                 self.remember(state, action, reward, next_state, terminal)
        
        state, action, reward, next_state, terminal = zip(*minibatch)
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)
        targets = self.model.predict(state)
        actions = self.model.predict(next_state)
        targets[range(self.batch_size), action] = reward + self.gamma * np.max(actions, axis=1) * np.invert(terminal)
        self.loss = self.model.train_on_batch(state, targets)

#         for state, action, reward, next_state, terminal in minibatch:
#             target = reward
#             if not terminal:
#                 target = (reward + self.gamma *
#                           np.amax(self.model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load_data(self):
        if os.path.exists(self.filename):
            self.model.load_weights(self.filename, True)


    def save_data(self):
        self.model.save_weights(self.filename, True)
