import gym
import gym_anytrading
import os
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.activations import relu, sigmoid, linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from tqdm import tqdm

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9975
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

env = gym.make('forex-v0',
               df = FOREX_EURUSD_1H_ASK,
               window_size = 10,
               frame_bound = (10, 300),
               unit_side = 'right')

class DQN():
    def __init__(self, action_size, state_size, lr, epsilon):
        self.action_size = action_size
        self.state_size = state_size
        self.epsilon = epsilon
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.1
        self.discount = 0.99
        self.lr = lr
        self.adam_optimizer = Adam(learning_rate = self.lr)
        self.train_start = 1000
        self.batch_size = 128
        self.mini_batch_size = 64
        self.memory_size = 1000
        self.replay_memory = deque(maxlen=self.memory_size)
        self.target_update_counter = 0
        self.update_target_every = 5
        # self.model = self.build_model()
        # self.target_model = self.build_model()
        self.model = tf.keras.models.load_model('saved_model/model.h5')
        self.target_model = tf.keras.models.load_model('saved_model/model.h5')

        
    def build_model(self):
        # Neural networks
        model = Sequential([
            InputLayer(input_shape= self.state_size, batch_size=self.mini_batch_size),
            Conv1D(filters=64,kernel_size=6, padding='same', activation='tanh'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32,kernel_size=3, padding='same', activation='tanh'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(self.action_size, activation='sigmoid'),
            Dense(self.action_size, activation='softmax'),
        ])
        # model.add(Dense(24, input_shape=self.state_size, activation='sigmoid'))
        # model.add(Dense(48, activation='relu',))
        # model.add(Dense(96, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))

        model.compile(optimizer=self.adam_optimizer, 
                      loss='mse')
        
        return model
    
    def add_to_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
                  
    def get_action(self, discrete_state):
        action = self.model.predict(np.expand_dims(discrete_state, 0))
        return action[0]
                  
    def train_model(self, terminal_state, step):
        if len(self.replay_memory ) < self.train_start:
            return
        # batch_size = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, self.mini_batch_size)
        
        # current_state = np.zeros((batch_size, self.state_size))
        # next_state = np.zeros((batch_size, self.state_size))
        

        current_state = [j[0] for j in mini_batch]
        next_state = [k[3] for k in mini_batch]
        current_qs_list = []
        future_qs_list = []


        # for current_state_single in current_state:
        #     current_qs = self.model.predict(np.expand_dims(np.array(current_state_single), 0))[0]
        #     current_qs_list.append(current_qs)
        # for next_state_singe in next_state:
        #     future_qs = self.target_model.predict(np.expand_dims(np.array(next_state_singe), 0))[0]
        #     future_qs_list.append(future_qs)
        current_qs_list = self.model.predict(np.array(current_state), batch_size=self.mini_batch_size)
        future_qs_list = self.target_model.predict(np.array(next_state), batch_size=self.mini_batch_size)
        
        X = []
        y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):
            if done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
                
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X), np.array(y), batch_size= self.mini_batch_size,
                      verbose=1, shuffle=False if terminal_state else None)
        if terminal_state:
            print('Saving model ...')
            self.model.save('saved_model/model.h5', overwrite=True)

        if terminal_state:
            self.target_update_counter +=1 
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


action_space = env.action_space.n
state_space = env.shape
learning_rate = 0.001
epsilon = 1
episodes = 20
agent = DQN(action_space, state_space, learning_rate, epsilon)
action_type = ['short', 'long']
# agent.model.summary()
agent.get_action(env.reset())
# print(np.array(env.reset()))


for episode in tqdm(range(episodes), ascii=True, unit='episode'):
# for episode in range(episodes):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    
    while not done:
        if np.random.rand() <= epsilon:
            action = np.random.randint(2)
            # print(f'Random action: {action_type[action]}')
        else:
            action = np.argmax(agent.get_action(current_state))
            # print(f'Network action: {action_type[action]}')
        new_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        
        agent.add_to_memory(current_state, action, reward, new_state, done)
        agent.train_model(done, step) 

        current_state = new_state
        step += 1

    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY
        epsilon = max(EPSILON_MIN, epsilon)

    print(f'Episode reward: {episode_reward}')
plt.cla()
env.render_all()
plt.show()