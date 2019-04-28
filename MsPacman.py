# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:32:08 2019

@author: Phillip Iwanow

program: DQN MsPacman
"""

import tensorflow as tf
import gym
import numpy as np
from collections import deque
import os
import sys

# setting up the enviroment
env = gym.make('MsPacman-v0')
obs = env.reset()

# preprocessing the input data

mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and and downsize
    img = img.mean(axis=2)
    img[img==mspacman_color] = 0 # improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to  1.
    return img.reshape(88, 80, 1)


# building the DQN

input_height = 88
input_width = 80
input_channel = 1
conv_n_maps = [32, 64, 64]
conv_kernel_size =[(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_padding = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(X_state, name):

    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_size, conv_strides, conv_padding,
                conv_activation):
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps,
                                          kernel_size=kernel_size, strides=strides, padding=padding,
                                          activation=activation, kernel_initializer=initializer)
            last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
            hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=hidden_activation,
                                     kernel_initializer=initializer)
            outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
            return outputs, trainable_vars_by_name


 # creating placeholder

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(target_q_values * tf.one_hot(X_action, n_outputs), axis=1, keepdims=True)

y = tf.placeholder(tf.float32, shape=[None, 1])
error = tf.abs(y - q_value)
clipped_error = tf.clip_by_value(error, 0.0, 1.0)
linear_error = 2 * (error - clipped_error)
loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

learning_rate = 0.001
momentum = 0.95

global_step = tf.Variable(0, trainable=False, name='global_step')
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss, global_step=global_step)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# implementing Replay memory

replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

# training

n_steps = 4000000
train_start = 10000
training_interval = 4
save_step = 1000
copy_step = 10000
gamma = 0.99
skip_start = 90
batch_size = 50
iteration = 0
checkpoint_path = "./my_dqn.ckpt"
done = True

with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        if done:
            obs = env.reset()
            for skip in range(skip_start):
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = epsilon_greedy(q_values, step)

            # Online DQN plays
            obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(obs)

            # remember last action
            replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            if iteration < train_start or iteration % training_interval != 0:
                continue

            X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                sample_memories(batch_size))
            next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * gamma * max_next_q_values

            training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

            print("\rIteration{}".format(iteration))

            if step % copy_steps == 0:
                copy_online_to_target.run()

            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)
