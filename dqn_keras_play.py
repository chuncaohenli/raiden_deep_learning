#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import random
import numpy as np
from collections import deque

import json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import tensorflow as tf
from raiden_env import *

GAME = 'raiden'  # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 9  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
# OBSERVATION = 3200.  # timesteps to observe before training
# EXPLORE = 3000000.  # frames over which to anneal epsilon
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
SAVE_EVERY = 500
EPISODES = 1000

img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames


def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                            input_shape=(img_rows, img_cols, img_channels)))  # 80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(9))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


def trainNetwork(model):
    # open up a game state to communicate with emulator
    game_state = Raiden_Env()
    game_state.reset()
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros([ACTIONS])
    do_nothing[8] = 0
    x_t, r_0, terminal, hp, live = game_state.step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # print (s_t.shape)

    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

    OBSERVE = 999999999  # We keep observe, never train
    epsilon = FINAL_EPSILON
    print("Now we load weight")
    model.load_weights("model.h5")
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("Weight load successfully")


    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal, hp, live = game_state.step(a_t)
        if terminal:
            break

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()


        s_t = s_t1
        t = t + 1

        print("TIMESTEP", t, \
              "/ STATE", 'Play', \
              "/ EPSILON", epsilon, \
              "/ ACTION", action_index,
              "/ REWARD", r_t, \
              "/ HP", hp, \
              "/ Live", live, \
              "/ Q_MAX ", np.max(Q_sa), \
              "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


def playGame():
    model = buildmodel()
    trainNetwork(model)


def main():
    playGame()


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)
    main()