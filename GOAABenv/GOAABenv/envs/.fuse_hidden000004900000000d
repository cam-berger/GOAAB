import numpy as np
import gym
import copy
from gym import spaces
import pandas as pd
import subprocess
import random
import matplotlib.pyplot as plt
import argparse
import shutil
import os
import sys
import _pickle as cPickle
from scapy.all import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import plot_confusion_matrix
from stable_baselines3 import PPO, A2C # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env

# Cleans a processed numerical dataset
def clean(df):
    # Naieve solution: Drop infinite and NaN values from processed data
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    return df.replace([np.inf, -np.inf, np.nan], value=0)

    return df

# Split the data into numerical features for X and Y
def get_xy(df, train):
    data = df.drop(['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], axis=1)
    if not train:
        length = data.shape[0]
        for i in range(0, length):
            data.at[i, 'Label'] = 0
        data = data.astype({'Label': 'int32'})
    if train:
        #make sure there are labels on data
        if data['Label'].dtype == 'object':
            print("Invalid Format: No Labels")
            sys.exit(-1)
    data = clean(data)
    x_data = data.drop('Label', axis=1)
    x = x_data.to_numpy()
    y = data['Label'].to_numpy()
    data = None
    x_data = None

    return x, y

# Split x and y for training and testing
def test_train(x, y):
    return train_test_split(x, y, test_size=.33)

# Module: Gaussian Naive Bayes
def runGNB(x_train, y_train, v):
    v = -1*(v%11 + 1)
    v = 1*(10**v)
    clf = GaussianNB(var_smoothing = v).fit(x_train, y_train)
    return clf

# Module: Standard Scalar
def runSS(x, train):
    dump = 'dumped_ss.pkl'
    trans = ''

    if train:
        trans = StandardScaler()
        return trans.fit_transform(x)
        with open(dump, 'wb') as fid:
            cPickle.dump(trans, fid)
    else:
        with open(dump, 'rb') as fid:
            trans = cPickle.load(fid)

    return trans.fit_transform(x)

# Module: Max Absolute Scalar
def runMaxAbs(x, train):
    dump = 'dumped_mas.pkl'

    if train:
        trans = MaxAbsScaler()
        return trans.fit_transform(x)
        with open(dump, 'wb') as fid:
            cPickle.dump(trans, fid)
    else:
        with open(dump, 'rb') as fid:
            trans = cPickle.load(fid)

    return trans.fit_transform(x)

# Module: Normalization
def runNorm(x, train, n):
    dump = 'dumped_norm.pkl'
    if train:
        n = n%3
        norms = ['l1', 'l2','max']
        trans = Normalizer(norm = norms[n])
        with open(dump, 'wb') as fid:
            cPickle.dump(trans, fid)
    else:
        with open(dump, 'rb') as fid:
            trans = cPickle.load(fid)

    return trans.fit_transform(x)

class GOAABenv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Define constants for clearer code
    PP1 = 0
    PP2 = 1
    PP3_1 = 2
    PP3_2 = 3
    CL1_1 = 4
    CL1_2 = 5
    CL1_3 = 6
    CL1_4 = 7
    CL1_5 = 8
    CL1_6 = 9
    CL1_7 = 10
    CL1_8 = 11
    CL1_9 = 12

    def __init__(self, rows, file_name):
        super(GOAABenv, self).__init__()
        self.rows = rows
        self.title = ''
        self.results = ''
        self.df = pd.read_csv(file_name)
        if self.df.shape[1] != 84:
            print("Invalid Format: Incorrect dimensions")
            sys.exit(-1)

        x, y = get_xy(self.df, True)
        self.num_feat = x.shape[1]
        # Define action and observation space
        # They must be gym.spaces objects (Discrete, Box)
        # Example when using discrete actions, we have two: left and right
        # We may have (for our state) a multitude of actions depending on which module we are utilizizing
        n_actions = 13
        self.action_space = spaces.Discrete(n_actions)

        # The observation will be the coordinate of the agent
        # for us it is simply the np array consisting of the data itself
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-1*np.inf, high=np.inf, shape=(self.rows, self.num_feat))

        '''
        A gym.spaces.Box() object is a cartesian product of n closed intervals:

                      1          2    ...  n_feat-1
                  ___-1_________-1____...____-1____
        1    inf | (-1, inf) (-1, inf) ... (-1, inf)
        2    inf | (-1, inf) (-1, inf) ... (-1, inf)
        :     :  |     :         :     .       :
        :     :  |     :         :       .     :
        rows inf | (-1, inf) (-1, inf) ... (-1, inf)

        '''
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the sample sata from the given dataset
        # Randomly choose self.rows (observations) from the dataset
        self.title = "Sample size = " + str(self.rows) + "x" + str(self.num_feat) + "\n\n"
        self.results = ''
        sample = self.df.sample(n = self.rows)

        # Split the dataset up into testing and training (x_train is considered the agent's state)
        self.agent_state, self.y = get_xy(sample, True)

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return self.agent_state

    def step(self, action):
        done = False
        reward = 0
        if action == self.PP1:
            # run standard scalar pre-processing module
            self.title += "Pre-Processing: Performing Standard Scalar\n"
            self.agent_state = runSS(self.agent_state, True)
            reward = -0.05
        elif action == self.PP2:
            # run Max Absolute Value Scaling
            self.title += "Pre-Processing: Performing MaxAbsScaler\n"
            self.agent_state = runMaxAbs(self.agent_state, True)
        elif action == self.PP3_1 or action == self.PP3_2:
            # run Normalization module
            self.title += "Pre-Processing: Performing Normalization\n"
            if action == self.PP3_1:
                self.agent_state = runNorm(self.agent_state, True, 0)
            if action == self.PP3_2:
                self.agent_state = runNorm(self.agent_state, True, 1)
            reward = -0.05
        elif action >= self.CL1_1:
            # run Support Vector Classifier on the dataset
            x_train, x_test, y_train, y_test = test_train(self.agent_state, self.y)
            v = int(action)-4
            clf = runGNB(x_train, y_train, v)
            self.title += "Classification: Training Gaussian Naive Bayes Classifier\n"
            v = -1*(v%11 + 1)
            v = 1*(10**v)
            self.title += "\tVariable smoothing parameter = " + str("{:1.0e}".format(v)) + "\n"
            y_pred = clf.predict(x_test)
            reward = balanced_accuracy_score(y_test, y_pred)

            #print confusion matrix
            class_names = ['Malicious','Normal']
            titles_options = [('Raw Confusion Matrix', None),
                  ("Normalized Confusion Matrix", 'true')]

            for name, normalize in titles_options:
                disp = plot_confusion_matrix(clf, x_test, y_test, display_labels=class_names,
                                              cmap=plt.cm.Blues,normalize=normalize)
                disp.ax_.set_title(name)

                self.results += name + "\n" + str(disp.confusion_matrix) + "\n"
            plt.close('all')
            self.results +=  "\nBalanced accuracy = " + str(reward) + "\n"
            # plt.show()
            done = True
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Optionally we can pass additional info, we are not using that for now
        info = {"Steps Taken":copy.copy(self.title), "Results":copy.copy(self.results)}

        return self.agent_state, reward, done, info

    def render(self, mode='console'):
        # if mode != 'console':
        #     raise NotImplementedError()
        # show where it is in the decision making process
        print(self.title)
        print(self.results)
        return

    def close(self):
        pass
