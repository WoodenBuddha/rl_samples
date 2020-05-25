import csv
import json
import os
import random
import pickle
import shutil

from collections import namedtuple
from datetime import datetime
from os.path import join
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class RLCollectables(object):
    def __init__(self):
        self.__rewards = []
        self.__losses = []

    def collect_reward(self, episode, step, reward):
        t = (episode, (step, reward))
        self.__rewards.append(t)

    def collect_loss(self, episode, step, loss):
        t = (episode, (step, loss))
        self.__losses.append(t)

    def get(self):
        return self.__rewards, self.__losses

    def save(self, folder=None):
        self.__save_collected(self.__losses, folder, 'loss')
        self.__save_collected(self.__rewards, folder, 'reward')

    def __save_collected(self, list, filepath, filename):
        dump_to_csv(list, filepath, filename)


def create_folder(path, name):
    if path is None: path = Path(__file__).parent.absolute()
    if name is None:
        name = 'None'
    elif name == 'timestamp':
        name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%ms")
    path = join(path, name)
    print('Creating {}'.format(path))
    Path(path).mkdir(parents=True, exist_ok=True)
    return path, name


def check_file(path):
    if path is None: return None
    my_file = Path(path)
    if my_file.is_file():
        return path
    else:
        return None


def rm_folder(path, folder):
    dirpath = Path(join(path, folder))
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)


def zipdir(path, name):
    handler = ZipFile(name + '.zip', 'w', ZIP_DEFLATED)

    for root, dirs, files in os.walk(path):
        for file in files:
            handler.write(join(root, file))

    handler.close()


def dump_to_json(dict, path, fname):
    data = json.dumps(dict)
    f = open(join(path, fname), "w")
    f.write(data)
    f.close()


def dump_to_pickle(list, path, fname):
    with open(join(path, fname), 'wb') as fp:
        pickle.dump(list, fp)

def dump_to_csv(list, path, fname):
    with open(join(path, fname), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list)
