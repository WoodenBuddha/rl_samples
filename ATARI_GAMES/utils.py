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

import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torch

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
        self.__durations = []
        self.is_ipython = 'inline' in matplotlib.get_backend()

        plt.ion()

    def collect_reward(self, episode, step, reward):
        t = (episode, (step, reward))
        self.__rewards.append(t)

    def collect_rewards(self, reward):
        self.__rewards.append(reward)

    def collect_loss(self, episode, step, loss):
        t = (episode, (step, loss))
        self.__losses.append(t)

    def collect_losses(self, loss):
        self.__losses.append(loss)

    def collect_durations(self, duration):
        self.__durations.append(duration)

    def plot(self, listname, slicing_offset=None, realtime=False):
        if slicing_offset is None:
            slicing_offset = 0

        if listname == 'loss':
            # collectable = self.__losses[-slice:]
            raise Exception('Not implemented yet [{}]'.format(listname))
        elif listname == 'reward':
            collectable = self.__rewards[-slicing_offset:]
        elif listname == 'duration':
            collectable = self.__durations[-slicing_offset:]
        else:
            raise Exception('Unexpected param [{}]'.format(listname))

        plt.figure(2)
        plt.clf()
        collectable = torch.tensor(collectable, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel(listname)
        plt.plot(collectable.numpy())



        # Take 100 episode averages and plot them too
        if len(collectable) >= 100:
            means = collectable.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def plot_not_realtime(self, listname):
        if listname == 'loss':
            collectable = self.__losses
        elif listname == 'reward':
            collectable = self.__rewards
        elif listname == 'duration':
            collectable = self.__durations
        else:
            raise Exception('Unexpected param [{}]'.format(listname))

        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel(listname)
        plt.plot(collectable)

        # means = collectable.unfold(0, 100, 1).mean(1).view(-1)
        # means = torch.cat((torch.zeros(99), means))

        # plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def get(self):
        return self.__rewards, self.__losses

    def save(self, folder=None):
        self.__save_collected(self.__losses, folder, 'loss')
        self.__save_collected(self.__rewards, folder, 'reward')
        self.__save_collected(self.__durations, folder, 'duration')
        # self.__save_all_fig(folder)

    def __save_all_fig(self, path):
        plt.figure(2)

        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(self.__losses)
        plt.savefig(join(path, 'loss.png'))

        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(range(len(self.__rewards)), self.__rewards)
        plt.savefig(join(path, 'reward.png'))

        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(self.__durations)
        plt.savefig(join(path, 'duration.png'))

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
