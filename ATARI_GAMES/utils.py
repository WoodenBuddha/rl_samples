import random
from collections import namedtuple
from datetime import datetime
from os.path import join
from pathlib import Path

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


# TODO: implement collectables
class RLCollectables(object):
    def __init__(self):
        self.__rewards = []
        self.__losses = []


def create_folder(path, name):
    if path is None: path = Path(__file__).parent.absolute()
    if name is None: name = 'None'
    elif name == 'timestamp': name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%ms")
    path = join(path, name)
    print('Creating {}'.format(path))
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def check_file(path):
    if path is None: return None
    my_file = Path(path)
    if my_file.is_file():
        return path
    else:
        return None