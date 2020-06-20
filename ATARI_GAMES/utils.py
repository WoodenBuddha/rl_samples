import csv
import json
import os
import pickle
import random
import shutil
from collections import namedtuple
from datetime import datetime
from os.path import join, split
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

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


def create_folder(path, name):
    if path is None: path = Path(__file__).parent.absolute()
    if name is None:
        name = 'None'
    elif name == 'timestamp':
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = join(path, name)
    print('Creating {}'.format(path))
    Path(path).mkdir(parents=True, exist_ok=True)
    return path, name


def remove_folder(path):
    dirpath = Path(path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)


def zip_folder(folder_path, zip_name):
    zname = zip_name + '.zip'
    handler = ZipFile(join(folder_path, zname), 'w', ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file != zname:
                handler.write(join(root, file), file)
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
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerow(list)


def load_csv_to_list(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    return reader


def move_file_to_dir(source_file, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(source_file, dest)

class Tracker:
    def __init__(self, debug=False):
        self.__storage = {}
        self.__file_format = 'csv'
        self.debug = debug

    def put(self, **args):
        for key, item in args.items():
            if key not in self.__storage:
                self.__storage[key] = []
            self.__storage[key].append(item)

    def get(self, key, ep=None):
        if key in self.__storage:
            if ep is None:
                return self.__storage[key]
            else:
                assert isinstance(ep, int) and not isinstance(ep, bool)
                return self.__storage[key][ep]
        else:
            print('No such key: [{}]'.format(key))

    def load(self, key, data):
        if key not in self.__storage:
            self.__storage[key] = []
        self.__storage[key].append(data)

    def get_keys(self):
        return self.__storage.keys()


class Serializer:
    def __init__(self, path, name):
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.__dir_path, self.__dir_name = create_folder(path, name)

    def save(self, obj, fname=None):
        if isinstance(obj, Tracker):
            self.__save_tracker(obj)
        elif isinstance(obj, list):
            assert fname is not None
            dump_to_csv(obj, self.__dir_path, fname)
        elif isinstance(obj, dict):
            assert fname is not None
            print(self.__dir_path)
            dump_to_json(obj, self.__dir_path, fname)
        else:
            assert fname is not None
            dump_to_pickle(obj, self.__dir_path, fname)

    def __save_tracker(self, tracker):
        assert isinstance(tracker, Tracker)
        for key in tracker.get_keys():
            dump_to_csv(tracker.get(key), self.__dir_path, key)

    def zip_n_rm(self):
        root, _ = split(self.__dir_path)
        zip_folder(root, self.__dir_name)
        remove_folder(self.__dir_path)

    def get_savedir(self):
        return self.__dir_path