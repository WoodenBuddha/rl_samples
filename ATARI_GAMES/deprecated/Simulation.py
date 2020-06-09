import os
from itertools import count

import numpy as np
import torch
from deprecated import deprecated

from core.Agent import Agent
from core.Environment import Environment
from utils import check_file, create_folder, RLCollectables, zipdir, dump_to_json, rm_folder

@deprecated
class Simulation(object):
    def __init__(self, env, agent, device=None, device_autodefine=True, num_episodes=1000, track=True, render=False,
                 path_to_pretrained=None, optim_freq=4):
        assert isinstance(env, Environment)
        assert isinstance(agent, Agent)
        assert optim_freq > 0
        self.__info = {}

        self.__env = env
        self.__agent = agent

        self.__add_info(env=env.__class__.__name__, agent=agent.__class__.__name__)

        _, screen_channels, screen_height, screen_width = self.__env.get_screen_shape()

        self.__agent.set_actions(self.__env.get_actions().n)
        self.__agent.set_width(screen_width)
        self.__agent.set_height(screen_height)
        self.__agent.set_in_channels(screen_channels)
        self.__agent.build()

        self.__set_simulation_params(num_episodes, track, render, device_autodefine, device, optim_freq)

        self.__collectables = RLCollectables()

        if path_to_pretrained is not None:
            self.load_pretrained_model(path_to_pretrained)

    def run(self, presentation_mode=False):
        while self.__ec <= self.__ne:
            if self.__track: print('Episode {}'.format(self.__ec))
            mean_reward, mean_loss = self.__run_episode(presentation_mode=presentation_mode)

            if self.__track: print(
                'Episode {0} loss {1}, mean reward {2}'.format(self.__ec, mean_loss, mean_reward))
            self.__ec += 1

        if self.__track:
            self.__collectables.plot_not_realtime('reward')
            self.__collectables.plot_not_realtime('loss')
            self.__collectables.plot_not_realtime('duration')

            # TODO: save collectables (locally/to drive)


    def __run_episode(self, presentation_mode=False):
        self.__env.reset()
        episode_mean_loss = []
        episode_mean_reward = []
        # TODO: add information tracking

        last_screen = self.__env.get_screen()
        current_screen = self.__env.get_screen()
        state = current_screen - last_screen

        for t in count():
            if self.__render or presentation_mode: self.__env.get_screen(mode='human')

            action = self.__agent.select_action(state, not presentation_mode)

            # Simulate iteration
            _, reward, done, _ = self.__env.make_step(action.item())

            episode_mean_reward.append(reward)

            reward = torch.tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = self.__env.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            self.__agent.memory_push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if not presentation_mode and t % self.__freq == 0:
                loss = self.__agent.optimize_policy()
                if loss is not None:
                    self.__collectables.collect_losses(loss)
                    episode_mean_loss.append(loss)

            if done:
                self.__collectables.collect_durations(t+1)
                self.__collectables.collect_rewards(episode_mean_reward)
                break
        return np.mean(episode_mean_reward), np.mean(episode_mean_loss)

    def present(self):
        try:
            while True:
                self.__run_episode(presentation_mode=True)
        except KeyboardInterrupt:
            print('Simulation interrupted..')

    def set_params(self, path):
        pass

    def load_pretrained_model(self, path_to_model):
        path = check_file(path_to_model)
        if path is None:
            raise Exception('Incorrect path to pretrained model!')
        else:
            self.__agent.load_state_dict(path_to_model)

    def _get_screen_size(self):
        assert self.__env is not None
        self._screen_size = self.__env.get_screen_shape()

    def __build_agent(self):
        pass

    def __set_simulation_params(self, num_episodes, track, render, device_autodefine, device, freq):
        self.__ec = 1
        self.__ne = num_episodes
        self.__track = track
        self.__render = render
        self.__freq = freq

        if device_autodefine:
            self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device is not None:
            self.__device = device

        self.__add_info(num_episodes=self.__ne, track=self.__track, render=self.__render)
        print('Episodes: {0}\nOptimization frequency: {1}\nDo render: {2}\nDo track: {3}'
              .format(self.__ne, self.__freq, self.__render, self.__track))

    def __save_info(self, path):
        assert self.__info is not None
        dump_to_json(self.__info, path, 'info')

    # TODO: save tracking result
    # TODO: save in g_drive
    def save_results(self, path=None, zip=True):
        if path is None:
            path = os.path.abspath(os.getcwd())

        folder, fname = create_folder(path, 'timestamp')
        self.__agent.dump_policy(folder)
        self.__collectables.save(folder)
        self.__save_info(folder)

        if zip:
            zipdir(folder, fname)
            rm_folder(folder, fname)

    def __add_info(self, **kwargs):
        for arg in kwargs:
            self.__info[arg] = kwargs.get(arg)
