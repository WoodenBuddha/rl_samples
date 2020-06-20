from itertools import count

import numpy as np
import torch

from core.Agent import Agent
from core.Environment import Environment
from utils import Tracker


class Simulation:
    def __init__(self, num_episodes=1000, render=False, presentation=False, optim_freq=128, target_update_freq=10,
                 track=True, seed=None, func=None):
        assert optim_freq > 0, 'Policy optimization frequency must be positive numeric'
        assert target_update_freq > 0, 'Target policy update frequency must be positive numeric'
        self.env = None
        self.agent = None
        self.ready = False

        self.__episodes_num = num_episodes
        self.__optim_freq = optim_freq
        self.__update_freq = target_update_freq
        self.__render = render

        self.__presentation = presentation
        if self.__presentation:
            self.__render = self.__presentation
            track = False

        self.__track = track
        if self.__track:
            self.tracker = Tracker()

        self.__seed = None
        if seed is not None:
            assert isinstance(seed, int) and not isinstance(seed, bool)
            self.__seed = seed
        self.func = func
        print(
            f'Simulation initialized: episodes=[{self.__episodes_num}], optim_freq=[{self.__optim_freq}], update_freq=[{self.__update_freq}]')

    def set_agent(self, agent):
        assert isinstance(agent, Agent), 'Agent must extend [{}]'.format(Agent.__class__)
        self.agent = agent

    def set_env(self, env):
        assert isinstance(env, Environment), 'Environment must extend [{}]'.format(Environment.__class__)
        self.env = env

    def build(self):
        assert self.agent is not None, 'Agent is not set'
        assert self.env is not None, 'Environment is not defined'

        # TODO: refactor model defining
        self.__build_convnet()

        self.ready = True

    def __build_convnet(self):
        # initiation block
        # TODO: replace with generic approach
        _, screen_channels, screen_height, screen_width = self.env.get_screen_shape()

        self.agent.set_actions(self.env.get_actions().n)
        self.agent.set_width(screen_width)
        self.agent.set_height(screen_height)
        self.agent.set_in_channels(screen_channels)
        self.agent.build()

    def __build_mlp(self):
        raise Exception('Not implemented: build agent of [MLP] type..')

    def run(self):
        assert self.ready is True

        counter = 1
        while counter <= self.__episodes_num:
            print('Episode {0}..'.format(counter))
            self.__run_episode()
            if counter % self.__update_freq == 0: self.agent.sync_policies()

            self.do_after_episode()
            counter += 1

    def __run_episode(self):
        if self.__seed is not None:
            self.env.seed(self.__seed)

        self.env.reset()

        last_screen = self.env.get_screen()
        current_screen = self.env.get_screen()
        state = current_screen - last_screen

        rewards, losses = [], []

        for t in count():
            if self.__render: self.env.get_screen(mode='human')

            action = self.agent.select_action(state, explore=not self.__presentation)

            # Simulate iteration
            next_obs, reward, done, _ = self.env.make_step(action.item())

            rewards.append(reward)

            reward = torch.tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = self.env.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            self.agent.memory_push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if not self.__presentation and t % self.__optim_freq == 0:
                loss = self.agent.optimize_policy()

                if loss is not None:
                    losses.append(loss)

            if done:
                if self.__track:
                    self.tracker.put(rewards=rewards, losses=losses)
                    print('Mean reward: {0}; Mean loss: {1}'.format(np.mean(rewards), np.mean(losses)))
                break

    def info(self):
        inf = {'episodes_number': self.__episodes_num, 'freq': self.__optim_freq, 'update_freq': self.__update_freq}
        return inf

    def do_after_episode(self, param=None):
        if self.func is not None:
            return self.func(param)
