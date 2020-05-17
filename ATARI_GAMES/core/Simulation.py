import torch

from itertools import count

from core.Agent import Agent
from core.Environment import Environment
from utils import check_file, create_folder


class Simulation(object):
    def __init__(self, env, agent, device=None, device_autodefine=True, num_episodes=1000, track=True, render=False, path_to_pretrained=None, collectables_path=None):
        print('Initializing simulation..')
        assert isinstance(env, Environment)
        assert isinstance(agent, Agent)
        self.__env = env
        self.__agent = agent

        _, screen_channels, screen_height, screen_width = self.__env.get_screen_shape()

        self.__agent.set_actions(self.__env.get_actions().n)
        self.__agent.set_width(screen_width)
        self.__agent.set_height(screen_height)
        self.__agent.set_in_channels(screen_channels)
        self.__agent.build()

        self.__set_simulation_params(num_episodes, track, render, device_autodefine, device, collectables_path)

        if path_to_pretrained is not None:
            self.load_pretrained_model(path_to_pretrained)
        print('Ready for simulation.')

    def run(self):
        while self.__ec <= self.__ne:
            if self.__track: print('Episode {}'.format(self.__ec))
            self.__run_episode()

            if self.__track: pass
            self.__ec += 1

        if self.__track:
            self.__agent.dump_policy()
            # TODO: save collectables (locally/to drive)

    def __run_episode(self, presentation_mode=False):
        self.__env.reset()

        # TODO: add information tracking

        last_screen = self.__env.get_screen()
        current_screen = self.__env.get_screen()
        state = current_screen - last_screen

        for t in count():
            if self.__render or presentation_mode: self.__env.get_screen(mode='human')

            action = self.__agent.select_action(state, not presentation_mode)

            # Simulate iteration
            _, reward, done, _ = self.__env.make_step(action.item())

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

            if not presentation_mode and t % self.__agent.batch_size == 0:
                self.__agent.optimize_policy()

            if done:
                break

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

    def __set_simulation_params(self, num_episodes, track, render, device_autodefine, device, collectables_path):
        self.__ec = 1
        self.__ne = num_episodes
        self.__track = track
        self.__render = render

        if device_autodefine:
          self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device is not None:
            self.__device = device

        self.__collectables_path = collectables_path

    # TODO: save tracking result
    # TODO: save in g_drive
    def save_results(self, path=None):
        folder = create_folder(path, 'timestamp')
        self.__agent.dump_policy(folder)
