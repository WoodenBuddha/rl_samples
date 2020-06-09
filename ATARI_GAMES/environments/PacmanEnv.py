import torch
import numpy as np

from core.Environment import Environment

color = np.array([210, 164, 74]).mean()
ENVIRONMENT_ = "MsPacman-v0"


class PacmanEnv(Environment):
    def __init__(self, render=False):
        super(PacmanEnv, self).__init__(env_name=ENVIRONMENT_, render=render)

    def get_screen(self, mode='rgb_array'):
        screen = super().get_screen(mode)
        if mode == 'rgb_array':
            screen = self.preprocess(screen)
            return screen

    @staticmethod
    def preprocess(raw_screen):
        # Crop screen
        raw_screen = raw_screen[1:176:2, ::2]

        # Grayscale
        raw_screen = raw_screen.mean(axis=2)

        # Improve contrast/Normalize screen
        raw_screen[raw_screen == color] = 0
        raw_screen = (raw_screen - 128) / 128 - 1

        # Resize and reshape
        screen = raw_screen.reshape(88, 80, 1)
        screen = screen.transpose((2, 0, 1))

        return torch.tensor(screen, dtype=torch.float).unsqueeze(0)
