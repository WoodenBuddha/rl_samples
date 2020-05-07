import torch
import numpy as np

from itertools import count
from Environment import Environment

ENVIRONMENT_ = "Pong-v0"

class PongEnv(Environment):
    def __init__(self, render=False):
        super(PongEnv, self).__init__(ENVIRONMENT_, render)

    def get_screen(self):
        if self.render: self.env.render()
        return self.preprocess(super().get_screen())

    def preprocess(self, raw):
        # Crop screen
        raw = raw[35:195]
        screen = raw.transpose((2, 0, 1))

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        return torch.tensor(screen, dtype=torch.float).unsqueeze(0)


    def reset(self):
        self.env.reset()


    def get_screen_shape(self):
        self.env.reset()
        return self.get_screen().shape

    def get_actions(self):
        return self.env.action_space

