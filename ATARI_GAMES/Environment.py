import gym
from gym import wrappers


class Environment():
    def __init__(self, env_name=None, render=False):
        assert env_name is not None
        self.env = gym.make(env_name)
        if not render: self.env = wrappers.Monitor(self.env, '', video_callable=False, force=True)

    def make(self, env_name=None):
        assert env_name is not None
        self.env = gym.make(env_name)

    def get_screen(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def make_step(self, action):
        return self.env.step(action)

    def get_screen_shape(self):
        self.reset()
        return self.get_screen().shape

    def reset(self):
        self.env.reset()

    def get_actions(self):
        return self.env.action_space
