import gym


class Environment:
    def __init__(self, env_name, render=False, partially_observed=None):
        assert env_name is not None, 'Environment must be specified'
        self.env_name = env_name
        self.pomdp = partially_observed
        self.env = gym.make(env_name)
        # if not render: self.env = wrappers.Monitor(self.env, '', video_callable=False, force=True)
        print(f'Environment initialized: env=[{self.env_name}]')

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

    def seed(self, seed):
        self.env.seed(seed)

    def info(self):
        inf = {'env_type':self.env_name, 'action_space':self.env.action_space.n, 'partially_observed':self.pomdp}
        return inf
