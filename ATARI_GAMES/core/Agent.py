# TODO: implement DDQN
# TODO: implement DRQN
# TODO: implement DARQN
class Agent:
    def __init__(self):
        print('{} initialization..'.format(self.__class__.__name__))
        self.agent_type = self.__class__.__name__
        self.model_type = None

    def build(self):
        raise Exception('Not implemented')

    def load_pretrained_model(self):
        raise Exception('Not implemented')

    def set_actions(self, actions):
        raise Exception('Not implemented')

    def set_width(self, width):
        raise Exception('Not implemented')

    def set_height(self, height):
        raise Exception('Not implemented')

    def set_in_channels(self, channels):
        raise Exception('Not implemented')

    def load_state_dict(self, state_dict):
        raise Exception('Not implemented')

    def dump_policy(self, fldr=None):
        raise Exception('Not implemented')

    def select_action(self, state, modifier):
        raise Exception('Not implemented')

    def optimize_policy(self):
        raise Exception('Not implemented')

    def memory_push(self, *args):
        raise Exception('Not implemented')

    def info(self):
        inf = {'agentType': self.agent_type, 'model': self.model_type}
        return inf