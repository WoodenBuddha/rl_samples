import math
import random
from os.path import join

import torch
import torch.nn.functional as F
import torch.optim as optim

from core.Agent import Agent
from agents.backbones.conv_nets import ConvNet
from utils import ReplayMemory
from utils import Transition

class DQNAgent(Agent):
    def __init__(self, in_channels=None, width=None, height=None, actions=None, device=None,
                 batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200):
        super(DQNAgent, self).__init__()
        self._actions = actions
        self._in_channels = in_channels
        self._width = width
        self._height = height
        self._device = device

        self._memory = ReplayMemory(10000)  # Default capacity
        self._steps_done = 0

        self.__init_hyperparams(batch_size=batch_size, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
        print(
            f'Agent initialized: agent=[{self.agent_type}], batch_size=[{batch_size}], gamma=[{gamma}], '
                f'eps_start=[{eps_start}], eps_end=[{eps_end}], eps_decay=[{eps_decay}]')

    def build(self):
        assert self._actions is not None
        assert self._width is not None
        assert self._height is not None
        assert self._in_channels is not None

        self.__policy_net = ConvNet(self._in_channels, self._width, self._height, self._actions)
        self.__target_net = ConvNet(self._in_channels, self._width, self._height, self._actions)
        self.sync_policies()

        # Set info about NN model
        # TODO: pass class
        self.model_type = self.__policy_net.__class__.__name__

        # Default loss and optim
        self._loss_func = F.smooth_l1_loss
        self._optimizer = optim.RMSprop(self.__policy_net.parameters())

    def __init_hyperparams(self, batch_size, gamma, eps_start, eps_end, eps_decay):
        assert eps_decay != 0
        self.__batch_size = batch_size
        self.__g = gamma
        self.__eps_start = eps_start
        self.__eps_end = eps_end
        self.__eps_decay = eps_decay

    def load_state_dict(self, state_dict):
        self.__policy_net.load_state_dict(torch.load(state_dict))
        self.sync_policies()

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, optim.Optimizer)
        self._optimizer = optimizer

    def set_loss(self, loss_func):
        self._loss_func = loss_func

    def memory_push(self, *args):
        self._memory.push(*args)

    def select_action(self, state, explore=True):
        """ Select action with epsilon-greedy strategy """
        ep_threshold = self.__eps_end + (self.__eps_start - self.__eps_end) * math.exp(-1. * self._steps_done / self.__eps_decay)
        self._steps_done += 1
        if explore and random.random() <= ep_threshold:
            # Exploration action -> get random action
            exploration_action = [[random.randrange(self._actions)]]
            return torch.tensor(exploration_action, device=self._device, dtype=torch.long)
        else:
            with torch.no_grad():
                # Exploitation action -> get action index with highest expected reward
                return self.__policy_net(state).max(1)[1].view(1, 1)

    def optimize_policy(self):
        """ Optimize policy """
        if len(self._memory) < self.__batch_size:
            return

        # Sample random transitions of batch size from memory.
        # Converts batch - array of Transitions to Transition of batch-arrays.
        # (see https://stackoverflow.com/a/19343/3343043 for detailed explanation)

        transitions = self._memory.sample(self.__batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # Tuple (mask) of Null/NotNull states.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self._device,
                                      dtype=torch.bool)

        # Concatenate tensors in 0-dim (stack non null state-tensors from batch)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Similar to np.vstack in 0-dim
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(state, action)
        # 'gather' will index the rows of the q-values (i.e.the per-sample q-values in a batch of q-values)
        # by the batch - list of actions. The result will be the same as if you had done the following:
        # q_vals = []
        # for qv, ac in zip(Q(obs_batch), act_batch):
        #     q_vals.append(qv[ac])
        #
        # q_vals = torch.cat(q_vals, dim=0)
        # state_action_value == Q_value
        state_action_values = self.__policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        #
        # According to V(s) == max(Q(s)) => next_state_value.max() == next_q_value
        # Therefore here: next_state_value == next_Q_value
        next_state_action_values = torch.zeros(self.__batch_size, device=self._device)
        next_state_action_values[non_final_mask] = self.__target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        # Make action and transit to next state, get REWARD for transition.
        # Exp_Q(s_{t+1}) = Q(s_{t+1} * GAMMA) + REWARD
        expected_next_state_action_values = (next_state_action_values * self.__g) + reward_batch

        # Compute loss
        # L1_smooth = {|x|1|α|x2if |x|>α;if |x|≤α}
        # L1_smooth(Q_value, Exp_Q_value)
        loss = self._loss_func(state_action_values, expected_next_state_action_values.unsqueeze(1))

        # Optimize policy
        self._optimizer.zero_grad()

        # Compute derivatives
        loss.backward()

        # Clip the final gradient from -1 to 1 for each wight
        for param in self.__policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # Update weights
        self._optimizer.step()

        return loss.item()

    def sync_policies(self):
        self.__target_net.load_state_dict(self.__policy_net.state_dict())

    def dump_policy(self, PATH=None, FILENAME=None):
        if PATH is None:
            PATH = ''
        elif PATH == 'current_dir':
            pass
        elif PATH == 'storage':
            pass

        if FILENAME is None:
            filename = 'model.pt'
        else:
            filename = FILENAME
        PATH = join(PATH, filename)

        torch.save(self.__target_net.state_dict(), PATH)

    def set_in_channels(self, in_channels):
        self._in_channels = in_channels

    def set_width(self, width):
        self._width = width

    def set_height(self, height):
        self._height = height

    def set_actions(self, actions):
        self._actions = actions

    def set_global_steps(self, steps):
        self._steps_done = steps