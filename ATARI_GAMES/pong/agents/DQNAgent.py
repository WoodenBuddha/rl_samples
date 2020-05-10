import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from pong.agents.backbones.conv_nets import ConvNet
from utils import ReplayMemory
from utils import Transition


class DQNAgent():
    def __init__(self, in_channels, width, height, actions, device=None):
        self.actions = actions
        self.policy_net = ConvNet(in_channels, width, height, self.actions)
        self.target_net = ConvNet(in_channels, width, height, self.actions)

        self.sync_policies()

        self.memory = ReplayMemory(10000)  # Default capacity

        # Default loss and optim
        self.loss_func_ = F.smooth_l1_loss
        self.optimizer_ = optim.RMSprop(self.policy_net.parameters())

        self.steps_done = 0
        self.device = device

        self.__init_hyperparams__()

    def __init_hyperparams__(self, BATCH_SIZE=128, GAMMA=0.999, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200,
                             TARGET_UPDATE=10):
        assert EPS_DECAY != 0
        self.batch_size = BATCH_SIZE
        self.g = GAMMA
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.target_update = TARGET_UPDATE

    def load_state_dict(self, state_dict):
        self.policy_net.load_state_dict(torch.load(state_dict))
        self.sync_policies()

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, optim.Optimizer)
        self.optimizer_ = optimizer

    def set_loss(self, loss_func):
        self.loss_func_ = loss_func

    def memory_push(self, *args):
        self.memory.push(*args)

    def select_action(self, state):
        """ Select action with epsilon-greedy strategy """
        ep_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > ep_threshold:
            with torch.no_grad():
                # Exploitation action -> get action index with highest expected reward
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Exploration action -> get random action
            exploration_action = [[random.randrange(self.actions)]]
            return torch.tensor(exploration_action, device=self.device, dtype=torch.long)

    def optimize_policy(self, loss_gatherer=None):
        """ Optimize policy """
        if len(self.memory) < self.batch_size:
            return

        # Sample random transitions of batch size from memory.
        # Converts batch - array of Transitions to Transition of batch-arrays.
        # (see https://stackoverflow.com/a/19343/3343043 for detailed explanation)

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # Tuple (mask) of Null/NotNull states.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
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
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        #
        # According to V(s) == max(Q(s)) => next_state_value.max() == next_q_value
        # Therefore here: next_state_value == next_Q_value
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        # Make action and transit to next state, get REWARD for transition.
        # Exp_Q(s_{t+1}) = Q(s_{t+1} * GAMMA) + REWARD
        expected_state_action_values = (next_state_values * self.g) + reward_batch

        # Compute loss
        # L1_smooth = {|x|1|α|x2if |x|>α;if |x|≤α}
        # L1_smooth(Q_value, Exp_Q_value)
        loss = self.loss_func_(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize policy
        self.optimizer_.zero_grad()

        # Compute derivatives
        loss.backward()

        # Clip the final gradient from -1 to 1 for each wight
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # Update weights
        self.optimizer_.step()

        if loss_gatherer is not None:
            if isinstance(loss_gatherer, list):
                loss_gatherer.append(loss.item())

    def sync_policies(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def dump_policy(self, PATH=None):
        if PATH is None:
            PATH = ''
        elif PATH == 'current_dir':
            pass
        elif PATH == 'storage':
            pass

        PATH += type(self.policy_net).__name__
        PATH += '.pt'

        torch.save(self.target_net.state_dict(), PATH)
