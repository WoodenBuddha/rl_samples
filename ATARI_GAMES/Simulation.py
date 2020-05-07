from itertools import count

import gym
import numpy as np
import torch
import torchvision.transforms as T

from pong.agents.DQNAgent import DQNAgent
from PacmanEnv import PacmanEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = PacmanEnv(render=True)

_, screen_channels, screen_height, screen_width = env.get_screen_shape()
n_actions = env.get_actions().n

agent = DQNAgent(screen_channels, screen_width, screen_height, n_actions)

num_episodes = 100
for i_episode in range(num_episodes):
    print('Episode {} started...'.format(i_episode))
    env.reset()
    reward_sum = 0
    running_reward = None

    last_screen = env.get_screen()
    current_screen = env.get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action

        action = agent.select_action(state)

        # Simulate iteration
        _, reward, done, _ = env.make_step(action.item())

        reward_sum += reward

        reward = torch.tensor([reward])

        # Observe new state
        last_screen = current_screen
        current_screen = env.get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        agent.memory_push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        if t % agent.batch_size == 0:
            agent.optimize_policy()

        if done:
            break

    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

    if i_episode % agent.target_update == 0:
        agent.sync_policies()

agent.dump_policy()
