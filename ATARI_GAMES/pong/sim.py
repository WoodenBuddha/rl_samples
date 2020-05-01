from itertools import count

import gym
import numpy as np
import torch
import torchvision.transforms as T

from pong.agents.DQNAgent import DQNAgent

resize = T.Compose([T.ToPILImage(),
                    T.ToTensor()])

env = gym.make("Pong-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_screen():
    raw_screen = env.render(mode='rgb_array')

    # Crop screen
    raw_screen = raw_screen[35:195]
    screen = raw_screen.transpose((2, 0, 1))

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return torch.tensor(screen, dtype=torch.float).unsqueeze(0)


env.reset()

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

agent = DQNAgent(screen_width, screen_height, n_actions)

# start simulation
num_episodes = 100
resume = False
render = True
for i_episode in range(num_episodes):
    print('Episode {} started...'.format(i_episode))
    env.reset()
    reward_sum = 0
    running_reward = None

    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action

        if render: env.render()

        action = agent.select_action(state)

        # Simulate iteration
        _, reward, done, _ = env.step(action.item())

        reward_sum += reward

        reward = torch.tensor([reward])

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
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
