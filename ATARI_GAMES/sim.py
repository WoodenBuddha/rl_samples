import os
import sys

from core.Simulation import *
from agents.DQNAgent import *
from environments.PacmanEnv import PacmanEnv
from utils import Serializer


def main(*args):
    episodes = 1000
    optimization_freq = 4
    update_freq = 10
    seed = None
    path = os.getcwd()
    path_to_model = 'C:\\Users\\Aidar\\Desktop\\RL\\rl_samples\\results\\2020_06_08_18_13_04_06s\\model.pt'
    fname = None

    for arg in args:
        k = arg.split('=')[0]
        v = arg.split('=')[1]

        if k == '-e':
            episodes = int(v)
        elif k == '-of':
            optimization_freq = int(v)
        elif k == '-uf':
            update_freq = int(v)
        elif k == '-s':
            seed = int(v)
        elif k == '-p':
            path = v
        elif k == '-f':
            fname = v
        else:
            raise Exception(f'Unknown argument [{k}]')

    agent = DQNAgent()
    env = PacmanEnv()
    simulation = Simulation(num_episodes=episodes, optim_freq=optimization_freq, target_update_freq=update_freq,
                            seed=seed, presentation=True)

    # Set objects
    simulation.set_agent(agent)
    simulation.set_env(env)

    simulation.build()
    agent.load_state_dict(path_to_model)

    simulation.run()

    serializer = Serializer(path, fname)

    sim_info = simulation.info()
    sim_info.update(agent.info())
    sim_info.update(env.info())

    serializer.save(sim_info, 'info')
    serializer.save(simulation.tracker)
    agent.dump_policy(serializer.get_savedir())
    serializer.zip_n_rm()


if __name__ == '__main__':
    main(*sys.argv[1:])
