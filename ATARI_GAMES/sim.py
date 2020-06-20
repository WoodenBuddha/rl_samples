import os
import sys

from core.Simulation import *
from agents.DQNAgent import *
from environments.PacmanEnv import PacmanEnv
from environments.PongEnv import PongEnv
from utils import Serializer


def main(*args):
    try:
        import google.colab
        import google.colab.files as files
        IN_COLAB = True
        print('Running in Google Colab')
    except:
        IN_COLAB = False

    episodes = 1000
    optimization_freq = 4
    update_freq = 10
    seed = None
    path = os.getcwd()
    path_to_model = None
    fname = None
    presentation = False

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
        elif k == '-pres':
            v = str(v).lower()
            if v == 'true' or v == 'yes' or v == 'y':
                presentation = True
        else:
            raise Exception(f'Unknown argument [{k}]')

    def func(simulation):
        assert isinstance(simulation, Simulation)
        if simulation.counter % 1000:
            print(f'Saving model on {simulation.counter} episode')
            filename = 'model_' + str(simulation.counter) + '.pt'
            simulation.agent.dump_policy(PATH=path, FILENAME=filename)
            if IN_COLAB:
                file = join(path, filename)
                files.download(file)

    agent = DQNAgent()
    env = PacmanEnv()
    simulation = Simulation(num_episodes=episodes, optim_freq=optimization_freq, target_update_freq=update_freq,
                            seed=seed, presentation=presentation, func=None)

    # Set objects
    simulation.set_agent(agent)
    simulation.set_env(env)

    simulation.build()
    if path_to_model is not None:
        agent.load_state_dict(path_to_model)

    serializer = Serializer(path, fname)

    simulation.run()


    sim_info = simulation.info()
    sim_info.update(agent.info())
    sim_info.update(env.info())

    serializer.save(sim_info, 'info')
    serializer.save(simulation.tracker)
    agent.dump_policy(serializer.get_savedir())
    serializer.zip_n_rm()


if __name__ == '__main__':
    main(*sys.argv[1:])
