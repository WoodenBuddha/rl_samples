from core.Simulation import *
from core.agents.DQNAgent import *
from pacman.PacmanEnv import *


def main(**kwargs):
    agent = DQNAgent()
    env = PacmanEnv()
    simulation = Simulation(env, agent, track=True, num_episodes=10000)

    # simulation.load_pretrained_model()
    simulation.run()
    simulation.save_results(path=None)

if __name__ == '__main__':
    main()
