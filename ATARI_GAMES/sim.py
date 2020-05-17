from pacman.PacmanEnv import PacmanEnv
from core.Simulation import Simulation
from core.agents.DQNAgent import DQNAgent
from utils import create_folder


def main(**kwargs):
    agent = DQNAgent()
    env = PacmanEnv()
    simulation = Simulation(env, agent, track=True)

    # simulation.load_pretrained_model()

    simulation.run()

if __name__ == '__main__':
    main()
