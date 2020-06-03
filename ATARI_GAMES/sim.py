from core.Simulation import *
from core.agents.DQNAgent import *
from pacman.PacmanEnv import PacmanEnv
from pong.PongEnv import PongEnv


def main(**kwargs):
    agent = DQNAgent()
    env = PongEnv()
    simulation = Simulation(env, agent, track=True, num_episodes=1000)

    # simulation.load_pretrained_model()
    simulation.run()
    simulation.save_results(path=None)

if __name__ == '__main__':
    main()
