from core.Simulation import *
from agents.DQNAgent import *
from environments.PacmanEnv import PacmanEnv


def main(**kwargs):
    agent = DQNAgent()
    env = PacmanEnv()
    simulation = Simulation()

    simulation.set_agent(agent)
    simulation.set_env(env)

    simulation.build()

    simulation.run()


if __name__ == '__main__':
    main()
