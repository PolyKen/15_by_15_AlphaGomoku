from AlphaGomoku import *
import warnings
import os


if __name__ == '__main__':
    # ignore warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")

    conf = Config()
    conf.set_mode(4)
    env = Env(conf)

    env.set_mcts_agent_version(black_ver=19, white_ver=21)
    env.start_mode()

    env.set_mcts_agent_version(black_ver=19, white_ver=22)
    env.start_mode()

    env.set_mcts_agent_version(black_ver=21, white_ver=22)
    env.start_mode()
