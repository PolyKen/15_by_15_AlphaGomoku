from .agent import Agent
from ..network.network import *
import random
from .mcts import *
import time


class AI(Agent):
    def __init__(self, color):
        Agent.__init__(self, color)


class MCTSAgent(AI):
    def __init__(self, conf, color):
        AI.__init__(self, color)
        network = Network(conf)
        self._mcts = MCTS(conf, network, color)
        self._network = network
        self._board_size = conf['board_size']

    def play(self, obs, action, stone_num):
        act_ind, pi = self._mcts.action(obs, action, stone_num)
        act_cor = index2coordinate(act_ind, self._board_size)
        return act_cor, pi

    def set_self_play(self, is_self_play):
        self._mcts.set_self_play(is_self_play)

    def reset_mcts(self):
        self._mcts.reset()
    
    def train(self, obs, color, pi, z):
        print('training begins:')
        start = time.clock()
        loss = self._network.train(obs, color, pi, z)
        end = time.clock()
        print('training time = ' + str(end - start))
        print('*********************************************')
        return loss
        
    def save_model(self):
        self._network.save_model()

    def load_model(self):
        self._network.load_model()


