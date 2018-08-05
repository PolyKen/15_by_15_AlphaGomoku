from .agent import Agent
from ..network.network import *
from .mcts import *
from ..decorator import *


class AI(Agent):
    def __init__(self, color):
        Agent.__init__(self, color)

    def play(self, *args, **kwargs):
        pass


class MCTSAgent(AI):
    def __init__(self, conf, color, use_stochastic_policy):
        AI.__init__(self, color)
        network = Network(conf)
        self._mcts = MCTS(conf, network, color, use_stochastic_policy)
        self._network = network
        self._board_size = conf['board_size']

    def play(self, obs, action, stone_num):
        act_ind, pi, prior_prob, value = self._mcts.action(obs, action, stone_num)
        act_cor = index2coordinate(act_ind, self._board_size)
        return act_cor, pi, prior_prob, value

    def set_self_play(self, is_self_play):
        self._mcts.set_self_play(is_self_play)

    def set_stochastic_policy(self, use_stochastic_policy):
        self._mcts.set_stochastic_policy(use_stochastic_policy)

    def reset_mcts(self):
        self._mcts.reset()

    @log
    def train(self, obs, color, last_move, pi, z):
        loss = self._network.train(obs, color, last_move, pi, z)
        return loss
        
    def save_model(self):
        self._network.save_model()
        print('> model saved')

    def load_model(self):
        self._network.load_model()


