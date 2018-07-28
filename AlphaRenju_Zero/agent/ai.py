from .agent import Agent
from ..network.network import *
import random
from .mcts import *
import time


def matrix2vector(mat):
    vec = [0 for i in range(225)]
    for i in range(15):
        for j in range(15):
            vec[15 * i + j] = mat[i][j]
    return vec

def vector2matrix(vec):
    mat = [[0 for i in range(15)] for i in range(15)]
    for i in range(15):
        for j in range(15):
            mat[i][j] = vec[15 * i + j]
    return mat

def vecsub2matsub(k):
    j = k % 15
    i = (k-j)/15
    return i, j

def matsub2vecsub(i, j):
    return 15*i + j

def sample(p):
    n = len(p)
    threshold = random.uniform(0,1)
    s = 0
    for i in range(n):
        s += p[i]
        if s >= threshold:
            return i


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


class NNAgent(AI):
    def __init__(self, network, color):
        AI.__init__(self, color)
        self._network = network

    def play(self, obs, action, *args):
        color = self.color()

        # first layer
        m_1 = [[0 for i in range(15)] for i in range(15)]
        for i in range(15):
            for j in range(15):
                if obs[i][j] == color:
                    m_1[i][j] = 1

        # second layer
        m_2 = [[0 for i in range(15)] for i in range(15)]
        for i in range(15):
            for j in range(15):
                if obs[i][j] == -color:
                    m_2[i][j] = 1

        # third(color) layer
        m_3 = [[color for i in range(15)] for i in range(15)]

        m = [m_1, m_2, m_3]
        p, v = self._network.calc(m)
        print(p)
        return vecsub2matsub(sample(p))



if __name__ == '__main__':
    p = [(i+1)/55 for i in range(10)]
    print(sum(p))
    print(p)
    print(sample(p))






