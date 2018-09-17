from .agent import Agent
from .mcts import coordinate2index
import numpy as np


class HumanAgent(Agent):
    def __init__(self, renderer, color, board_size):
        self._renderer = renderer
        self._color = color
        self._board_size = board_size

    def play(self, obs, action, stone_num, *args):
        x, y = self._renderer.ask_for_click()
        ind = coordinate2index((x, y), self._board_size)
        pi = np.zeros(self._board_size * self._board_size)
        pi[ind] = 1
        return (x, y), pi, None, None
