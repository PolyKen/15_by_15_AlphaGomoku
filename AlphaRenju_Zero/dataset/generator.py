import numpy as np


class Generator:
    def __init__(self, conf):
        self._board_size = conf['board_size']

    def generate_live_4_attack(self, sample_num=10000):
        pass

    def generate_live_4_defend(self, sample_num=10000):
        pass

    def generate_dead_4_defend(self, sample_num=10000):
        pass

    def generate_live_3_defend(self, sample_num=10000):
        pass

    def _empty_board(self):
        board = [[0 for _ in range(self._board_size)] for _ in range(self._board_size)]
        return np.array(board)

    def _add_noise(self, board, max_stone_num, *fix_pos):
        stone_num = np.random.random_integers(4, max_stone_num)
        board = np.array(board)
        black_stone_ind = np.where(board == 1)
        white_stone_ind = np.where(board == -1)
        black_stone_num = len(black_stone_ind[0])
        white_stone_num = len(white_stone_ind[0])
        delta = black_stone_num - white_stone_num


