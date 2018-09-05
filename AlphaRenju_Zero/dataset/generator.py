from ..ui import *
from ..config import *
from ..utils import *
from .dataset import *


class Generator:
    def __init__(self, board_size, max_noise_stone_num):
        self._board_size = board_size
        self._max_noise_stone_num = max_noise_stone_num

    @log
    def generate_live_4_attack(self, sample_num=10000):
        color = np.random.random_integers(0, 1) * 2 - 1
        record = GameRecord()
        for i in range(sample_num):
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)

            for x, y in pos_list:
                board[x][y] = color
            self._add_noise(board=board, next_player=color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)
            pi = np.array([0 for _ in range(self._board_size ** 2)])
            if len(fix_pos_list) == 2:
                ind_1 = coordinate2index(fix_pos_list[0], self._board_size)
                ind_2 = coordinate2index(fix_pos_list[1], self._board_size)
                pi[ind_1], pi[ind_2] = 0.5, 0.5
            else:
                ind = coordinate2index(fix_pos_list[0], self._board_size)
                pi[ind] = 1
            record.add(obs=board, color=color, last_move=pos_list[0], pi=pi, z=1)
        return record

    def generate_live_4_defend(self, sample_num=10000):
        pass

    def generate_dead_4_defend(self, sample_num=10000):
        pass

    def generate_live_3_defend(self, sample_num=10000):
        pass

    def _generate_consecutive_line(self, consecutive_num):
        start_pos = np.random.random_integers(0, self._board_size - 1, 2)
        end_pos = [-1, -1]
        while end_pos[0] < 0 or end_pos[0] > 14 or end_pos[1] < 0 or end_pos[1] > 14:
            dx, dy = list(np.random.random_integers(-1, 1, 2))
            if dx == 0 and dy == 0:
                continue
            end_pos[0], end_pos[1] = start_pos[0] + consecutive_num * dx, start_pos[1] + consecutive_num * dy
        fix_pos_list = []
        if dx == 0:
            x_list = [start_pos[0]] * consecutive_num
        else:
            x_list = list(range(start_pos[0], end_pos[0], dx))
        if dy == 0:
            y_list = [start_pos[1]] * consecutive_num
        else:
            y_list = list(range(start_pos[1], end_pos[1], dy))

        fp_1 = [start_pos[0] - dx, start_pos[1] - dy]
        if fp_1[0] in list(range(0, self._board_size)) and fp_1[1] in list(range(0, self._board_size)):
            fix_pos_list.append(fp_1)
        fp_2 = [end_pos[0] + dx, end_pos[1] + dy]
        if fp_2[0] in list(range(0, self._board_size)) and fp_2[1] in list(range(0, self._board_size)):
            fix_pos_list.append(fp_2)

        pos_list = list(zip(x_list, y_list))
        return pos_list, fix_pos_list

    def _empty_board(self):
        empty_board = [[0 for _ in range(self._board_size)] for _ in range(self._board_size)]
        return np.array(empty_board)

    def _add_noise(self, board, next_player, max_stone_num, fix_pos_list):
        stone_num = np.random.random_integers(4, max_stone_num)
        black_stone_ind = np.where(board == BLACK)
        white_stone_ind = np.where(board == WHITE)
        black_stone_num = len(black_stone_ind[0])
        white_stone_num = len(white_stone_ind[0])

        delta = black_stone_num - white_stone_num

        black_stone_num = int((stone_num - delta) / 2)
        if next_player == BLACK:
            white_stone_num = black_stone_num + delta
        else:
            white_stone_num = black_stone_num + delta - 1

        while white_stone_num > 0:
            pos = list(np.random.random_integers(0, self._board_size - 1, 2))
            if board[pos[0]][pos[1]] == 0 and pos not in fix_pos_list:
                white_stone_num -= 1
                board[pos[0]][pos[1]] = WHITE

        while black_stone_num > 0:
            pos = list(np.random.random_integers(0, self._board_size - 1, 2))
            if board[pos[0]][pos[1]] == 0 and pos not in fix_pos_list:
                black_stone_num -= 1
                board[pos[0]][pos[1]] = BLACK
