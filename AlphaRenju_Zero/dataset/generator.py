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
        record = GameRecord()
        i = 0
        while i < sample_num:
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0:
                continue

            for x, y in pos_list:
                board[x][y] = color

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])
            if len(fix_pos_list) == 2:
                ind_1 = coordinate2index(fix_pos_list[0], self._board_size)
                ind_2 = coordinate2index(fix_pos_list[1], self._board_size)
                pi[ind_1], pi[ind_2] = 0.5, 0.5
            if len(fix_pos_list) == 1:
                ind = coordinate2index(fix_pos_list[0], self._board_size)
                pi[ind] = 1

            self._add_noise(board=board, next_player=color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=color, last_move=pos_list[0], pi=pi, z=1)
            i += 1
        return record

    @log
    def generate_live_4_defend(self, sample_num=10000):
        record = GameRecord()
        i = 0
        while i < sample_num:
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0:
                continue

            for x, y in pos_list:
                board[x][y] = color

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])
            if len(fix_pos_list) == 2:
                ind_1 = coordinate2index(fix_pos_list[0], self._board_size)
                ind_2 = coordinate2index(fix_pos_list[1], self._board_size)
                pi[ind_1], pi[ind_2] = 0.5, 0.5
            if len(fix_pos_list) == 1:
                ind = coordinate2index(fix_pos_list[0], self._board_size)
                pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=-color, last_move=pos_list[0], pi=pi, z=-1)
            i += 1
        return record

    @log
    def generate_dead_4_oooo_defend(self, sample_num=10000):
        record = GameRecord()
        i = 0
        while i < sample_num:
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0:
                continue

            for x, y in pos_list:
                board[x][y] = color

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])
            if len(fix_pos_list) == 2:
                ind = coordinate2index(fix_pos_list[0], self._board_size)
                pi[ind] = 1
                fx, fy = fix_pos_list[1][0], fix_pos_list[1][1]
                board[fx][fy] = -color
            if len(fix_pos_list) == 1:
                ind = coordinate2index(fix_pos_list[0], self._board_size)
                pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=-color, last_move=pos_list[0], pi=pi,
                       z=0)  # last move should be next to an empty position
            i += 1
        return record

    @log
    def generate_dead_4_ooo_o_defend(self, sample_num=10000):
        record = GameRecord()
        for _ in range(sample_num):
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, _ = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list = [pos_list[3]]

            for x, y in pos_list:
                board[x][y] = color
            board[pos_list[3][0]][pos_list[3][1]] = 0

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])

            ind = coordinate2index(pos_list[3], self._board_size)
            pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=-color, last_move=pos_list[1], pi=pi,
                       z=0)  # last move should be next to an empty position
        return record

    @log
    def generate_dead_4_oo_oo_defend(self, sample_num=10000):
        record = GameRecord()
        for _ in range(sample_num):
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, _ = self._generate_consecutive_line(consecutive_num=5)
            fix_pos_list = [pos_list[2]]

            for x, y in pos_list:
                board[x][y] = color
            board[pos_list[2][0]][pos_list[2][1]] = 0

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])

            ind = coordinate2index(pos_list[2], self._board_size)
            pi[ind] = 1

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=-color, last_move=pos_list[1], pi=pi,
                       z=0)  # last move should be next to an empty position
        return record

    @log
    def generate_live_3_ooo_attack(self, sample_num=10000):
        record = GameRecord()
        i = 0
        while i < sample_num:
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=3)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            for x, y in pos_list:
                board[x][y] = color

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])
            ind_1 = coordinate2index(fix_pos_list[0], self._board_size)
            ind_2 = coordinate2index(fix_pos_list[1], self._board_size)
            pi[ind_1], pi[ind_2] = 0.5, 0.5

            self._add_noise(board=board, next_player=color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=color, last_move=pos_list[1], pi=pi, z=1)
            i += 1
        return record

    @log
    def generate_live_3_oo_o_attack(self, sample_num=10000):
        record = GameRecord()
        i = 0
        while i < sample_num:
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            fix_pos_list.append(list(pos_list[2]))

            for x, y in pos_list:
                board[x][y] = color
            board[pos_list[2][0]][pos_list[2][1]] = 0

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])
            ind = coordinate2index(pos_list[2], self._board_size)
            pi[ind] = 1

            self._add_noise(board=board, next_player=color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=color, last_move=pos_list[1], pi=pi, z=1)
            i += 1
        return record

    @log
    def generate_live_3_ooo_defend(self, sample_num=10000):
        record = GameRecord()
        i = 0
        while i < sample_num:
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=3)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            for x, y in pos_list:
                board[x][y] = color

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])
            ind_1 = coordinate2index(fix_pos_list[0], self._board_size)
            ind_2 = coordinate2index(fix_pos_list[1], self._board_size)
            pi[ind_1], pi[ind_2] = 0.5, 0.5
            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=-color, last_move=pos_list[1], pi=pi,
                       z=0)  # last move should be next to an empty position
            i += 1
        return record

    @log
    def generate_live_3_oo_o_defend(self, sample_num=10000):
        record = GameRecord()
        i = 0
        while i < sample_num:
            color = np.random.random_integers(0, 1) * 2 - 1
            board = self._empty_board()
            pos_list, fix_pos_list = self._generate_consecutive_line(consecutive_num=4)
            if len(fix_pos_list) == 0 or len(fix_pos_list) == 1:
                continue

            fix_pos_list.append(list(pos_list[2]))

            for x, y in pos_list:
                board[x][y] = color
            board[pos_list[2][0]][pos_list[2][1]] = 0

            pi = np.array([0.0 for _ in range(self._board_size ** 2)])
            ind_1 = coordinate2index(fix_pos_list[0], self._board_size)
            ind_2 = coordinate2index(fix_pos_list[1], self._board_size)
            ind_3 = coordinate2index(fix_pos_list[2], self._board_size)
            pi[ind_1], pi[ind_2], pi[ind_3] = 0.25, 0.25, 0.5

            self._add_noise(board=board, next_player=-color, max_stone_num=self._max_noise_stone_num,
                            fix_pos_list=fix_pos_list)

            record.add(obs=board, color=-color, last_move=pos_list[1], pi=pi,
                       z=0)  # last move should be next to an empty position
            i += 1
        return record

    def _generate_consecutive_line(self, consecutive_num):
        start_pos = np.random.random_integers(0, self._board_size - 1, 2)
        end_pos = [-1, -1]
        while end_pos[0] < 0 or end_pos[0] > 14 or end_pos[1] < 0 or end_pos[1] > 14:
            dx, dy = list(np.random.random_integers(-1, 1, 2))
            if dx == 0 and dy == 0:
                continue
            end_pos[0] = start_pos[0] + (consecutive_num - 1) * dx
            end_pos[1] = start_pos[1] + (consecutive_num - 1) * dy
        fix_pos_list = []
        if dx == 0:
            x_list = [start_pos[0]] * consecutive_num
        else:
            x_list = list(range(start_pos[0], end_pos[0] + dx, dx))
        if dy == 0:
            y_list = [start_pos[1]] * consecutive_num
        else:
            y_list = list(range(start_pos[1], end_pos[1] + dy, dy))

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
        stone_num = np.random.random_integers(30, max_stone_num)
        black_stone_ind = np.where(board == BLACK)
        white_stone_ind = np.where(board == WHITE)
        black_stone_num = len(black_stone_ind[0])
        white_stone_num = len(white_stone_ind[0])
        black_origin, white_origin = black_stone_num, white_stone_num

        delta = black_stone_num - white_stone_num
        # 假设下一步轮到黑棋走，要放x个黑棋，y个白棋，则x+b=y+w, x+y=stone_num
        # x-y=-delta, 2x=stone_num-delta
        # 假设下一步轮到白棋走，要放x个黑棋，y个白棋，则x+b+1=y+w, x+y=stone_num
        # x-y=-delta-1, 2x=stone_num-delta-1

        if next_player == BLACK:
            black_stone_num = int((stone_num - delta) / 2)
            white_stone_num = black_stone_num + delta
            if black_stone_num + black_origin > white_stone_num + white_origin:
                white_stone_num += 1
        else:
            black_stone_num = int((stone_num - delta - 1) / 2)
            white_stone_num = black_stone_num + delta
            if black_stone_num + black_origin == white_stone_num + white_origin:
                black_stone_num += 1

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
