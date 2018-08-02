import time
from ..rules import *
import numpy as np


class Board:
    def __init__(self, renderer, board_size=15):
        self._board = [[0 for i in range(board_size)] for i in range(board_size)]
        self._board_size = board_size
        self._player = BLACK
        self._winner = 0
        self._round = 0
        self._last_move = None

        if renderer is None:
            self._display = False
        else:
            self._display = True
        self._renderer = renderer

    def __str__(self):
        print('round = ' + str(self.round()))
        print('last move = ' + str(self.last_move()))
        if self.current_player() == BLACK:
            return 'current_player = BLACK'
        else:
            return 'current_player = WHITE'

    # return the board
    def board(self):
        return np.copy(self._board)

    # player take an action(coordinate)
    def move(self, player, action, info=None):
        x = action[0]   # row
        y = action[1]   # col

        # waiting until renderer is initialized
        while self._display and (not self._renderer.is_initialized()):
            time.sleep(.2)

        if not isinstance(x, int) or not isinstance(y, int):
            print("x, y should be an integer:", x, y)
            return 1, self.board()
        if x < 0 or x > self._board_size - 1 or y < 0 or y > self._board_size - 1:
            print("x, y should be in [0, 14]")
            return 1, self.board()

        if player == BLACK:
            if self._display:
                self._renderer.move(player, (x, y), info)
            self._board[x][y] = BLACK
            self._player = WHITE
            self._round += 1
        else:
            if self._display:
                self._renderer.move(player, (x, y), info)
            self._board[x][y] = WHITE
            self._player = BLACK

        self._last_move = action

    def clear(self):
        self._board = [[0 for i in range(self._board_size)] for i in range(self._board_size)]
        self._player = BLACK
        self._winner = 0
        self._round = 0
        self._last_move = None
        if self._display:
            self._renderer.paint_background()
            while not self._renderer.is_initialized():
                time.sleep(.1)

    def read(self, new_board):
        self.clear()
        black_num = 0
        white_num = 0

        for row in range(self._board_size):
            for col in range(self._board_size):
                if new_board[row][col] == BLACK:
                    self.move(1, (row, col))
                    black_num += 1
                elif new_board[row][col] == WHITE:
                    self.move(-1, (row, col))
                    white_num += 1

        self._round = black_num
        if black_num == white_num:
            self._player = BLACK
        elif black_num == white_num + 1:
            self._player = WHITE
        else:
            print("Illegal Position")

    def round(self):
        return self._round

    def current_player(self):
        return self._player

    def last_move(self):
        return self._last_move

    def stone_num(self):
        if self._player == BLACK:
            return 2*self._round
        else:
            return 2*self._round - 1


