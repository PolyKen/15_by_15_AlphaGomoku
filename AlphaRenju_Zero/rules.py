import numpy as np

BLACK = 1
WHITE = -1


class Rules:

    def __init__(self, conf):
        self._board_size = conf['board_size']
        self._board = [[0 for j in range(conf['board_size'])] for i in range(conf['board_size'])]
        self._conf = conf

        # The list that records the locations of all live3 and live4
        self._live3_list = []
        self._live4_list = []

        self._stone_number = 0

    def _read(self, board):
        self._board = board

    """The function that returns the current playboard"""    
    def board(self):
        return self._board
        
    """The function that helps the implementation of count_consecutive function"""
    def _count_on_direction(self, i, j, i_direction, j_direction, color):
        # Note: idirection has three options , i.e. -1,0,and 1 ,where -1 for up
        # 1 for down and 0 for unchange. jdirection also has three options -1,0,1
        # ,where -1 for left, 1 for right and 0 for unchange. We should mention that
        # idirection and jdirection can't equal to 0 at the same time.
        # We count the number of consecutive stones with the color given to function
        # in the given direction.
        count = 0
        for step in range(1,5):  # We only needs to consider stones within 4 steps
            if i + step * i_direction < 0 or i + step * i_direction > self._board_size - 1 or j + step * j_direction < 0 or j + step * j_direction > self._board_size - 1:
                break
            if self._board[i + step * i_direction][j + step * j_direction] == color:
                count += 1
            else:
                break
        return count
    
    """The function that returns the maximal number of consecutive stones"""
    def _count_consecutive(self, i, j, color):
        # The purpose of the function is to check if the consecutive number reaches 5 so that
        # the Renju can end or if it is more than 5 so that the forbidden overline is formed(Black can't overline)
        lines = [[[0, 1], [0, -1]], [[1, -1], [-1, 1]], [[1, 0], [-1, 0]], [[-1, -1], [1, 1]]]
        counts = []
        for line in lines:
            count = 1
            for direction in line:
                count += self._count_on_direction(i, j, direction[0], direction[1], color)
            counts.append(count)
        return 5 if 5 in counts else max(counts)
        # Note: the reason why we return this is that once 5 is found, then overline is ignored
    
    """The function that updates all the locations of live3 pattern"""
    """We use fliters to capture features of board"""
    def _update_live3_list(self):
        M = -10
        N = -20
        L = -50
        feature_A = [M, 1, 1, 1, N]
        feature_B = [M, 1, 1, N, 1, L]
        feature_C = [M, 1, N, 1, 1, L]

        self._live3_list.clear()

        # A，horizontal and vertical
        for i in range(15):
            for j in range(11):
                u = self._board[i][j : j + 5]
                v = [self._board[k][i] for k in range(j, j + 5)]
                flag_H = self._dot(u, feature_A)
                flag_V = self._dot(v, feature_A)
                if flag_H == 3:
                    pos = [[i, k] for k in range(j + 1, j + 4)]
                    self._live3_list.append(pos)
                if flag_V == 3:
                    pos = [[k, i] for k in range(j + 1, j + 4)]
                    self._live3_list.append(pos)

        # A, diagonal
        for i in range(11):
            for j in range(11):
                u = [self._board[i + k][j + k] for k in range(5)]
                v = [self._board[i + k][14 - j - k] for k in range(5)]
                flag_L = self._dot(u, feature_A)
                flag_R = self._dot(v, feature_A)
                if flag_L == 3:
                    pos = [[i + k, j + k] for k in range(1, 4)]
                    self._live3_list.append(pos)
                if flag_R == 3:
                    pos = [[i + k, 14 - j - k] for k in range(1, 4)]
                    self._live3_list.append(pos)

        # B, horizontal and vertical
        for i in range(15):
            for j in range(10):
                u = self._board[i][j: j + 6]
                v = [self._board[k][i] for k in range(j, j + 6)]
                flag_H = self._dot(u, feature_B)
                flag_V = self._dot(v, feature_B)
                if flag_H == 3:
                    pos = [[i, j + 1], [i, j + 2], [i, j + 4]]
                    self._live3_list.append(pos)
                if flag_V == 3:
                    pos = [[j + 1, i], [j + 2, i], [j + 4, i]]
                    self._live3_list.append(pos)

        # B, diagonal
        for i in range(10):
            for j in range(10):
                u = [self._board[i + k][j + k] for k in range(6)]
                v = [self._board[i + k][14 - j - k] for k in range(6)]
                flag_L = self._dot(u, feature_B)
                flag_R = self._dot(v, feature_B)
                if flag_L == 3:
                    pos = [[i + 1, j + 1], [i + 2, j + 2], [i + 4, j + 4]]
                    self._live3_list.append(pos)
                if flag_R == 3:
                    pos = [[i + 1, 13 - j], [i + 2, 12 - j], [i + 4, 10 - j]]
                    self._live3_list.append(pos)

        # C, horizontal and vertical
        for i in range(15):
            for j in range(10):
                u = self._board[i][j: j + 6]
                v = [self._board[k][i] for k in range(j, j + 6)]
                flag_H = self._dot(u, feature_C)
                flag_V = self._dot(v, feature_C)
                if flag_H == 3:
                    pos = [[i, j + 1], [i, j + 3], [i, j + 4]]
                    self._live3_list.append(pos)
                if flag_V == 3:
                    pos = [[j + 1, i], [j + 3, i], [j + 4, i]]
                    self._live3_list.append(pos)

        # C, diagonal
        for i in range(10):
            for j in range(10):
                u = [self._board[i + k][j + k] for k in range(6)]
                v = [self._board[i + k][14 - j - k] for k in range(6)]
                flag_L = self._dot(u, feature_C)
                flag_R = self._dot(v, feature_C)
                if flag_L == 3:
                    pos = [[i + 1, j + 1], [i + 3, j + 3], [i + 4, j + 4]]
                    self._live3_list.append(pos)
                if flag_R == 3:
                    pos = [[i + 1, 13 - j], [i + 3, 11 - j], [i + 4, 10 - j]]
                    self._live3_list.append(pos)
    
    """The function that updates all the locations of live4 pattern"""
    """We use fliters to capture features of board"""
    def _update_live4_list(self):
        M = -100
        feature_A = [M, 1, 1, 1, 1, M] # 4 or 104
        feature_B = [0, 1, 1, M, 1, 1, 0] # 4 only
        feature_C = [0, 1, M, 1, 1, 1, 0] # 4 only
        feature_D = [0, 1, 1, 1, M, 1, 0] # 4 only
        feature_E = [1, 1, 1, 1, M] # 4 only
        feature_F = [M, 1, 1, 1, 1] # 4 only
        feature_G = [1, 1, 1, M, 1] # 4 only
        feature_H = [1, M, 1, 1, 1] # 4 only
        feature_I = [1, 1, M, 1, 1] # 4 only

        self._live4_list.clear()

        # A，horizontal and vertical, feature_A = [M, 1, 1, 1, 1, M]
        for i in range(15):
            for j in range(10):
                u = self._board[i][j: j + 6]
                v = [self._board[k][i] for k in range(j, j + 6)]
                flag_H = self._dot(u, feature_A)
                flag_V = self._dot(v, feature_A)
                if flag_H == 4 or flag_H == 104:
                    pos = [[i, k] for k in range(j + 1, j + 5)]
                    self._live4_list.append(pos)
                if flag_V == 4 or flag_V == 104:
                    pos = [[k, i] for k in range(j + 1, j + 5)]
                    self._live4_list.append(pos)

        # A, diagonal
        for i in range(10):
            for j in range(10):
                u = [self._board[i + k][j + k] for k in range(6)]
                v = [self._board[i + k][14 - j - k] for k in range(6)]
                flag_L = self._dot(u, feature_A)
                flag_R = self._dot(v, feature_A)
                if flag_L == 4 or flag_L == 104:
                    pos = [[i + k, j + k] for k in range(1, 5)]
                    self._live4_list.append(pos)
                if flag_R == 4 or flag_R == 104:
                    pos = [[i + k, 14 - j - k] for k in range(1, 5)]
                    self._live4_list.append(pos)

        # B, horizontal and vertical,  feature_B = [0, 1, 1, M, 1, 1, 0]
        for i in range(15):
            for j in range(9):
                u = self._board[i][j: j + 7]
                v = [self._board[k][i] for k in range(j, j + 7)]
                flag_H = self._dot(u, feature_B)
                flag_V = self._dot(v, feature_B)
                if flag_H == 4:
                    pos = [[i, j + 1], [i, j + 2], [i, j + 4], [i, j + 5]]
                    self._live4_list.append(pos)
                if flag_V == 4:
                    pos = [[j + 1, i], [j + 2, i], [j + 4, i], [j + 5, i]]
                    self._live4_list.append(pos)

        # B, diagonal
        for i in range(9):
            for j in range(9):
                u = [self._board[i + k][j + k] for k in range(7)]
                v = [self._board[i + k][14 - j - k] for k in range(7)]
                flag_L = self._dot(u, feature_B)
                flag_R = self._dot(v, feature_B)
                if flag_L == 4:
                    pos = [[i + 1, j + 1], [i + 2, j + 2], [i + 4, j + 4], [i + 5, j + 5]]
                    self._live4_list.append(pos)
                if flag_R == 4:
                    pos = [[i + 1, 13 - j], [i + 2, 12 - j], [i + 4, 10 - j], [i + 5, 9 - j]]
                    self._live4_list.append(pos)

        # C, horizontal and vertical,  feature_C = [0, 1, M, 1, 1, 1, 0]
        for i in range(15):
            for j in range(9):
                u = self._board[i][j: j + 7]
                v = [self._board[k][i] for k in range(j, j + 7)]
                flag_H = self._dot(u, feature_C)
                flag_V = self._dot(v, feature_C)
                if flag_H == 4:
                    pos = [[i, j + 1], [i, j + 3], [i, j + 4], [i, j + 5]]
                    self._live4_list.append(pos)
                if flag_V == 4:
                    pos = [[j + 1, i], [j + 3, i], [j + 4, i], [j + 5, i]]
                    self._live4_list.append(pos)

        # C, diagonal
        for i in range(9):
            for j in range(9):
                u = [self._board[i + k][j + k] for k in range(7)]
                v = [self._board[i + k][14 - j - k] for k in range(7)]
                flag_L = self._dot(u, feature_C)
                flag_R = self._dot(v, feature_C)
                if flag_L == 4:
                    pos = [[i + 1, j + 1], [i + 3, j + 3], [i + 4, j + 4], [i + 5, j + 5]]
                    self._live4_list.append(pos)
                if flag_R == 4:
                    pos = [[i + 1, 13 - j], [i + 3, 11 - j], [i + 4, 10 - j], [i + 5, 9 - j]]
                    self._live4_list.append(pos)

        # D, horizontal and vertical,  feature_D = [0, 1, 1, 1, M, 1, 0]
        for i in range(15):
            for j in range(9):
                u = self._board[i][j: j + 7]
                v = [self._board[k][i] for k in range(j, j + 7)]
                flag_H = self._dot(u, feature_D)
                flag_V = self._dot(v, feature_D)
                if flag_H == 4:
                    pos = [[i, j + 1], [i, j + 2], [i, j + 3], [i, j + 5]]
                    self._live4_list.append(pos)
                if flag_V == 4:
                    pos = [[j + 1, i], [j + 2, i], [j + 3, i], [j + 5, i]]
                    self._live4_list.append(pos)

        # D, diagonal
        for i in range(9):
            for j in range(9):
                u = [self._board[i + k][j + k] for k in range(7)]
                v = [self._board[i + k][14 - j - k] for k in range(7)]
                flag_L = self._dot(u, feature_D)
                flag_R = self._dot(v, feature_D)
                if flag_L == 4:
                    pos = [[i + 1, j + 1], [i + 2, j + 2], [i + 3, j + 3], [i + 5, j + 5]]
                    self._live4_list.append(pos)
                if flag_R == 4:
                    pos = [[i + 1, 13 - j], [i + 2, 12 - j], [i + 3, 11 - j], [i + 5, 9 - j]]
                    self._live4_list.append(pos)

        # E feature_E = [1, 1, 1, 1, M]
        # horizontal and vertical, boundry
        for i in range(15):
            u1 = self._board[i][0:5]
            u2 = [self._board[j][i] for j in range(5)]
            if self._dot(u1, feature_E) == 4:
                pos = [[i, k] for k in range(4)]
                self._live4_list.append(pos)
            if self._dot(u2, feature_E) == 4:
                pos = [[k, i] for k in range(4)]
                self._live4_list.append(pos)

        # diagonal, from row 1 to row 5. Noted that there is no need for the row 11 to row 15 since
        # it can be detected by feature (diagonal) except two special position. Similarly, for
        # feature F there is no need for row 1 to row 5 except two special position.
        # diagonal (up border)
        for i in range(11):
            u3 = [self._board[k][i + k] for k in range(5)]
            u4 = [self._board[k][i - k + 4] for k in range(5)]
            if self._dot(u3, feature_E) == 4:
                pos = [[k, i + k] for k in range(4)]
                self._live4_list.append(pos)
            if self._dot(u4, feature_E) == 4:
                pos = [[k, i - k + 4] for k in range(4)]
                self._live4_list.append(pos)

        # diagonal (left and right border)
        for i in range(1, 11):
            u5 = [self._board[i + k][k] for k in range(5)]
            u6 = [self._board[i + k][14 - k] for k in range(5)]
            if self._dot(u5, feature_E) == 4:
                pos = [[i + k, k] for k in range(4)]
                self._live4_list.append(pos)
            if self._dot(u6, feature_E) == 4:
                pos = [[i + k, 14 - k] for k in range(4)]
                self._live4_list.append(pos)

        # F feature_F = [M, 1, 1, 1, 1], similar to feature_E
        # horizontal and vertical, right and bottom border
        for i in range(15):
            u1 = self._board[i][10:15]
            u2 = [self._board[j][i] for j in range(10, 15)]
            if self._dot(u1, feature_F) == 4:
                pos = [[i, k] for k in range(11, 15)]
                self._live4_list.append(pos)
            if self._dot(u2, feature_F) == 4:
                pos = [[k, i] for k in range(11, 15)]
                self._live4_list.append(pos)

        # diagonal, bottom border
        for i in range(11):
            u3 = [self._board[k + 10][i + k] for k in range(5)]
            u4 = [self._board[k + 10][i - k + 4] for k in range(5)]
            if self._dot(u3, feature_F) == 4:
                pos = [[k + 10, i + k] for k in range(1, 5)]
                self._live4_list.append(pos)
            if self._dot(u4, feature_F) == 4:
                pos = [[k + 10, i - k + 4] for k in range(1, 5)]
                self._live4_list.append(pos)

        # diagonal, left and right border
        for i in range(1, 11):
            u5 = [self._board[i + k][10 + k] for k in range(5)]
            u6 = [self._board[i + k][4 - k] for k in range(5)]
            if self._dot(u5, feature_F) == 4:
                pos = [[i + k, 10 + k] for k in range(1, 5)]
                self._live4_list.append(pos)
            if self._dot(u6, feature_F) == 4:
                pos = [[i + k, 4 - k] for k in range(1, 5)]
                self._live4_list.append(pos)

        # G feature_G = [1, 1, 1, M, 1]
        # horizontal and vertical
        for i in range(15):
            u1 = self._board[i][0:5]
            u2 = self._board[i][10:15]
            u3 = [self._board[k][i] for k in range(5)]
            u4 = [self._board[k][i] for k in range(10, 15)]
            if self._dot(u1, feature_G) == 4:
                pos = [[i, 0], [i, 1], [i, 2], [i, 4]]
                self._live4_list.append(pos)
            if self._dot(u2, feature_G) == 4:
                pos = [[i, 10], [i, 11], [i, 12], [i, 14]]
                self._live4_list.append(pos)
            if self._dot(u3, feature_G) == 4:
                pos = [[0, i], [1, i], [2, i], [4, i]]
                self._live4_list.append(pos)
            if self._dot(u4, feature_G) == 4:
                pos = [[10, i], [11, i], [12, i], [14, i]]
                self._live4_list.append(pos)

        # diagonal. repetition exists so it's better to pick them out: u13-u16
        for i in range(10):
            u5 = [self._board[i + k + 1][k] for k in range(5)]
            u6 = [self._board[k][i + k + 1] for k in range(5)]
            u7 = [self._board[i + k][k + 10] for k in range(5)]
            u8 = [self._board[k + 10][i + k] for k in range(5)]
            u9 = [self._board[k][i + 4 - k] for k in range(5)]
            u10 = [self._board[i + k + 1][14 - k] for k in range(5)]
            u11 = [self._board[k + 10][i + 5 - k] for k in range(5)]
            u12 = [self._board[i + k][4 - k] for k in range(5)]
            if self._dot(u5, feature_G) == 4:
                pos = [[i + 1, 0],[i + 2, 1], [i + 3, 2], [i + 5, 4]]
                self._live4_list.append(pos)
            if self._dot(u6, feature_G) == 4:
                pos = [[0, i + 1],[1, i + 2], [2, i + 3], [4, i + 5]]
                self._live4_list.append(pos)
            if self._dot(u7, feature_G) == 4:
                pos = [[i, 10], [i + 1, 11], [i + 2, 12], [i + 4, 14]]
                self._live4_list.append(pos)
            if self._dot(u8, feature_G) == 4:
                pos = [[10, i], [11, i + 1], [12, i + 2], [14, i + 4]]
                self._live4_list.append(pos)
            if self._dot(u9, feature_G) == 4:
                pos = [[0, i + 4], [1, i + 3], [2, i + 2], [4, i]]
                self._live4_list.append(pos)
            if self._dot(u10, feature_G) == 4:
                pos = [[i + 1, 14], [i + 2, 13], [i + 3, 12], [i + 5, 10]]
                self._live4_list.append(pos)
            if self._dot(u11, feature_G) == 4:
                pos = [[10, i + 5], [11, i + 4], [12, i + 3], [14, i + 1]]
                self._live4_list.append(pos)
            if self._dot(u12, feature_G) == 4:
                pos = [[i, 4], [i + 1, 3], [i + 2, 2], [i + 4, 0]]
                self._live4_list.append(pos)

        u13 = [self._board[k][k] for k in range(5)]
        u14 = [self._board[k][k] for k in range(10, 15)]
        u15 = [self._board[k][14 - k] for k in range(5)]
        u16 = [self._board[k][14 - k] for k in range(10, 15)]

        if self._dot(u13, feature_G) == 4:
            pos = [[0, 0], [1, 1], [2, 2], [4, 4]]
            self._live4_list.append(pos)
        if self._dot(u14, feature_G) == 4:
            pos = [[10, 10], [11, 11], [12, 12], [14, 14]]
            self._live4_list.append(pos)
        if self._dot(u15, feature_G) == 4:
            pos = [[0, 14], [1, 13], [2, 12], [4, 10]]
            self._live4_list.append(pos)
        if self._dot(u16, feature_G) == 4:
            pos = [[10, 4], [11, 3], [12, 2], [14, 0]]
            self._live4_list.append(pos)

        # H feature_H = [1, M, 1, 1, 1]
        for i in range(15):
            u1 = self._board[i][0:5]
            u2 = self._board[i][10:15]
            u3 = [self._board[k][i] for k in range(5)]
            u4 = [self._board[k][i] for k in range(10, 15)]
            if self._dot(u1, feature_H) == 4:
                pos = [[i, 0], [i, 2], [i, 3], [i, 4]]
                self._live4_list.append(pos)
            if self._dot(u2, feature_H) == 4:
                pos = [[i, 10], [i, 12], [i, 13], [i, 14]]
                self._live4_list.append(pos)
            if self._dot(u3, feature_H) == 4:
                pos = [[0, i], [2, i], [3, i], [4, i]]
                self._live4_list.append(pos)
            if self._dot(u4, feature_H) == 4:
                pos = [[10, i], [12, i], [13, i], [14, i]]
                self._live4_list.append(pos)

        for i in range(10):
            u5 = [self._board[i + k + 1][k] for k in range(5)]
            u6 = [self._board[k][i + k + 1] for k in range(5)]
            u7 = [self._board[i + k][k + 10] for k in range(5)]
            u8 = [self._board[k + 10][i + k] for k in range(5)]
            u9 = [self._board[k][i + 4 - k] for k in range(5)]
            u10 = [self._board[i + k + 1][14 - k] for k in range(5)]
            u11 = [self._board[k + 10][i + 5 - k] for k in range(5)]
            u12 = [self._board[i + k][4 - k] for k in range(5)]
            if self._dot(u5, feature_H) == 4:
                pos = [[i + 1, 0],[i + 3, 2], [i + 4, 3], [i + 5, 4]]
                self._live4_list.append(pos)
            if self._dot(u6, feature_H) == 4:
                pos = [[0, i + 1],[2, i + 3], [3, i + 4], [4, i + 5]]
                self._live4_list.append(pos)
            if self._dot(u7, feature_H) == 4:
                pos = [[i, 10], [i + 2, 12], [i + 3, 13], [i + 4, 14]]
                self._live4_list.append(pos)
            if self._dot(u8, feature_H) == 4:
                pos = [[10, i], [12, i + 2], [13, i + 3], [14, i + 4]]
                self._live4_list.append(pos)
            if self._dot(u9, feature_H) == 4:
                pos = [[0, i + 4], [2, i + 2], [3, i + 1], [4, i]]
                self._live4_list.append(pos)
            if self._dot(u10, feature_H) == 4:
                pos = [[i + 1, 14], [i + 3, 12], [i + 4, 11], [i + 5, 10]]
                self._live4_list.append(pos)
            if self._dot(u11, feature_H) == 4:
                pos = [[10, i + 5], [12, i + 3], [13, i + 2], [14, i + 1]]
                self._live4_list.append(pos)
            if self._dot(u12, feature_H) == 4:
                pos = [[i, 4], [i + 2, 2], [i + 3, 1], [i + 4, 0]]
                self._live4_list.append(pos)

        u13 = [self._board[k][k] for k in range(5)]
        u14 = [self._board[k][k] for k in range(10, 15)]
        u15 = [self._board[k][14 - k] for k in range(5)]
        u16 = [self._board[k][14 - k] for k in range(10, 15)]

        if self._dot(u13, feature_H) == 4:
            pos = [[0, 0], [2, 2], [3, 3], [4, 4]]
            self._live4_list.append(pos)
        if self._dot(u14, feature_H) == 4:
            pos = [[10, 10], [12, 12], [13, 13], [14, 14]]
            self._live4_list.append(pos)
        if self._dot(u15, feature_H) == 4:
            pos = [[0, 14], [2, 12], [3, 11], [4, 10]]
            self._live4_list.append(pos)
        if self._dot(u16, feature_H) == 4:
            pos = [[10, 4], [12, 2], [13, 1], [14, 0]]
            self._live4_list.append(pos)

        # I feature_I = [1, 1, M, 1, 1] # 4 only
        for i in range(15):
            u1 = self._board[i][0:5]
            u2 = self._board[i][10:15]
            u3 = [self._board[k][i] for k in range(5)]
            u4 = [self._board[k][i] for k in range(10, 15)]
            if self._dot(u1, feature_I) == 4:
                pos = [[i, 0], [i, 1], [i, 3], [i, 4]]
                self._live4_list.append(pos)
            if self._dot(u2, feature_I) == 4:
                pos = [[i, 10], [i, 11], [i, 13], [i, 14]]
                self._live4_list.append(pos)
            if self._dot(u3, feature_I) == 4:
                pos = [[0, i], [1, i], [3, i], [4, i]]
                self._live4_list.append(pos)
            if self._dot(u4, feature_I) == 4:
                pos = [[10, i], [11, i], [13, i], [14, i]]
                self._live4_list.append(pos)

        for i in range(10):
            u5 = [self._board[i + k + 1][k] for k in range(5)]
            u6 = [self._board[k][i + k + 1] for k in range(5)]
            u7 = [self._board[i + k][k + 10] for k in range(5)]
            u8 = [self._board[k + 10][i + k] for k in range(5)]
            u9 = [self._board[k][i + 4 - k] for k in range(5)]
            u10 = [self._board[i + k + 1][14 - k] for k in range(5)]
            u11 = [self._board[k + 10][i + 5 - k] for k in range(5)]
            u12 = [self._board[i + k][4 - k] for k in range(5)]
            if self._dot(u5, feature_I) == 4:
                pos = [[i + 1, 0],[i + 2, 1], [i + 4, 3], [i + 5, 4]]
                self._live4_list.append(pos)
            if self._dot(u6, feature_I) == 4:
                pos = [[0, i + 1],[1, i + 2], [3, i + 4], [4, i + 5]]
                self._live4_list.append(pos)
            if self._dot(u7, feature_I) == 4:
                pos = [[i, 10], [i + 1, 11], [i + 3, 13], [i + 4, 14]]
                self._live4_list.append(pos)
            if self._dot(u8, feature_I) == 4:
                pos = [[10, i], [11, i + 1], [13, i + 3], [14, i + 4]]
                self._live4_list.append(pos)
            if self._dot(u9, feature_I) == 4:
                pos = [[0, i + 4], [1, i + 3], [3, i + 1], [4, i]]
                self._live4_list.append(pos)
            if self._dot(u10, feature_I) == 4:
                pos = [[i + 1, 14], [i + 2, 13], [i + 4, 11], [i + 5, 10]]
                self._live4_list.append(pos)
            if self._dot(u11, feature_I) == 4:
                pos = [[10, i + 5], [11, i + 4], [13, i + 2], [14, i + 1]]
                self._live4_list.append(pos)
            if self._dot(u12, feature_I) == 4:
                pos = [[i, 4], [i + 1, 3], [i + 3, 1], [i + 4, 0]]
                self._live4_list.append(pos)

        u13 = [self._board[k][k] for k in range(5)]
        u14 = [self._board[k][k] for k in range(10, 15)]
        u15 = [self._board[k][14 - k] for k in range(5)]
        u16 = [self._board[k][14 - k] for k in range(10, 15)]

        if self._dot(u13, feature_I) == 4:
            pos = [[0, 0], [1, 1], [3, 3], [4, 4]]
            self._live4_list.append(pos)
        if self._dot(u14, feature_I) == 4:
            pos = [[10, 10], [11, 11], [13, 13], [14, 14]]
            self._live4_list.append(pos)
        if self._dot(u15, feature_I) == 4:
            pos = [[0, 14], [1, 13], [3, 11], [4, 10]]
            self._live4_list.append(pos)
        if self._dot(u16, feature_I) == 4:
            pos = [[10, 4], [11, 3], [13, 1], [14, 0]]
            self._live4_list.append(pos)
           
    """The function checks for double three which is forbidden for Black"""
    def _check_forbidden_moves(self):
        if not self._conf['forbidden_moves']:
            return False
        self._update_live3_list()
        self._update_live4_list()
        count_valid_live3 = len(self._live3_list)
        count_valid_live4 = len(self._live4_list)
        for live3 in self._live3_list:
            for live4 in self._live4_list:
                if any([live3 == live4[i:i+len(live3)] for i in range(len(live4)-len(live3)+1)]):  # live3 in live4! should be viewed as live4
                    count_valid_live3 -= 1
                else:
                    continue
        if (count_valid_live3 >= 2) or (count_valid_live4 >= 2): #Found double-three or double-four! Forbidden for Black!
            return True
        else:
            return False
          
    """This is the function that """
    def check_rules(self, board, action, color):
        self._read(board)
        i = action[0]
        j = action[1]

        if self._board[i][j] != 0:
            return 'occupied'

        self._board[i][j] = color

        if color == BLACK:  # Black Player
            # Check overline and winning pattern
            count = self._count_consecutive(i, j, color)  # Count the maximal consecutive number
            if count >= 5:
                if count == 5:
                    # Winning pattern for Black
                    # print("live3 = " + str(self._live3_list))
                    # print("live4 = " + str(self._live4_list))
                    # print("C5")
                    return 'blackwins'
                else:
                    # Overline forbidden move for Black, Black loses the Game
                    # print("live3 = " + str(self._live3_list))
                    # print("live4 = " + str(self._live4_list))
                    if self._conf['forbidden_moves']:
                        # print("Forbidden Move: C6+")
                        return 'whitewins'
                    else:
                        # print('C5')
                        return 'blackwins'
            # Check double three and double four
            signal = self._check_forbidden_moves()
            if signal:
                # If we find the forbidden moves, then White wins
                # print("live3 = " + str(self._live3_list))
                # print("live4 = " + str(self._live4_list))
                # print("Forbidden Move: D3 D4")
                return 'whitewins'
        else:  # White Player , i.e. color == -1
            count = self._count_consecutive(i, j, color)  # Count the maximal consecutive number
            if count >= 5:
                # Winning pattern for White
                # print("live3 = " + str(self._live3_list))
                # print("live4 = " + str(self._live4_list))
                # print("C5+")
                return 'whitewins'

        # If the board is full while we still don't have the winner , then draw
        if sum(sum(np.array(np.array(board) == 0, dtype=int))) == 0:
            return 'draw'

        return 'continue'

    def _dot(self, A, B):
        if len(A) != len(B):
            return 'error'
        s = 0
        for i in range(len(A)):
            s += A[i] * B[i]
        return s