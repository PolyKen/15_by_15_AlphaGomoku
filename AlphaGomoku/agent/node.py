from math import sqrt
import numpy as np
from ..config import *


class Node:
    count = 0
    backup_count = 0
    conflict_count = 0

    def __init__(self, prior_prob, parent, color, virtual_loss):

        # actually N, Q, W, U are properties of edge
        self.N = 0  # Number of visits
        self._Q = 0  # Quality of the edge
        self.W = 0  # Intermediate value for Q update
        self._P = prior_prob  # Prior probability predicted by network
        self._U = 0

        self._virtual_loss = virtual_loss
        self.select_num = 0

        self._parent = parent
        self._children = []  # if self._children is an empty list, it is viewed as a leaf node

        # when it is an end leaf
        self.is_end = False
        self.value = 0

        self.color = color  # color of next player
        self.num = Node.count
        Node.count += 1

    def P(self):
        return self._P

    def Q(self):
        return self._Q

    def U(self):
        return self._U

    def parent(self):
        return self._parent

    def children(self):
        return self._children

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return self._children == []

    def upper_confidence_bound(self, c_puct):
        try:
            self._U = c_puct * self._P * sqrt(self._parent.N) / (1 + self.N)
        except ValueError:
            print('> valueError: Node.upper_confidence_bound')
            print(self._U, self._P, self._parent.N, self.N)
        return self._U + self._Q

    def select(self, c_puct, legal_vec_current):
        ucb_list = np.array([node.upper_confidence_bound(c_puct) for node in self._children])
        ind = np.argsort(ucb_list)
        for i in range(len(ind)):
            if legal_vec_current[ind[-(i + 1)]] == 1:
                action = ind[-(i + 1)]
                break
        next_node = self._children[action]
        return next_node, action

    def expand(self, prior_prob, board_size=15):
        if not self.is_leaf():
            print('> error: node.expand')
            return
        for i in range(board_size * board_size):
            prob = prior_prob[i]
            self._children.append(Node(prob, self, -self.color, self._virtual_loss))

    def backup(self, value):
        # remove virtual loss
        if self.select_num > 0:
            self.select_num -= 1
            self.N -= self._virtual_loss
            if self.N < 0:
                self.N += self._virtual_loss

        self.N += 1
        self.W += value
        self._Q = self.W / self.N
        if not self.is_root():
            self._parent.backup(-decay * value)
