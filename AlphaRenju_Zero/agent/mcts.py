from .node import Node
import numpy as np
from ..rules import *
import time
import asyncio


class MCTS:
    def __init__(self, conf, net, color, use_stochastic_policy):
        # hyperparameters
        self._c_puct = conf['c_puct']  # PUCT
        self._simulation_times = conf['simulation_times']  # number of simulation
        self._tau = conf['initial_tau']  # temperature parameter
        self._epsilon = conf['epsilon']  # proportion of dirichlet noise
        self._use_dirichlet = conf['use_dirichlet']
        self._alpha = conf['alpha']
        self._board_size = conf['board_size']
        self._color = color  # MCTS Agent's color ( 1 for black; -1 for white)

        self._root = Node(1.0, None, BLACK, conf['virtual_loss'])  # Monte Carlo tree

        self._network = net  # Residual neural network
        self._is_self_play = conf['is_self_play']
        self._use_stochastic_policy = use_stochastic_policy
        self._careful_stage = conf['careful_stage']

        self._loop = asyncio.get_event_loop()
        self._coroutine_num = conf['coroutine_num']
        self._virtual_loss = conf['virtual_loss']
        self._expanding_list = []

    def set_self_play(self, is_self_play):
        self._is_self_play = is_self_play

    def set_stochastic_policy(self, use_stochastic_policy):
        self._use_stochastic_policy = use_stochastic_policy

    def reset(self):
        self._root = Node(1.0, None, BLACK, self._virtual_loss)

    def action(self, board, last_action, stage):
        # so far the root corresponds to the last board = board - last_action
        if self._root.is_leaf():
            last_board = np.copy(board)
            if last_action is not None:
                row, col = last_action[0], last_action[1]
                last_board[row][col] = 0
            self._simulate(last_board, last_action)

        # move the root to the child (node) corresponding to the board
        if last_action is not None:
            last_action_ind = coordinate2index(last_action, self._board_size)
            self._root = self._root.children()[last_action_ind]

        # now the root corresponds to the board
        pi = self._predict(board, last_action)

        # action decision
        if self._use_stochastic_policy and stage <= self._careful_stage:  # stochastic policy
            position_list = [i for i in range(self._board_size * self._board_size)]
            action = np.random.choice(position_list, p=pi)
        else:  # deterministic policy
            action = np.argmax(pi)

        next_node = self._root.children()[action]
        prior_prob = next_node.P()
        value = next_node.value

        # Adjust the Root Node and discard the remainder of the tree
        if not self._is_self_play:
            self._root = self._root.children()[action]
        return action, pi, prior_prob, value  # You need to store pi for training use
    
    def _predict(self, board, last_move):
        self._simulate(board, last_move)
        pi = np.array([node.N**(1/self._tau) for node in self._root.children()])
        pi = pi / sum(pi)
        return pi

    def _simulate(self, root_board, last_move):
        tasks = [self._simulate_coroutine(root_board, last_move)] * self._coroutine_num
        self._loop.run_until_complete(asyncio.wait(tasks))

    async def _simulate_coroutine(self, root_board, last_move):    # ROOT BOARD MUST CORRESPOND TO THE ROOT NODE!!!
        legal_vec_root = board2legalvec(root_board)
        playouts = int(self._simulation_times / self._coroutine_num)
        for epoch in range(playouts):
            current_node = self._root
            legal_vec_current = np.copy(legal_vec_root)  # deep copy
            current_color = self._root.color
            current_board = np.copy(root_board)
            action = None

            while not current_node.is_leaf():
                current_node, action = current_node.select(self._c_puct, legal_vec_current)

                # add virtual loss in order to make other coroutines avoid this node
                current_node.select_num += 1
                current_node.N += self._virtual_loss

                # update legal vector
                legal_vec_current[action] = 0

                # update current board
                row, col = index2coordinate(action, self._board_size)
                current_board[row][col] = current_color

                # update current color
                current_color = -current_color

                # if current node is not a leaf node, then it can't be in expanding list.
                # if current node is a leaf node, it may be expanding in other coroutines, so here we wait until it
                # is expanded (so that it is no longer a leaf node)
                while current_node in self._expanding_list:
                    await asyncio.sleep(1e-4)

            # so far, current node must be a leaf node (including end node)
            if current_node.is_end:
                current_node.backup(-current_node.value)
                continue

            # add current node to expanding list
            self._expanding_list.append(current_node)

            # calculate the prior probabilities and value
            p, v = self._network.predict(current_board, current_color, last_move)
            current_node.value = v
            prior_prob = p[0]
            if self._use_dirichlet:
                alpha = [self._alpha] * (self._board_size * self._board_size)
                noise = np.random.dirichlet(alpha)
                prior_prob = (1-self._epsilon) * prior_prob + self._epsilon * noise

            # now check whether this leaf node is an end node or not
            if action is not None:
                end_flag = check_rules(current_board, action, -current_color)
                if end_flag == 'blackwins' or end_flag == 'whitewins' or end_flag == 'full':
                    current_node.is_end = True
                    if end_flag == 'full':
                        current_node.value = 0
                    else:
                        current_node.value = -1
                else:
                    current_node.expand(prior_prob, self._board_size)
            else:
                # if action is None, then the root node must be a leaf
                current_node.expand(prior_prob, self._board_size)

            self._expanding_list.remove(current_node)

            # backup
            current_node.backup(-current_node.value)


def index2coordinate(index, size):
    row = index // size
    col = index % size
    return int(row), int(col)


def coordinate2index(cor, size):
    return size * cor[0] + cor[1]


def board2legalvec(board):
    vec = np.array(np.array(board) == 0, dtype=np.int)
    return vec.flatten()


def check_rules(board, action, color):
    stone_num = sum(sum(np.abs(board)))
    if stone_num <= 8:  # Impossible to end since the maximal length of consecutive lines with the same color is four.
        return 'continue'
    else:
        if stone_num == board.shape[0] * board.shape[0]:
            return 'full'
        else:  # Greedy Match
            cor = index2coordinate(action, board.shape[0])
            # Horizontal Check
            count = 1
            for i in range(1, 5):
                if cor[1] + i <= board.shape[0] - 1:
                    if board[cor[0]][cor[1]+i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if cor[1] - i >= 0:
                    if board[cor[0]][cor[1]-i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            if count >= 5:
                if color == 1:
                    return 'blackwins'
                else:
                    return 'whitewins'
            # Vertical Check
            count = 1
            for i in range(1, 5):
                if cor[0]+i <= board.shape[0]-1:
                    if board[cor[0]+i][cor[1]] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if cor[0]-i >= 0:
                    if board[cor[0]-i][cor[1]] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            if count >= 5:
                if color == 1:
                    return 'blackwins'
                else:
                    return 'whitewins'
            # Diagonal Check
            count = 1
            for i in range(1, 5):
                if (cor[0]+i <= board.shape[0]-1) and (cor[1]+i <= board.shape[0]-1):
                    if board[cor[0]+i][cor[1]+i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if (cor[0]-i >= 0) and (cor[1]-i >= 0):
                    if board[cor[0]-i][cor[1]-i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            if count >= 5:
                if color == 1:
                    return 'blackwins'
                else:
                    return 'whitewins'
            # Anti-Diagonal Check
            count = 1
            for i in range(1, 5):
                if (cor[0]+i <= board.shape[0]-1) and (cor[1]-i >= 0):
                    if board[cor[0]+i][cor[1]-i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if (cor[0]-i >= 0) and (cor[1]+i <= board.shape[0]-1):
                    if board[cor[0]-i][cor[1]+i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            if count >= 5:
                if color == 1:
                    return 'blackwins'
                else:
                    return 'whitewins'
