from .node import Node
import numpy as np
from ..rules import *
from ..utils import *
import time
import threading


class MCTS:
    def __init__(self, conf, black_net, white_net, color, use_stochastic_policy):
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

        self._black_net = black_net
        self._white_net = white_net

        self._is_self_play = conf['is_self_play']
        self._use_stochastic_policy = use_stochastic_policy
        self._careful_stage = conf['careful_stage']

        self._threading_num = conf['threading_num']
        self._virtual_loss = conf['virtual_loss']
        self._expanding_list = []

    def set_self_play(self, is_self_play):
        self._is_self_play = is_self_play

    def set_stochastic_policy(self, use_stochastic_policy):
        self._use_stochastic_policy = use_stochastic_policy

    def reset(self):
        self._root = Node(1.0, None, BLACK, self._virtual_loss)

    def action(self, board, last_action, stage):
        # step 1: rebase tree
        # so far the root corresponds to the last board = board - last_action
        # thus we need to find out the node that correspond to the argument [board]

        # if the current root is a leaf node, then we should simulate in advance
        if self._root.is_leaf():
            last_board = np.copy(board)
            # A special case: if the board is empty, then last_action is None
            if last_action is not None:
                row, col = last_action[0], last_action[1]
                last_board[row][col] = 0

            # now the last_board is correspond to the root
            self._simulate(last_board, last_action)

        # if the current root is not a leaf, then we can move the root to the child node correspond
        # to the board directly
        if last_action is not None:
            # last action might be None (when the board is empty)
            last_action_ind = coordinate2index(last_action, self._board_size)
            self._root = self._root.children()[last_action_ind]

        # now the root corresponds to the board
        original_pi, pi = self._predict(board, last_action)

        # action decision
        if self._use_stochastic_policy and stage <= self._careful_stage:  # stochastic policy
            position_list = [i for i in range(self._board_size * self._board_size)]
            action = np.random.choice(position_list, p=pi)
        else:  # deterministic policy
            action = np.argmax(pi)

        next_node = self._root.children()[action]
        prior_prob = next_node.P()
        value = next_node.value

        # adjust the root node and discard the remainder of the tree
        if not self._is_self_play:
            self._root = self._root.children()[action]

        return action, original_pi, prior_prob, value
        # return pi for training use

    def _predict(self, board, last_move):
        # now board correspond to the root, last_move is the last move of the board
        self._simulate(board, last_move)
        # generate the action distribution
        original_pi = np.array([node.N for node in self._root.children()])
        pi = np.array([node.N ** (1 / self._tau) for node in self._root.children()])
        if len(pi) != len(board) ** 2:
            print('>> error: MCTS._predict')
            print(len(pi))
            return
        original_pi /= sum(original_pi)
        pi /= sum(pi)

        return original_pi, pi

    # ROOT BOARD MUST CORRESPOND TO THE ROOT NODE!!!
    def _get_simulate_thread_target(self, root_board, last_move):
        def _simulate_thread():
            legal_vec_root = board2legalvec(root_board)
            each_simulation_times = int(self._simulation_times / self._threading_num)

            for epoch in range(each_simulation_times):
                # initiate the current node as the root node and initiate the current color as the color of the root
                current_node = self._root
                current_color = self._root.color

                legal_vec_current = np.copy(legal_vec_root)  # deep copy
                current_board = np.copy(root_board)

                # initiate select_action as last_move
                select_action = last_move

                # so far, root node might be a leaf (eg: root_board is empty)

                # if the root node is not a leaf, then it will enter the following loop
                while not current_node.is_leaf():
                    current_node, select_action_ind = current_node.select(self._c_puct, legal_vec_current)

                    # add virtual loss in order to make other threads avoid this node
                    current_node.select_num += 1
                    current_node.N += self._virtual_loss

                    # update legal vector
                    legal_vec_current[select_action_ind] = 0

                    # update current board
                    row, col = index2coordinate(select_action_ind, self._board_size)
                    current_board[row][col] = current_color
                    select_action = (row, col)

                    # update current color
                    current_color = -current_color

                    # if current node is not a leaf node, then it can't be in expanding list.
                    # if current node is a leaf node, it may be expanding in other thread, so here we wait until it
                    # is expanded (so that it is no longer a leaf node)
                    while current_node in self._expanding_list:
                        time.sleep(1e-4)

                # so far, current node must be a leaf node (including end node)
                if current_node.is_end:
                    current_node.backup(-current_node.value)
                    continue

                # add current node to expanding list
                if current_node not in self._expanding_list:
                    self._expanding_list.append(current_node)
                else:
                    continue

                # calculate the prior probabilities and value
                if current_color is BLACK:
                    net = self._black_net
                else:
                    net = self._white_net
                p, v = net.predict(board=current_board,
                                   color=current_color,
                                   last_move=select_action)
                current_node.value = v
                prior_prob = p[0]

                if self._use_dirichlet:
                    alpha = [self._alpha] * (self._board_size * self._board_size)
                    noise = np.random.dirichlet(alpha)
                    prior_prob = (1 - self._epsilon) * prior_prob + self._epsilon * noise

                # now check whether this leaf node is an end node or not
                if select_action is not None:
                    end_flag = check_rules(current_board, select_action, -current_color)
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

        return _simulate_thread

    def _simulate(self, root_board, last_move):
        target = self._get_simulate_thread_target(root_board, last_move)
        thread_list = []
        for i in range(self._threading_num):
            thr = threading.Thread(target=target, name='thread_' + str(i + 1))
            thr.start()
            thread_list.append(thr)
            time.sleep(1e-3)
        for thr in thread_list:
            thr.join()


def check_rules(board, action_cor, color):
    stone_num = sum(sum(np.abs(board)))
    if stone_num <= 8:  # Impossible to end since the maximal length of consecutive lines with the same color is four.
        return 'continue'
    else:
        if stone_num == board.shape[0] * board.shape[0]:
            return 'full'
        else:  # Greedy Match
            # cor = index2coordinate(action, board.shape[0])
            # Horizontal Check
            count = 1
            for i in range(1, 5):
                if action_cor[1] + i <= board.shape[0] - 1:
                    if board[action_cor[0]][action_cor[1] + i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if action_cor[1] - i >= 0:
                    if board[action_cor[0]][action_cor[1] - i] == color:
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
                if action_cor[0] + i <= board.shape[0] - 1:
                    if board[action_cor[0] + i][action_cor[1]] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if action_cor[0] - i >= 0:
                    if board[action_cor[0] - i][action_cor[1]] == color:
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
                if (action_cor[0] + i <= board.shape[0] - 1) and (action_cor[1] + i <= board.shape[0] - 1):
                    if board[action_cor[0] + i][action_cor[1] + i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if (action_cor[0] - i >= 0) and (action_cor[1] - i >= 0):
                    if board[action_cor[0] - i][action_cor[1] - i] == color:
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
                if (action_cor[0] + i <= board.shape[0] - 1) and (action_cor[1] - i >= 0):
                    if board[action_cor[0] + i][action_cor[1] - i] == color:
                        count += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                if (action_cor[0] - i >= 0) and (action_cor[1] + i <= board.shape[0] - 1):
                    if board[action_cor[0] - i][action_cor[1] + i] == color:
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
