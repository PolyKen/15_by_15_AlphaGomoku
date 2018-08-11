from .agent import Agent
from ..network.network import *
from .mcts import *
from ..decorator import *
import asyncio


MIN = -99999999
MAX = 99999999

score_5 = 100000
score_4_live = 10000
score_4 = 1000
score_3_live = 1000
score_3 = 100
score_2_live = 50
score_2 = 10


class AI(Agent):
    def __init__(self, color):
        Agent.__init__(self, color)

    def play(self, *args, **kwargs):
        pass


class MCTSAgent(AI):
    def __init__(self, conf, color, use_stochastic_policy):
        AI.__init__(self, color)
        network = Network(conf)
        self._mcts = MCTS(conf, network, color, use_stochastic_policy)
        self._network = network
        self._board_size = conf['board_size']

    def play(self, obs, action, stone_num):
        act_ind, pi, prior_prob, value = self._mcts.action(obs, action, stone_num)
        act_cor = index2coordinate(act_ind, self._board_size)
        return act_cor, pi, prior_prob, value

    def set_self_play(self, is_self_play):
        self._mcts.set_self_play(is_self_play)

    def set_stochastic_policy(self, use_stochastic_policy):
        self._mcts.set_stochastic_policy(use_stochastic_policy)

    def reset_mcts(self):
        self._mcts.reset()

    @log
    def train(self, obs, color, last_move, pi, z):
        loss = self._network.train(obs, color, last_move, pi, z)
        return loss
        
    def save_model(self):
        self._network.save_model()
        print('> model saved')

    def load_model(self):
        self._network.load_model()


class NaiveAgent(AI):
    def __init__(self, color, depth=2):
        AI.__init__(self, color)
        self._loop = asyncio.get_event_loop()
        self._action_list = []
        self._score_list = []
        self._depth = depth
        self._cut_count = 0
        self._last_move_list = []

    def play(self, obs, last_move, *args):
        self._action_list = []
        self._score_list = []
        if last_move is not None:
            self._last_move_list.append(last_move)

        size = obs.shape[0]
        if sum(sum(abs(obs))) == 0:
            pi = [0 for _ in range(size*size)]
            pi[int((size*size)/2)] = 1
            self._last_move_list.append((7, 7))
            return (7, 7), pi, None, None

        pos_list = self._generate(obs)
        alpha, beta = MIN, MAX
        action = pos_list[0]
        for i, j in pos_list:
            new_obs = obs.copy()
            new_obs[i][j] = self.color
            value = self._min(new_obs, (i, j), alpha, beta, self._depth)
            self._action_list.append((int(i), int(j)))
            self._score_list.append(value)
            if value > alpha:
                alpha = value
                action = (int(i), int(j))
            # print(str((i, j)) + ': ' + str(score))

        s_min = min(self._score_list)
        s_max = max(self._score_list)

        try:
            pi = [((score - s_min)/(s_max - s_min))**10 for score in self._score_list]
            m = sum(pi)
            pi = [score/m for score in pi]
        except ZeroDivisionError:
            m = sum(self._score_list)
            pi = [score/m for score in self._score_list]

        ind = np.random.choice([i for i in range(len(pi))], p=pi)
        # action = self._action_list[ind]

        pi = [0 for _ in range(size*size)]
        pi[coordinate2index(action, size)] = 1

        self._last_move_list.append(action)
        return action, pi, None, None

    # if an obs is in max layer, then the agent is supposed to select the action with max score
    # alpha represents the lower bound of the value of this node
    def _max(self, obs, last_move, alpha, beta, depth):
        if alpha >= beta:
            return alpha
        if depth == 0:
            return self.evaluate(obs)

        self._last_move_list.append(last_move)
        pos_list = self._generate(obs)

        for i, j in pos_list:
            obs[i][j] = self.color
            value = self._min(obs, (i, j), alpha, beta, depth - 1)
            if value > alpha:
                alpha = value
            obs[i][j] = 0

        self._last_move_list.pop()
        return alpha

    # if an obs is in min layer, then the agent is supposed to select the action with min scores
    # beta represents the upper bound of the value of this node
    def _min(self, obs, last_move, alpha, beta, depth):
        if alpha >= beta:
            return beta
            # this indicates that the parent node (belongs to max layer) will select a node with value
            # no less than alpha, however, the value of child selected in this node (belongs to min layer)
            # will no more than beta <= alpha, so there is no need to search this node
        if depth == 0:
            return self.evaluate(obs)

        self._last_move_list.append(last_move)
        pos_list = self._generate(obs)

        for i, j in pos_list:
            obs[i][j] = -self.color
            value = self._max(obs, (i, j), alpha, beta, depth - 1)
            if value < beta:
                beta = value
            obs[i][j] = 0

        self._last_move_list.pop()
        return beta

    # the obs is better for this agent if the score is larger
    def evaluate(self, obs):
        pos_ind = np.where(obs)
        pos_set = [(pos_ind[0][i], pos_ind[1][i]) for i in range(len(pos_ind[0]))]

        score = 0

        for x, y in pos_set:
            c = obs[x][y]
            each = self.evaluate_point(obs, (x, y))
            score += (self.color * c) * each

        return score

    def _check_consecutive(self, obs, pos, direction):
        i, j = pos[0], pos[1]
        color = obs[i][j]

        count = 0
        for k in range(5):
            if i + k*direction[0] in range(0, 15) and j + k*direction[1] in range(0, 15):
                c = obs[i+k*direction[0]][j+k*direction[1]]
                if c == color:
                    count += 1
                elif c == -color:
                    break
        return count, color

    def evaluate_point(self, obs, pos):
        i, j = pos[0], pos[1]
        color = obs[i][j]
        dir_set = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_count = 0
        max_score = 0
        for dir in dir_set:
            score = 0
            count_1, count_2 = 0, 0
            space_1, space_2 = 0, 0
            block_1, block_2 = 0, 0
            for k in range(1, 5):
                if i + k*dir[0] in range(0, 15) and j + k*dir[1] in range(0, 15):
                    if obs[i+k*dir[0]][j+k*dir[1]] == color:
                        count_1 += 1
                    if obs[i+k*dir[0]][j+k*dir[1]] == -color:
                        block_1 = 1
                        break
                    if obs[i+k*dir[0]][j+k*dir[1]] == 0:
                        space_1 += 1
                        if space_1 == 2:
                            break
            for k in range(1, 5):
                if i - k*dir[0] in range(0, 15) and j - k*dir[1] in range(0, 15):
                    if obs[i-k*dir[0]][j-k*dir[1]] == color:
                        count_2 += 1
                    if obs[i-k*dir[0]][j-k*dir[1]] == -color:
                        block_2 = 1
                        break
                    if obs[i-k*dir[0]][j-k*dir[1]] == 0:
                        space_2 += 1
                        if space_2 == 2:
                            break
            count = 1 + count_1 + count_2

            if count < max_count:
                continue

            if count == 5:
                return score_5
            if count == 4:
                if block_1 == 0 and block_2 == 0 and space_1 == 2 and space_2 == 2:
                    score = score_4_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_4
            if count == 3:
                if block_1 == 0 and block_2 == 0 and space_1 == 2 and space_2 == 2:
                    score = score_3_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_3
            if count == 2:
                if block_1 == 0 and block_2 == 0 and space_1 == 2 and space_2 == 2:
                    score = score_2_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_2

            if score >= max_score:
                if max_score == score_4:
                    max_score = score_4_live
                else:
                    max_score = score

        return max_score

    def _generate(self, obs):
        good_pts = []
        good_scores = []
        near = []
        scores = []
        dir_set = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

        if len(self._last_move_list) > 5:
            last_move_list = self._last_move_list[-5:]
        else:
            last_move_list = self._last_move_list

        for x0, y0 in last_move_list:
            for dir in dir_set:
                if x0 + dir[0] in range(0, 15) and y0 + dir[1] in range(0, 15):
                    pos = (x0 + dir[0], y0 + dir[1])
                    if obs[pos[0]][pos[1]] == 0 and pos not in good_pts and pos not in near:
                        obs[pos[0]][pos[1]] = 1
                        score_atk = self.evaluate_point(obs, pos)
                        obs[pos[0]][pos[1]] = -1
                        score_def = self.evaluate_point(obs, pos)
                        score = max(score_atk, score_def)
                        if score >= score_4:
                            good_pts.append(pos)
                            good_scores.append(score)
                            if score == score_5:
                                good_pts.reverse()
                                return good_pts
                        else:
                            near.append(pos)
                            scores.append(score)
                        obs[pos[0]][pos[1]] = 0

        if len(good_pts) > 0:
            lst = np.array([good_pts, good_scores])
            good_pts = lst[:, lst[1].argsort()][0]
            pos_list = list(good_pts)
        else:
            lst = np.array([near, scores])
            near = lst[:, lst[1].argsort()][0]
            pos_list = list(near)
        pos_list.reverse()
        return pos_list
