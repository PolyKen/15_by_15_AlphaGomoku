from .agent import Agent
from ..network.network import *
from .mcts import *
from ..utils import *
import asyncio

MIN = -99999999
MAX = 99999999

score_5 = 5
score_4_live = 4.5
score_4_and_3_live = 4.3
score_4 = 4
score_double_3_live = 3.8
score_3_live = 3.5
score_3 = 3
score_2_live = 2.5
score_2 = 2


class AI(Agent):
    def __init__(self, color):
        Agent.__init__(self, color)

    def play(self, *args, **kwargs):
        pass


class MCTSAgent(AI):
    def __init__(self, conf, color, use_stochastic_policy):
        AI.__init__(self, color)
        conf.update(net_para_file='AlphaRenju_Zero/network/model/model_b_' + str(conf['board_size']) + '.h5')
        black_net = Network(conf)
        conf.update(net_para_file='AlphaRenju_Zero/network/model/model_w_' + str(conf['board_size']) + '.h5')
        white_net = Network(conf)
        self._mcts = MCTS(conf, black_net, white_net, color, use_stochastic_policy)
        self._black_net = black_net
        self._white_net = white_net
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
        obs_b, obs_w = obs[0::2], obs[1::2]
        color_b, color_w = color[0::2], color[1::2]
        last_move_b, last_move_w = last_move[0::2], last_move[1::2]
        pi_b, pi_w = pi[0::2], pi[1::2]
        z_b, z_w = z[0::2], z[1::2]

        loss_b = self._black_net.train(obs_b, color_b, last_move_b, pi_b, z_b)
        loss_w = self._white_net.train(obs_w, color_w, last_move_w, pi_w, z_w)
        return loss_b, loss_w

    def save_model(self):
        self._black_net.save_model()
        self._white_net.save_model()
        print('> model saved')

    def load_model(self):
        self._black_net.load_model()
        self._white_net.load_model()


class NaiveAgent(AI):
    def __init__(self, color, depth=2):  # depth must be even
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
            pi = [0 for _ in range(size * size)]
            pi[int((size * size) / 2)] = 1
            self._last_move_list.append((7, 7))
            return (7, 7), pi, None, None

        pos_list = self._generate(obs, all=True)
        alpha, beta = MIN, MAX
        best_action_list = []
        tasks = []
        # score_dict = dict()
        for i, j in pos_list:
            new_obs = obs.copy()
            new_obs[i][j] = self.color
            value = self._min(new_obs, (i, j), alpha, beta, self._depth - 1)
            self._action_list.append((int(i), int(j)))
            # tasks.append(self._min_coroutine(new_obs, (i, j), alpha, beta, self._depth - 1, score_dict))
            self._score_list.append(value)
            if value > alpha:
                alpha = value
                best_action_list = [(int(i), int(j))]
            elif value == alpha:
                best_action_list.append((int(i), int(j)))

        # self._loop.run_until_complete(asyncio.wait(tasks))

        # best_action_list = get_best_action_list(score_dict)

        ind = np.random.choice([i for i in range(len(best_action_list))])
        action = best_action_list[ind]

        pi = [0 for _ in range(size * size)]
        pi[coordinate2index(action, size)] = 1

        self._last_move_list.append(action)
        return action, pi, None, None

    async def _min_coroutine(self, obs, last_move, alpha, beta, depth, score_dict):
        value = self._min(obs, last_move, alpha, beta, depth)
        score_dict[(int(last_move[0]), int(last_move[1]))] = value

    # if an obs is in max layer, then the agent is supposed to select the action with max score
    # alpha represents the lower bound of the value of this node
    def _max(self, obs, last_move, alpha, beta, depth):
        self._last_move_list.append(last_move)
        if depth == 0:
            score = self.evaluate(obs)
            self._last_move_list.pop()
            return score

        pos_list = self._generate(obs)

        for i, j in pos_list:
            obs[i][j] = self.color
            value = self._min(obs, (i, j), alpha, beta, depth - 1)
            if value > alpha:
                alpha = value
            obs[i][j] = 0
            if alpha > beta:
                break

        self._last_move_list.pop()
        return alpha

    # if an obs is in min layer, then the agent is supposed to select the action with min scores
    # beta represents the upper bound of the value of this node
    def _min(self, obs, last_move, alpha, beta, depth):
        self._last_move_list.append(last_move)
        if depth == 0:
            score = self.evaluate(obs)
            self._last_move_list.pop()
            return score

        pos_list = self._generate(obs)

        for i, j in pos_list:
            obs[i][j] = -self.color
            value = self._max(obs, (i, j), alpha, beta, depth - 1)
            # print((i, j), value)
            if value < beta:
                beta = value
            obs[i][j] = 0
            if alpha > beta:
                break
                # this indicates that the parent node (belongs to max layer) will select a node with value
                # no less than alpha, however, the value of child selected in this node (belongs to min layer)
                # will no more than beta <= alpha, so there is no need to search this node

        self._last_move_list.pop()
        return beta

    def evaluate(self, obs):
        pos_ind = np.where(obs)
        pos_set = [(pos_ind[0][i], pos_ind[1][i]) for i in range(len(pos_ind[0]))]

        score = 0
        for x, y in pos_set:
            c = obs[x][y]
            pt_score = self.evaluate_point(obs, (x, y))
            if c != self.color:
                pt_score += 0.1
                if abs(score) < pt_score:
                    score = -pt_score
            else:
                if abs(score) < pt_score:
                    score = pt_score

        return score

        # score_list = []
        # tasks = []
        # for x, y in pos_set:
        #     c = obs[x][y]
        #     tasks.append(self._evaluate_point_coroutine(obs, (x, y), c == self.color, score_list))
        #
        # self._loop.run_until_complete(asyncio.wait(tasks))
        # max_score = max(score_list)
        # min_score = min(score_list)
        # if max_score > abs(min_score):
        #     return max_score
        # else:
        #     return min_score

    async def _evaluate_point_coroutine(self, obs, pos, same_color, score_list):
        pt_score = self.evaluate_point(obs, pos)
        if not same_color:
            pt_score *= -1
        score_list.append(pt_score)

    def evaluate_point(self, obs, pos):
        i, j = pos[0], pos[1]
        color = obs[i][j]
        dir_set = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_count = 0
        max_score = 0
        for dir in dir_set:
            score = 0
            count = 1
            consecutive_count = 1
            space_1, space_2 = 0, 0
            block_1, block_2 = 0, 0
            consecutive_flag = True
            for k in range(1, 5):
                if i + k * dir[0] in range(0, 15) and j + k * dir[1] in range(0, 15):
                    if obs[i + k * dir[0]][j + k * dir[1]] == color:
                        count += 1
                        if consecutive_flag:
                            consecutive_count += 1
                    if obs[i + k * dir[0]][j + k * dir[1]] == -color:
                        if space_1 == 0:
                            block_1 = 1
                        break
                    if obs[i + k * dir[0]][j + k * dir[1]] == 0:
                        space_1 += 1
                        consecutive_flag = False
                        if space_1 == 2:
                            break
            consecutive_flag = True
            for k in range(1, 5):
                if i - k * dir[0] in range(0, 15) and j - k * dir[1] in range(0, 15):
                    if obs[i - k * dir[0]][j - k * dir[1]] == color:
                        count += 1
                        if consecutive_flag:
                            consecutive_count += 1
                    if obs[i - k * dir[0]][j - k * dir[1]] == -color:
                        if space_2 == 0:
                            block_2 = 1
                        break
                    if obs[i - k * dir[0]][j - k * dir[1]] == 0:
                        space_2 += 1
                        consecutive_flag = False
                        if space_2 == 2:
                            break

            if count < max_count:
                continue
            else:
                max_count = count

            if count == 5:
                return score_5
            if count == 4:
                if block_1 == 0 and block_2 == 0 and consecutive_count == count:
                    score = score_4_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_4
            if count == 3:
                if block_1 == 0 and block_2 == 0:
                    score = score_3_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_3
            if count == 2:
                if block_1 == 0 and block_2 == 0:
                    score = score_2_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_2

            if score >= max_score:
                if max_score == score_4:
                    max_score = score_4_live  # double live 3 or live 3 + 4 or double 4
                else:
                    max_score = score

        return max_score

    def _generate(self, obs, all=False):
        good_pts = []
        good_scores = []
        near = []
        scores = []
        dir_set = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

        if all:
            indices = np.where(obs)
            check_list = [(indices[0][i], indices[1][i]) for i in range(len(indices[0]))]
        else:
            if len(self._last_move_list) > 5:
                check_list = self._last_move_list[-5:]
            else:
                check_list = self._last_move_list

        for x0, y0 in check_list:
            for dir in dir_set:
                if x0 + dir[0] in range(0, 15) and y0 + dir[1] in range(0, 15):
                    pos = (x0 + dir[0], y0 + dir[1])
                    if obs[pos[0]][pos[1]] == 0 and pos not in good_pts and pos not in near:
                        obs[pos[0]][pos[1]] = self.color
                        score_atk = self.evaluate_point(obs, pos)
                        obs[pos[0]][pos[1]] = -self.color
                        score_def = self.evaluate_point(obs, pos)
                        score = max(score_atk, score_def)
                        if score >= score_3_live:
                            good_pts.append(pos)
                            good_scores.append(score)
                            if score_atk == score_5:
                                break
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


class FastAgent(AI):
    def __init__(self, color, depth=1):  # depth must be even
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
        if sum(sum(abs(obs))) == 0:                     # 若AI执黑，第一步一定下在棋盘中央位置
            pi = [0 for _ in range(size * size)]
            pi[int((size * size) / 2)] = 1
            self._last_move_list.append((7, 7))
            return (7, 7), pi, None, None

        pos_list = self._generate(obs, all=True)        # 第一层生成的搜索点最多
        alpha, beta = MIN, MAX
        tasks = []
        score_dict = dict()
        for i, j in pos_list:
            new_obs = obs.copy()
            new_obs[i][j] = self.color
            tasks.append(self._min_coroutine(new_obs, (i, j), alpha, beta, self._depth - 1, score_dict))

        self._loop.run_until_complete(asyncio.wait(tasks))

        best_action_list = get_best_action_list(score_dict)
        print_score_dict(score_dict)

        ind = np.random.choice([i for i in range(len(best_action_list))])
        action = best_action_list[ind]

        pi = [0 for _ in range(size * size)]
        pi[coordinate2index(action, size)] = 1

        self._last_move_list.append(action)
        return action, pi, None, None

    async def _min_coroutine(self, obs, last_move, alpha, beta, depth, score_dict):
        value = self._min(obs, last_move, alpha, beta, depth)
        score_dict[(int(last_move[0]), int(last_move[1]))] = value

    # if an obs is in max layer, then the agent is supposed to select the action with max score
    # alpha represents the lower bound of the value of this node
    def _max(self, obs, last_move, alpha, beta, depth):
        self._last_move_list.append(last_move)
        if depth == 0:
            score = self.evaluate(obs)
            self._last_move_list.pop()
            return score

        pos_list = self._generate(obs)

        for i, j in pos_list:
            obs[i][j] = self.color
            value = self._min(obs, (i, j), alpha, beta, depth - 1)
            if value > alpha:
                alpha = value
            obs[i][j] = 0
            if alpha > beta:
                break

        self._last_move_list.pop()
        return alpha

    # if an obs is in min layer, then the agent is supposed to select the action with min scores
    # beta represents the upper bound of the value of this node
    def _min(self, obs, last_move, alpha, beta, depth):
        self._last_move_list.append(last_move)
        if depth == 0:
            score = self.evaluate(obs)
            self._last_move_list.pop()
            return score

        pos_list = self._generate(obs)

        for i, j in pos_list:
            obs[i][j] = -self.color
            value = self._max(obs, (i, j), alpha, beta, depth - 1)
            # print((i, j), value)
            if value < beta:
                beta = value
            obs[i][j] = 0
            if alpha > beta:
                break
                # this indicates that the parent node (belongs to max layer) will select a node with value
                # no less than alpha, however, the value of child selected in this node (belongs to min layer)
                # will no more than beta <= alpha, so there is no need to search this node

        self._last_move_list.pop()
        return beta

    def evaluate(self, obs):
        pos_ind = np.where(obs)
        pos_set = [(pos_ind[0][i], pos_ind[1][i]) for i in range(len(pos_ind[0]))]

        score = 0
        for x, y in pos_set:
            c = obs[x][y]
            pt_score = self.evaluate_point(obs, (x, y))
            if c != self.color:
                pt_score += 0.1
                if abs(score) < pt_score:
                    score = -pt_score
            else:
                if abs(score) < pt_score:
                    score = pt_score

        return score

    async def _evaluate_point_coroutine(self, obs, pos, same_color, score_list):
        pt_score = self.evaluate_point(obs, pos)
        if not same_color:
            pt_score *= -1
        score_list.append(pt_score)

    def evaluate_point(self, obs, pos):
        i, j = pos[0], pos[1]
        color = obs[i][j]
        dir_set = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_count = 0
        max_score = 0

        for dir in dir_set:
            score = 0
            count = 1
            consecutive_count = 1
            space_1, space_2 = 0, 0
            block_1, block_2 = 0, 0
            consecutive_flag = True

            for k in range(1, 5):
                if i + k * dir[0] in range(0, 15) and j + k * dir[1] in range(0, 15):
                    if obs[i + k * dir[0]][j + k * dir[1]] == color:
                        count += 1
                        if consecutive_flag:
                            consecutive_count += 1
                    if obs[i + k * dir[0]][j + k * dir[1]] == -color:
                        if space_1 == 0:
                            block_1 = 1
                        break
                    if obs[i + k * dir[0]][j + k * dir[1]] == 0:
                        space_1 += 1
                        consecutive_flag = False
                        if space_1 == 2:
                            break

            consecutive_flag = True

            for k in range(1, 5):
                if i - k * dir[0] in range(0, 15) and j - k * dir[1] in range(0, 15):
                    if obs[i - k * dir[0]][j - k * dir[1]] == color:
                        count += 1
                        if consecutive_flag:
                            consecutive_count += 1
                    if obs[i - k * dir[0]][j - k * dir[1]] == -color:
                        if space_2 == 0:
                            block_2 = 1
                        break
                    if obs[i - k * dir[0]][j - k * dir[1]] == 0:
                        space_2 += 1
                        consecutive_flag = False
                        if space_2 == 2:
                            break

            if count > max_count:
                max_count = count

            if count == 5:
                return score_5
            if count == 4:
                if block_1 == 0 and block_2 == 0 and consecutive_count == count:
                    score = score_4_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_4
            if count == 3:
                if block_1 == 0 and block_2 == 0:
                    score = score_3_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_3
            if count == 2:
                if block_1 == 0 and block_2 == 0:
                    score = score_2_live
                elif block_1 == 0 or block_2 == 0:
                    score = score_2

            if max_score == score_3_live and score == score_3_live:
                score = score_double_3_live
                # print('double 3!')
            if max_score == score_4 and score == score_3_live:
                score = score_4_and_3_live
                # print('4 + live3!')
            if max_score == score_3_live and score == score_4:
                score = score_4_and_3_live
                # print('4 + live3!')

            if score > max_score:
                max_score = score

        return max_score

    def _generate(self, obs, all=False):
        good_pts = []
        good_scores = []
        near = []
        scores = []
        dir_set = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

        if all:
            indices = np.where(obs)
            check_list = [(indices[0][i], indices[1][i]) for i in range(len(indices[0]))]
        else:
            if len(self._last_move_list) > 5:
                check_list = self._last_move_list[-5:]
            else:
                check_list = self._last_move_list

        for x0, y0 in check_list:
            for dir in dir_set:
                if x0 + dir[0] in range(0, 15) and y0 + dir[1] in range(0, 15):
                    pos = (x0 + dir[0], y0 + dir[1])
                    if obs[pos[0]][pos[1]] == 0 and pos not in good_pts and pos not in near:
                        obs[pos[0]][pos[1]] = self.color
                        score_atk = self.evaluate_point(obs, pos)
                        obs[pos[0]][pos[1]] = -self.color
                        score_def = self.evaluate_point(obs, pos)
                        score = max(score_atk, score_def)
                        if score >= score_3_live:
                            good_pts.append(pos)
                            good_scores.append(score)
                            if score_atk == score_5:
                                break
                        else:
                            near.append(pos)
                            scores.append(score)
                        obs[pos[0]][pos[1]] = 0

        if len(good_pts) > 0:
            lst = np.array([good_pts, good_scores])
            good_pts = lst[:, lst[1].argsort()][0]
            pos_list = list(good_pts)
            if all:
                print('best score:', max(good_scores), 'best pos generated:', pos_list[-1])
        else:
            lst = np.array([near, scores])
            near = lst[:, lst[1].argsort()][0]
            pos_list = list(near)
            if all:
                print('best score:', max(scores), 'best pos generated:', pos_list[-1])

        pos_list.reverse()
        return pos_list


def get_best_action_list(score_dict):
    best_action_list = []
    max_score = MIN
    for key in score_dict:
        if max_score < score_dict[key]:
            best_action_list = [key]
            max_score = score_dict[key]
        if max_score == score_dict[key]:
            best_action_list.append(key)
    return best_action_list


def print_score_dict(score_dict):
    for key in score_dict:
        print(str(key) + ': ' + str(score_dict[key]))
