from .agent import Agent
from ..network.network import *
from .mcts import *
from ..utils import *

MIN = -99999999
MAX = 99999999

score_5 = 5
score_4_live = 4.5
score_4_and_3_live = 4.3
score_4 = 4
score_double_3_live = 3.8
score_3_live = 3.5
score_3 = 3
score_double_2_live = 3
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


class FastAgent(AI):
    def __init__(self, color, depth=1):  # depth must be even
        AI.__init__(self, color)
        self._action_list = []
        self._score_list = []
        self._depth = depth
        self._cut_count = 0
        self._last_move_list = []
        self._atk_def_ratio = 0.1

    def play(self, obs, action, stone_num, *args):
        self._action_list = []
        self._score_list = []
        if action is not None:
            self._last_move_list.append(action)

        size = obs.shape[0]
        if sum(sum(abs(obs))) == 0:  # 若AI执黑，第一步一定下在棋盘中央位置
            pi = [0 for _ in range(size * size)]
            pi[int((size * size) / 2)] = 1
            self._last_move_list.append((7, 7))
            return (7, 7), pi, None, None

        pos_list = self.generate(obs, all=True)
        print('position generated: ', pos_list)
        alpha, beta = MIN, MAX
        score_dict = dict()
        thread_list = []

        for i, j in pos_list:
            new_obs = obs.copy()
            new_obs[i][j] = self.color
            target = self._get_thread_target(obs=new_obs, last_move=(i, j), alpha=alpha, beta=beta,
                                             depth=self._depth - 1, score_dict=score_dict)
            thr = threading.Thread(target=target, name='thread ' + str((i, j)))
            thread_list.append(thr)
            thr.start()

        for thr in thread_list:
            thr.join()

        best_action_list = get_best_action_list(score_dict)
        print('best action list:', best_action_list, ' score = ', score_dict[best_action_list[0]])

        ind = np.random.choice([i for i in range(len(best_action_list))])
        action = best_action_list[ind]

        pi = [0 for _ in range(size * size)]
        pi[coordinate2index(action, size)] = 1

        self._last_move_list.append(action)
        return action, pi, best_action_list, score_dict

    def _get_thread_target(self, obs, last_move, alpha, beta, depth, score_dict):
        def _min():
            _beta = beta
            self._last_move_list.append(last_move)
            if depth == 0:
                score_atk, score_def = self.evaluate(obs)
                self._last_move_list.pop()
                # 对于只搜一层的情况下，必须要教会AI防守活三和冲四。这里的做法是手动提高对方活三和冲四的分数
                if score_def < score_3_live:
                    if score_atk > score_def:
                        score = score_atk - self._atk_def_ratio * score_def
                    else:
                        score = -score_def + self._atk_def_ratio * score_atk
                else:
                    if score_def == score_3_live:
                        if score_atk >= score_4:
                            score = score_atk - self._atk_def_ratio * score_def
                        else:
                            score = -score_4
                    else:
                        # 为了防止AI在对方有活四的情况下放弃治疗
                        if score_def >= score_4_live:
                            score = score_5 if score_atk == score_5 else -score_5
                        else:
                            score = score_5 if score_atk == score_5 else -score_4_live
                x, y = int(last_move[0]), int(last_move[1])
                score_dict[(x, y)] = score
                print((x, y), 'atk=', score_atk, 'def=', score_def, 'total=', score)
                return score

            pos_list = self.generate(obs)
            for i, j in pos_list:
                obs[i][j] = -self.color
                value = self._max(obs, (i, j), alpha, _beta, depth - 1)
                if value < _beta:
                    _beta = value
                obs[i][j] = 0
                if alpha > _beta:
                    break
                    # this indicates that the parent node (belongs to max layer) will select a node with value
                    # no less than alpha, however, the value of child selected in this node (belongs to min layer)
                    # will no more than beta <= alpha, so there is no need to search this node

            self._last_move_list.pop()
            x, y = int(last_move[0]), int(last_move[1])
            score_dict[(x, y)] = _beta
            self._action_list.append((x, y))

        return _min

    # if an obs is in max layer, then the agent is supposed to select the action with max score
    # alpha represents the lower bound of the value of this node
    def _max(self, obs, last_move, alpha, beta, depth):
        self._last_move_list.append(last_move)
        if depth == 0:
            score_atk, score_def = self.evaluate(obs)
            self._last_move_list.pop()
            score = score_atk if score_atk > score_def else -score_def
            return score

        pos_list = self.generate(obs)

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
            score_atk, score_def = self.evaluate(obs)
            self._last_move_list.pop()
            score = score_atk if score_atk > score_def else -score_def
            return score

        pos_list = self.generate(obs)

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

        score_atk, score_def = 0, 0
        for x, y in pos_set:
            c = obs[x][y]
            pt_score = self.evaluate_point(obs, (x, y))
            if c != self.color:
                score_def = max(score_def, pt_score)
            else:
                score_atk = max(score_atk, pt_score)

        return score_atk, score_def

    def evaluate_point(self, obs, pos):
        i, j = pos[0], pos[1]
        color = obs[i][j]
        dir_set = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_count = 0
        max_consecutive_count = 0
        max_score = 0

        for dir in dir_set:
            score = 0
            count_1, count_2 = 1, 1
            consecutive_count_1, consecutive_count_2 = 1, 1
            space_1, space_2 = 0, 0
            block_1, block_2 = 0, 0
            consecutive_flag = True

            for k in range(1, 5):
                if i + k * dir[0] in range(0, 15) and j + k * dir[1] in range(0, 15):
                    if obs[i + k * dir[0]][j + k * dir[1]] == color:
                        if space_1 == 2:
                            break
                        count_1 += 1
                        if consecutive_flag:
                            consecutive_count_1 += 1
                    if obs[i + k * dir[0]][j + k * dir[1]] == -color:
                        block_1 = 1
                        break
                    if obs[i + k * dir[0]][j + k * dir[1]] == 0:
                        space_1 += 1
                        consecutive_flag = False
                        if space_1 == 3:
                            break
                else:
                    block_1 = 1
                    break

            consecutive_flag = True

            for k in range(1, 5):
                if i - k * dir[0] in range(0, 15) and j - k * dir[1] in range(0, 15):
                    if obs[i - k * dir[0]][j - k * dir[1]] == color:
                        if space_2 == 2:
                            break
                        count_2 += 1
                        if consecutive_flag:
                            consecutive_count_2 += 1
                    if obs[i - k * dir[0]][j - k * dir[1]] == -color:
                        block_2 = 1
                        break
                    if obs[i - k * dir[0]][j - k * dir[1]] == 0:
                        space_2 += 1
                        consecutive_flag = False
                        if space_2 == 3:
                            break
                else:
                    block_2 = 1
                    break

            # there are several cases:
            # 1. ooox: block=1, space=0, count=consecutive_count
            # 2. ooo__: block=0, space=2, count=consecutive_count
            # 3. ooo_x: block=1, space=1, count=consecutive_count
            # 4. oo_ox: block=1, space=1, count>consecutive_count

            count = max(count_1 + consecutive_count_2, count_2 + consecutive_count_1) - 1

            consecutive_count = consecutive_count_1 + consecutive_count_2 - 1

            if consecutive_count >= 5:
                return score_5

            if count == 4:
                if consecutive_count == 4:  # ??oooo??
                    if space_1 >= 1 and space_2 >= 1:  # ?_oooo_?
                        score = score_4_live
                    else:
                        if space_1 == 0 and space_2 == 0:  # xoooox
                            pass
                        else:  # xoooo_
                            score = score_4
                else:
                    if consecutive_count == 3:  # ??ooo_o??
                        score = score_4
                    else:  # (consecutive_count == 2) ??oo_oo??
                        score = score_4

            if count == 3:
                if consecutive_count == 3:  # ??ooo??
                    if space_1 >= 1 and space_2 >= 1:  # ?_ooo_?
                        score = score_3_live
                    else:
                        if space_1 == 0 and space_2 == 0:  # xooox
                            pass
                        else:  # xooo_
                            score = score_3
                else:  # (consecutive_count == 2) ??oo_o??
                    if consecutive_count_1 == 2:
                        if space_1 >= 1 and space_2 >= 2:  # ?_oo_o_?
                            score = score_3_live
                        else:
                            if space_1 == 0 and space_2 == 1:  # xoo_ox
                                pass
                            else:
                                score = score_3
                    else:  # (consecutive_count_2 == 2)
                        if space_2 >= 1 and space_1 >= 2:  # ?_o_oo_?
                            score = score_3_live
                        else:
                            if space_1 == 1 and space_2 == 0:  # xo_oox
                                pass
                            else:
                                score = score_3

            if count == 2:
                if consecutive_count == 2:  # ??oo??
                    if space_1 <= 1 and space_2 <= 1:  # x?oo?x
                        pass
                    else:
                        if space_1 == 0 or space_2 == 0:  # xoo__?
                            if space_1 == 3 or space_2 == 3:  # xoo___
                                score = score_2
                            else:
                                pass
                        else:  # ?__oo_??
                            score = score_2_live

                else:  # ??o_o??
                    if space_1 + space_2 < 3:
                        pass
                    else:
                        if count_1 == 2:
                            if space_2 == 0:  # (space_1 == 3) __o_ox
                                score = score_2
                            else:
                                score = score_2_live
                        else:  # (count_2 == 2)
                            if space_1 == 0:  # (space_2 == 3) xo_o__
                                score = score_2
                            else:
                                score = score_2_live

            # bonus
            if max_score == score_2_live and score == score_2_live:
                score = score_double_2_live
            if max_score == score_3_live and score == score_3_live:
                score = score_double_3_live
            if max_score == score_4 and score == score_3_live:
                score = score_4_and_3_live
            if max_score == score_3_live and score == score_4:
                score = score_4_and_3_live

            if count > max_count:
                max_count = count
            if consecutive_count > max_consecutive_count:
                max_consecutive_count = consecutive_count

            if score > max_score:
                max_score = score

        return max_score

    def generate(self, obs, all=False):
        good_pts = []
        good_scores = []
        pts = []
        scores = []
        dir_set = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

        if all:
            indices = np.where(obs)
            check_list = [(indices[0][i], indices[1][i]) for i in range(len(indices[0]))]
        else:
            if len(self._last_move_list) > 7:
                check_list = self._last_move_list[-7:]
            else:
                check_list = self._last_move_list

        for x0, y0 in check_list:
            for dir in dir_set:
                if x0 + dir[0] in range(0, 15) and y0 + dir[1] in range(0, 15):
                    pos = (x0 + dir[0], y0 + dir[1])
                    if obs[pos[0]][pos[1]] == 0 and pos not in pts:
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
                        pts.append(pos)
                        scores.append(score)
                        obs[pos[0]][pos[1]] = 0

        if len(good_pts) > 0 and max(good_scores) >= score_4:
            print('good')
            pts = good_pts
            scores = good_scores
        lst = np.array([pts, scores])
        pts = lst[:, lst[1].argsort()][0]
        pos_list = list(pts)

        pos_list.reverse()
        return pos_list


def get_best_action_list(score_dict):
    best_action_list = []
    max_score = MIN
    for key in score_dict:
        if max_score < score_dict[key]:
            best_action_list = [key]
            max_score = score_dict[key]
        elif max_score == score_dict[key]:
            best_action_list.append(key)
    return best_action_list


def print_score_dict(score_dict):
    for key in score_dict:
        print(str(key) + ': ' + str(score_dict[key]))
