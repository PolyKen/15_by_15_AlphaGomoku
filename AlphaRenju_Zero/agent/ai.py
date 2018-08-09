from .agent import Agent
from ..network.network import *
from .mcts import *
from ..decorator import *
import asyncio


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
    def __init__(self, color):
        AI.__init__(self, color)
        self._alpha = 0.8
        self._loop = asyncio.get_event_loop()
        self._action_list = []
        self._score_list = []

    def play(self, obs, *args):
        self._action_list = []
        self._score_list = []
        ind = np.where(obs)
        x_min, x_max = min(ind[0]), max(ind[0])
        y_min, y_max = min(ind[1]), max(ind[1])
        size = obs.shape[0]

        tasks = []

        for i in range(size):
            if i < x_min - 1 or i > x_max + 1:
                continue
            for j in range(size):
                if j < y_min - 1 or j > y_max + 1:
                    continue
                if obs[i][j] != 0:
                    continue    # occupied
                else:
                    new_obs = obs.copy()
                    new_obs[i][j] = self.color
                    tasks.append(self._simulate(new_obs, (i, j)))

        self._loop.run_until_complete(asyncio.wait(tasks))

        m = sum(self._score_list)
        pi = [score / m for score in self._score_list]
        ind = np.random.choice([i for i in range(len(self._action_list))], p=pi)
        action = self._action_list[ind]
        pi = [0 for i in range(size*size)]
        pi[ind] = 1
        print(action)
        return action, pi, None, None

    async def _simulate(self, obs, action):
        max_score = -1000000
        current_score = self._evaluate(obs, self.color)
        ind = np.where(obs)
        x_min, x_max = min(ind[0]), max(ind[0])
        y_min, y_max = min(ind[1]), max(ind[1])
        for i in range(obs.shape[0]):
            if i < x_min - 1 or i > x_max + 1:
                continue
            for j in range(obs.shape[1]):
                if j < y_min - 1 or j > y_max + 1:
                    continue
                if obs[i][j] != 0:
                    continue
                else:
                    new_obs = obs.copy()
                    new_obs[i][j] = -self.color
                    score = 0
                    # score = self._evaluate(new_obs, -self.color)
                    if score > max_score:
                        max_score = score

        score = current_score - self._alpha * max_score
        self._action_list.append(action)
        self._score_list.append(score)

    def _evaluate(self, obs, color):
        score_5 = 10000 * self._convolution_5(obs, color)
        convs = self._convolution_4_1(obs, color)
        score_4_1 = 1000 * convs[0] + 100 * convs[1]
        score_4_2 = 100 * self._convolution_4_2(obs, color)
        score_4_3 = 100 * self._convolution_4_3(obs, color)
        convs = self._convolution_3_1(obs, color)
        score_3_1 = 100 * convs[0] + 50 * convs[1]
        convs = self._convolution_3_2(obs, color)
        score_3_2 = 100 * convs[0] + 50 * convs[1]
        convs = self._convolution_3_3(obs, color)
        score_3_3 = 100 * convs[0] + 50 * convs[1]
        convs = self._convolution_2_1(obs, color)
        score_2_1 = 10 * convs[0] + 5 * convs[1]
        convs = self._convolution_2_2(obs, color)
        score_2_2 = 10 * convs[0] + 5 * convs[1]
        convs = self._convolution_2_3(obs, color)
        score_2_3 = 10 * convs[0] + 5 * convs[1]

        score = score_5
        score += score_4_1 + score_4_2 + score_4_3
        score += score_3_1 + score_3_2 + score_3_3
        score += score_2_1 + score_2_2 + score_2_3

        return score

    def _convolution_5(self, obs, color):
        filter = color * np.array([1] * 5)
        size = len(filter)
        board_size = obs.shape[0]
        count = 0
        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j+size])
                if result == 5:
                    count += 1
                if result == -5:
                    count -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i+size, j])
                if result == 5:
                    count += 1
                if result == -5:
                    count -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i+k][j+k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 5:
                    count += 1
                if result == -5:
                    count -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i+k][j-k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 5:
                    count += 1
                if result == -5:
                    count -= 1

        return score


    def _convolution_4_1(self, obs, color):
        filter = color * np.array([1000, 1, 1, 1, 1, 1000])
        size = len(filter)
        board_size = obs.shape[0]

        live = 0
        dead = 0
        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j+size])
                if result == 4:
                    live += 1
                if result == -996:
                    dead += 1
                if result == -4:
                    live -= 1
                if result == 996:
                    dead -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i+size, j])
                if result == 4:
                    live += 1
                if result == -996:
                    dead += 1
                if result == -4:
                    live -= 1
                if result == 996:
                    dead -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i+k][j+k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 4:
                    live += 1
                if result == -996:
                    dead += 1
                if result == -4:
                    live -= 1
                if result == 996:
                    dead -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i+k][j-k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 4:
                    live += 1
                if result == -996:
                    dead += 1
                if result == -4:
                    live -= 1
                if result == 996:
                    dead -= 1

        return live, dead

    def _convolution_4_2(self, obs, color):
        filter = color * np.array([1, 1, 1000, 1, 1])
        size = len(filter)
        board_size = obs.shape[0]
        count = 0
        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j + size])
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i + size, j])
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i+k][j+k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 4:
                    count += 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i+k][j-k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1

        return count

    def _convolution_4_3(self, obs, color):
        filter_1 = color * np.array([1, 1, 1, 1000, 1])
        filter_2 = color * np.array([1, 1000, 1, 1, 1])
        size = len(filter_1)
        board_size = obs.shape[0]
        count = 0
        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter_1, obs[i, j:j+size])
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1
                result = np.dot(filter_2, obs[i, j:j+size])
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter_1, obs[i:i + size, j])
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1
                result = np.dot(filter_2, obs[i:i + size, j])
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i + k][j + k] for k in range(size)]
                result = np.dot(filter_1, list)
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1
                result = np.dot(filter_2, list)
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i + k][j - k] for k in range(size)]
                result = np.dot(filter_1, list)
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1
                result = np.dot(filter_2, list)
                if result == 4:
                    count += 1
                if result == -4:
                    count -= 1

        return count

    def _convolution_3_1(self, obs, color):
        filter = color * np.array([1000, 1, 1, 1, 1000])
        size = len(filter)
        board_size = obs.shape[0]
        live = 0
        dead = 0
        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j + size])
                if result == 3:
                    live += 1
                if result == -1997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 1997:
                    dead -= 1


        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i + size, j])
                if result == 3:
                    live += 1
                if result == -1997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 1997:
                    dead -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i+k][j+k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 3:
                    live += 1
                if result == -1997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 1997:
                    dead -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i+k][j-k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 3:
                    live += 1
                if result == -1997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 1997:
                    dead -= 1

        return live, dead

    def _convolution_3_2(self, obs, color):
        filter_1 = color * np.array([1000, 1, 10000, 1, 1, 1000])
        filter_2 = color * np.array([1000, 1, 1, 10000, 1, 1000])
        size = len(filter_1)
        board_size = obs.shape[0]
        live = 0
        dead = 0

        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter_1, obs[i, j:j + size])
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1
                result = np.dot(filter_2, obs[i, j:j + size])
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter_1, obs[i:i + size, j])
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1
                result = np.dot(filter_2, obs[i:i + size, j])
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i + k][j + k] for k in range(size)]
                result = np.dot(filter_1, list)
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1
                result = np.dot(filter_2, list)
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1


        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i + k][j - k] for k in range(size)]
                result = np.dot(filter_1, list)
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1
                result = np.dot(filter_2, list)
                if result == 3:
                    live += 1
                if result == -997:
                    dead += 1
                if result == -3:
                    live -= 1
                if result == 997:
                    dead -= 1

        return live, dead

    def _convolution_3_3(self, obs, color):
        filter = color * np.array([1000, 1, 1, 1000])
        size = len(filter)
        board_size = obs.shape[0]
        live = 0
        dead = 0

        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j + size])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i + size, j])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i + k][j + k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i + k][j - k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        return live, dead

    def _convolution_2_1(self, obs, color):
        filter = color * np.array([1000, 1, 1, 1000])
        size = len(filter)
        board_size = obs.shape[0]
        live = 0
        dead = 0

        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j + size])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i + size, j])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i + k][j + k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i + k][j - k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        return live, dead

    def _convolution_2_2(self, obs, color):
        filter = color * np.array([1000, 1, 10000, 1, 1000])
        size = len(filter)
        board_size = obs.shape[0]
        live = 0
        dead = 0

        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j + size])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i + size, j])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i + k][j + k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i + k][j - k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        return live, dead

    def _convolution_2_3(self, obs, color):
        filter = color * np.array([1000, 1, 10000, 10000, 1, 1000])
        size = len(filter)
        board_size = obs.shape[0]
        live = 0
        dead = 0

        # horizontal
        for i in range(board_size):
            for j in range(board_size - size + 1):
                result = np.dot(filter, obs[i, j:j + size])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # vertical
        for i in range(board_size - size + 1):
            for j in range(board_size):
                result = np.dot(filter, obs[i:i + size, j])
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # diagonal
        for i in range(board_size - size + 1):
            for j in range(board_size - size + 1):
                list = [obs[i + k][j + k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        # anti-diagonal
        for i in range(board_size - size + 1):
            for j in range(size - 1, board_size):
                list = [obs[i + k][j - k] for k in range(size)]
                result = np.dot(filter, list)
                if result == 2:
                    live += 1
                if result == -998:
                    dead += 1
                if result == -2:
                    live -= 1
                if result == 998:
                    dead -= 1

        return live, dead