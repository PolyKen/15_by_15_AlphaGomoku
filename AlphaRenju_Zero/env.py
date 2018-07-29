from . import *
from .dataset.dataset import *
import time
import matplotlib.pyplot as plt


class Env:
    def __init__(self, conf):
        if not display_mode:
            conf['display'] = False
            print('display mode is not available (requires pygame and threading)')

        self._conf = conf
        self._is_self_play = conf['is_self_play']

        self._rules = Rules(conf)
        self._renderer = Renderer(conf['screen_size'], conf['board_size']) if conf['display'] else None
        self._board = Board(self._renderer, conf['board_size'])

        self._network_version = 0

        self._agent_1 = MCTSAgent(conf, color=BLACK)
        # self._agent_1 = HumanAgent(self._renderer, color=BLACK)
        # self._agent_2 = HumanAgent(self._renderer, color=WHITE)
        self._agent_2 = MCTSAgent(conf, color=WHITE)
        self._agent_eval = MCTSAgent(conf, color=WHITE)
        self._agent_eval.set_self_play(False)

        if self._is_self_play:
            self._agent_2 = self._agent_1

        self._epoch = conf['epoch']
        self._sample_percentage = conf['sample_percentage']
        self._games_num = conf['games_num']
        self._evaluate_games_num = conf['evaluate_games_num']

        self._loss_list = []

    def run(self, record=None):
        result = None
        while True:
            if self._is_self_play:
                self._agent_1.color = self._board.current_player()
            action, pi = self._current_agent().play(self._obs(), self._board.last_move(), self._board.stone_num())
            result = self._check_rules(action)
            if result == 'continue':
                color = self._board.current_player()
                # print(result + ': ', action, color)
                self._board.move(color, action)
                if record is not None and pi is not None:
                    obs = self._board.board()
                    record.add(obs, color, pi)
            if result == 'occupied':
                print(result + ': ' + str(action))
                continue
            if result == 'blackwins' or result == 'whitewins' or result == 'draw':
                self._board.move(self._board.current_player(), action)
                print(result)
                color = self._board.current_player()
                self._board.move(color, action)
                if record is not None and pi is not None:
                    obs = self._board.board()
                    record.add(obs, color, pi)
                    if result == 'blackwins':
                        flag = 1
                    if result == 'whitewins':
                        flag = -1
                    if result == 'draw':
                        flag = 0
                    record.set_z(flag)
                break
        self._board.clear()
        if type(self._agent_1) == MCTSAgent:
            self._agent_1.reset_mcts()
        if type(self._agent_2) == MCTSAgent:
            self._agent_2.reset_mcts()
        if result == 'blackwins':
            return BLACK
        if result == 'whitewins':
            return WHITE
        return 0

    def train(self):
        data_set = DataSet()
        for epoch in range(self._epoch):
            print('epoch = ' + str(epoch+1))

            # self-play
            for i in range(self._games_num):
                record = GameRecord()
                print('game_num = ' + str(i+1))
                start = time.clock()
                self.run(record)
                end = time.clock()
                print('game time = ' + str(end-start))
                data_set.add_record(record)

            # train
            obs, col, pi, z = data_set.get_sample(self._sample_percentage)
            loss = self._agent_1.train(obs, col, pi, z)
            self._loss_list.append(loss)

            # evaluate
            if epoch >= self._conf['evaluate_start_epoch'] - 1:
                if self.evaluate():
                    self._agent_1.save_model()
                    self._network_version += 1
                    data_set.clear()
                else:
                    self._agent_1.load_model()
                print('network version = ' + str(self._network_version))
            else:
                self._agent_1.save_model()
                data_set.clear()
            print('*****************************************************')

        # save loss
        hist_path = self._conf['fit_history_file'] + '_loss.txt'
        with open(hist_path, 'a') as f:
            f.write(str(self._loss_list))
        # plot loss
        x = range(1, len(self._loss_list)+1)
        y = self._loss_list
        plt.plot(x, y)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    def evaluate(self):
        print('Evaluation begins:')

        # switch mode
        self._is_self_play = False
        self._agent_1.set_self_play(False)
        self._agent_2 = self._agent_eval
        self._agent_2.load_model()

        new_model_wins_num = 0
        total_num = self._evaluate_games_num

        for i in range(int(total_num/2)):
            new_model_wins_num += max(self.run(), 0)   # new model plays BLACK
            print('number of new model wins: ' + str(new_model_wins_num) + '/' + str(i+1))

        # switch agents
        self._agent_1, self._agent_2 = self._agent_2, self._agent_1
        self._agent_1.color = BLACK
        self._agent_2.color = WHITE

        for i in range(int(total_num/2)):
            new_model_wins_num -= min(self.run(), 0)
            print('number of new model wins: ' + str(new_model_wins_num) + '/' + str(i+1+int(total_num/2)))

        # so far self._agent_1 -> self._agent_eval

        self._agent_1 = self._agent_2
        self._agent_1.color = BLACK
        self._agent_1.set_self_play(True)
        self._is_self_play = True

        rate = new_model_wins_num / total_num
        print('winning rate = ' + str(rate))
        if rate > 0.55:
            print('adopt new model')
            return True
        else:
            print('discard new model')
            return False

    def _obs(self):
        return self._board.board()

    def _current_agent(self):
        if self._board.current_player() == BLACK:
            return self._agent_1
        else:
            return self._agent_2

    def _check_rules(self, action):
        return self._rules.check_rules(self._board.board(), action, self._board.current_player())
