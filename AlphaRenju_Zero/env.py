from . import *
from .dataset.dataset import *
import matplotlib.pyplot as plt


class Env:
    def __init__(self, conf):
        if not display_mode:
            conf['display'] = False
            print('> error: display mode is not available (requires pygame and threading)')

        self._conf = conf
        self._is_self_play = conf['is_self_play']

        self._rules = Rules(conf)
        self._renderer = Renderer(conf['screen_size'], conf['board_size']) if conf['display'] else None
        self._board = Board(self._renderer, conf['board_size'])

        self._network_version = 0

        # Training
        if conf['mode'] == 1 or conf['mode'] == 0:
            self._agent_1 = MCTSAgent(conf, color=BLACK)
        # AI vs Human
        if conf['mode'] == 2:
            self._agent_1 = MCTSAgent(conf, color=BLACK)
            self._agent_2 = HumanAgent(self._renderer, color=WHITE, board_size=conf['board_size'])
        # Human vs Human
        if conf['mode'] == 3:
            self._agent_1 = HumanAgent(self._renderer, color=BLACK, board_size=conf['board_size'])
            self._agent_2 = HumanAgent(self._renderer, color=WHITE, board_size=conf['board_size'])
        if conf['mode'] == 4:
            self._agent_1 = MCTSAgent(conf, color=BLACK)
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

    @log
    def run(self, record=None):
        while True:
            if self._is_self_play:
                self._agent_1.color = self._board.current_player()
            action, pi = self._current_agent().play(self._obs(), self._board.last_move(), self._board.stone_num())
            result = self._check_rules(action)
            if result == 'continue':
                # print(result + ': ', action, color)
                if record is not None:
                    record.add(self._obs(), self._board.current_player(), self._board.last_move(), pi)
                self._board.move(self._board.current_player(), action)
            if result == 'occupied':
                print(result + ': ' + str(action))
                continue
            if result == 'blackwins' or result == 'whitewins' or result == 'draw':
                print(result)
                if record is not None:
                    record.add(self._obs(), self._board.current_player(), self._board.last_move(), pi)
                self._board.move(self._board.current_player(), action)

                # add last position of this game
                pi_0 = np.zeros(self._conf['board_size'] * self._conf['board_size'])
                record.add(self._obs(), self._board.current_player(), self._board.last_move(), pi_0)
                if result == 'blackwins':
                    flag = BLACK
                if result == 'whitewins':
                    flag = WHITE
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
        # use human play data to initialize network
        '''
        human_play_data_set = DataSet()
        human_play_data_set.load(self._conf['human_play_data_path'])
        obs, col, last_move, pi, z = human_play_data_set.get_sample(1)
        self._agent_1.train(obs, col, last_move, pi, z)
        self._agent_1.save_model()
        '''

        # training based on self-play
        data_set = DataSet()
        for epoch in range(self._epoch):
            print('> epoch = ' + str(epoch+1))

            # self-play
            for i in range(self._games_num):
                record = GameRecord()
                print('> game num = ' + str(i+1))
                self.run(record)
                data_set.add_record(record)

            # train
            obs, col, last_move, pi, z = data_set.get_sample(self._sample_percentage)
            loss = self._agent_1.train(obs, col, last_move, pi, z)
            self._loss_list.append(loss)

            # evaluate
            if epoch >= self._conf['evaluate_start_epoch'] - 1:
                if self.evaluate():
                    self._agent_1.save_model()
                    self._network_version += 1
                    data_set.clear()
                else:
                    self._agent_1.load_model()
                print('> network version = ' + str(self._network_version))
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
        plt.savefig(self._conf['fit_history_file'] + str('.png'), dpi=300)
        plt.show()

    def evaluate(self):
        print('> Evaluation begins:')

        # switch mode
        self._is_self_play = False
        self._agent_1.set_self_play(False)
        self._agent_2 = self._agent_eval
        self._agent_2.load_model()

        new_model_wins_num = 0
        old_model_wins_num = 0
        total_num = self._evaluate_games_num

        # new model plays BLACK
        for i in range(int(total_num/2)):
            result = self.run()
            if result == 1:
                new_model_wins_num += 1
            if result == -1:
                old_model_wins_num += 1
            print('> eval game ' + str(i+1) + ' , score: ' + str(new_model_wins_num) + ':' + str(old_model_wins_num))

        # switch agents
        self._agent_1, self._agent_2 = self._agent_2, self._agent_1
        self._agent_1.color = BLACK
        self._agent_2.color = WHITE

        for i in range(int(total_num/2)):
            result = self.run()
            if result == 1:
                old_model_wins_num += 1
            if result == -1:
                new_model_wins_num += 1
            print('> eval game ' + str(i + 1) + ' , score: ' + str(new_model_wins_num) + ':' + str(old_model_wins_num))

        # so far self._agent_1 -> self._agent_eval

        self._agent_1 = self._agent_2
        self._agent_1.color = BLACK
        self._agent_1.set_self_play(True)
        self._is_self_play = True

        if new_model_wins_num == 0:
            rate = 0
        else:
            rate = new_model_wins_num / (new_model_wins_num + old_model_wins_num)
        print('> winning rate of new model = ' + str(rate))
        if rate > 0.5:
            print('> New model adopted')
            return True
        else:
            print('> New model discarded')
            return False

    def collect_human_data(self):
        if not self._conf['display']:
            print('> error: please set [display] = True in Config')
            return
        if self._conf['is_self_play']:
            print('> error: please set [is_self_play] = False in Config')
            return

        human_data_set = DataSet()

        for i in range(self._games_num):
            record = GameRecord()
            print('> game num = ' + str(i + 1))
            self.run(record)
            human_data_set.add_record(record)

        human_data_set.save(self._conf['human_play_data_path'])

    def _obs(self):
        return self._board.board()

    def _current_agent(self):
        if self._board.current_player() == BLACK:
            return self._agent_1
        else:
            return self._agent_2

    def _check_rules(self, action):
        return self._rules.check_rules(self._board.board(), action, self._board.current_player())
