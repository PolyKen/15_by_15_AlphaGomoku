from . import *
from .dataset.dataset import *
import matplotlib.pyplot as plt
import os
import re
from .rules import *


class Env:
    def __init__(self, conf):
        if not display_mode:
            conf['display'] = False
            print('> error: display mode is not available (requires pygame and threading)')

        self._conf = conf
        self._is_self_play = conf['is_self_play']
        self._show_score = conf['show_score']

        self._rules = Rules(conf)
        self._renderer = Renderer(conf['screen_size'], conf['board_size']) if conf['display'] else None
        self._board = Board(self._renderer, conf['board_size'])
        self._value_list = []

        self._network_version = 0

        self._evaluator_agent = FastAgent(color=BLACK)

        # Training
        if conf['mode'] in [0, 1, 6, 7]:
            self._agent_1 = MCTSAgent(conf, color=BLACK, use_stochastic_policy=True)
            self._agent_2 = None
        # AI vs Human
        if conf['mode'] == 2:
            self._agent_1 = MCTSAgent(conf, color=BLACK, use_stochastic_policy=False)
            self._agent_2 = HumanAgent(self._renderer, color=WHITE, board_size=conf['board_size'])
        if conf['mode'] == 2.5:
            self._agent_1 = HumanAgent(self._renderer, color=BLACK, board_size=conf['board_size'])
            self._agent_2 = MCTSAgent(conf, color=WHITE, use_stochastic_policy=False)
        # Human vs Human
        if conf['mode'] == 3 or conf['mode'] == 5:
            self._agent_1 = HumanAgent(self._renderer, color=BLACK, board_size=conf['board_size'])
            self._agent_2 = HumanAgent(self._renderer, color=WHITE, board_size=conf['board_size'])
        if conf['mode'] == 4:
            self._agent_1 = MCTSAgent(conf, color=BLACK, use_stochastic_policy=False)
            self._agent_2 = MCTSAgent(conf, color=WHITE, use_stochastic_policy=False)

        if conf['mode'] == 8:
            self._agent_1 = MCTSAgent(conf, color=BLACK, use_stochastic_policy=False)
            self._agent_2 = HumanAgent(self._renderer, color=WHITE, board_size=conf['board_size'])

        if conf['mode'] == 9:
            self._agent_1 = FastAgent(color=BLACK)
            self._agent_2 = HumanAgent(self._renderer, color=WHITE, board_size=conf['board_size'])

        if conf['mode'] == 10:
            self._agent_1 = FastAgent(color=BLACK)
            self._agent_2 = FastAgent(color=WHITE)

        if conf['mode'] in [0, 1, 7]:
            self._agent_eval = MCTSAgent(conf, color=WHITE, use_stochastic_policy=False)
            self._agent_eval.set_self_play(False)

        if self._is_self_play:
            self._agent_2 = self._agent_1

        self._epoch = conf['epoch']
        self._sample_percentage = conf['sample_percentage']
        self._games_num = conf['games_num']
        self._evaluate_games_num = conf['evaluate_games_num']

        self._loss_list = []

    @log
    def run(self, use_stochastic_policy, record=None):
        if type(self._agent_1) == MCTSAgent:
            self._agent_1.set_stochastic_policy(use_stochastic_policy)
        if type(self._agent_2) == MCTSAgent:
            self._agent_2.set_stochastic_policy(use_stochastic_policy)

        self._value_list = []
        Node.count = 0

        while True:
            if self._is_self_play:
                self._agent_1.color = self._board.current_player()

            # input.obs: current board
            # input.action: the last move of current board
            # input.stone_num: the stone num of current board
            # output.action: the action given by current agent
            # output.pi: the action distribution given by current agent, it will be added in the game record
            # output.prior_prob: the prior probability of this action given by the neural network of current agent
            # output.value: the winning rate given by the current agent
            action, pi, prior_prob, value = self._current_agent().play(obs=self._obs(), action=self._board.last_move(),
                                                                       stone_num=self._board.stone_num())

            # show score: an agent will work as an evaluator, giving its evaluation of each possible position
            if self._show_score:
                legal_moves = self._evaluator_agent.generate(self._obs(), all=True)
                score_list = []
                for legal_move in legal_moves:
                    temp_board = np.copy(self._obs())
                    temp_board[legal_move[0]][legal_move[1]] = self._board.current_player()
                    score = self._evaluator_agent.evaluate(temp_board)
                    score_list.append(score)
                self._board.show_scores(action_list=legal_moves, score_list=score_list)

            if prior_prob is None:
                info = '1_2'
            else:
                prior_prob = str(round(float(prior_prob), 3))
                value = str(round(-value, 3))
                # now value indicates the winning rate of the last player of the current observation
                info = prior_prob + '_' + value

            result = self._check_rules(action)
            if result == 'continue':
                if record is not None:
                    record.add(self._obs(), self._board.current_player(), self._board.last_move(), pi)

                # self._evaluator_agent.color = self._board.current_player()
                self._board.move(self._board.current_player(), action, info)
                # print(self._evaluator_agent.evaluate(self._obs()))

                if value is not None:
                    self._value_list.append(float(value))
                if len(self._value_list) >= 5 and self._board.stone_num() >= 30:
                    if self._conf['mode'] in [2, 2.5] and sum(list(map(np.abs, self._value_list[-5:]))) < 0.06:
                        self._value_list = []
                        if ask_for_draw() == 1:
                            show_result(2, 'draw')
                            time.sleep(20)
                            break

            if result == 'occupied':
                print(result + ': ' + str(action))
                continue
            if result == 'blackwins' or result == 'whitewins' or result == 'draw':
                if record is not None:
                    record.add(self._obs(), self._board.current_player(), self._board.last_move(), pi)
                self._board.move(self._board.current_player(), action, info)

                show_result(self._conf['mode'], result)

                if record is not None:
                    if result == 'blackwins':
                        flag = BLACK
                    if result == 'whitewins':
                        flag = WHITE
                    if result == 'draw':
                        flag = 0
                    record.set_z(flag)
                if self._conf['mode'] in [2, 2.5, 3, 4, 9]:
                    time.sleep(20)
                break
        self._board.clear()
        print('> Node number of game tree = ' + str(Node.count))
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
        if self._conf['is_supervised']:
            human_play_data_set = DataSet()
            human_play_data_set.load(self._conf['human_play_data_path'])
            obs, col, last_move, pi, z = human_play_data_set.get_sample(1)
            print('> ' + str(len(obs)) + ' positions of data loaded')
            for i in range(50):
                print('supervise stage = ' + str(i + 1))
                new_obs = obs.copy()
                new_col = col.copy()
                new_last_move = last_move.copy()
                new_pi = pi.copy()
                new_z = z.copy()
                self._agent_1.train(new_obs, new_col, new_last_move, new_pi, new_z)

        self._agent_1.save_model()

        # training based on self-play
        data_set = DataSet()
        for epoch in range(self._epoch):
            print('> epoch = ' + str(epoch + 1))

            # self-play
            for i in range(self._games_num):
                record = GameRecord()
                print('> game num = ' + str(i + 1))
                self.run(use_stochastic_policy=True, record=record)
                data_set.add_record(record)

            # train
            obs, col, last_move, pi, z = data_set.get_sample(self._sample_percentage)
            loss = self._agent_1.train(obs, col, last_move, pi, z)
            self._loss_list.append(loss)

            # evaluate
            self.evaluate()
            self._agent_1.save_model()
            self._network_version += 1
            data_set.clear()
            print('> network version = ' + str(self._network_version))
            print('*****************************************************')

        # save loss
        hist_path = self._conf['fit_history_file'] + '_loss.txt'
        with open(hist_path, 'a') as f:
            f.write(str(self._loss_list))
        # plot loss
        x = range(1, len(self._loss_list) + 1)
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
        draw_num = 0
        total_num = self._evaluate_games_num
        end = False

        # new model plays BLACK
        for i in range(int(total_num / 2)):
            result = self.run(use_stochastic_policy=True, record=None)
            if result == BLACK:
                new_model_wins_num += 1
            if result == WHITE:
                old_model_wins_num += 1
            if result == 0:
                draw_num += 1
            print('> eval game ' + str(i + 1) + ' , score: ' + str(new_model_wins_num) + ':' + str(old_model_wins_num))
            if new_model_wins_num > (total_num - draw_num) / 2:
                end = True
                # break
            if old_model_wins_num > (total_num - draw_num) / 2:
                end = True
                # break

        # switch agents
        self._agent_1, self._agent_2 = self._agent_2, self._agent_1
        self._agent_1.color = BLACK
        self._agent_2.color = WHITE

        if not end:
            for i in range(int(total_num / 2)):
                result = self.run(use_stochastic_policy=True, record=None)
                if result == BLACK:
                    old_model_wins_num += 1
                if result == WHITE:
                    new_model_wins_num += 1
                if result == 0:
                    draw_num += 1
                print('> eval game ' + str(i + 1 + int(total_num / 2)) + ' , score: ' + str(
                    new_model_wins_num) + ':' + str(old_model_wins_num))
                if new_model_wins_num > (total_num - draw_num) / 2:
                    pass
                    # break
                if old_model_wins_num > (total_num - draw_num) / 2:
                    pass
                    # break

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
        human_data_set = DataSet()
        human_data_set.load(self._conf['human_play_data_path'])

        for i in range(self._games_num):
            record = GameRecord()
            print('> game num = ' + str(i + 1))
            self.run(use_stochastic_policy=False, record=record)
            human_data_set.add_record(record)
            human_data_set.save(self._conf['human_play_data_path'])

    def collect_human_vs_ai_data(self):
        data_set = DataSet()
        data_set.load(self._conf['human_play_data_path'])

        for i in range(self._games_num):
            record = GameRecord()
            print('> game num = ' + str(i + 1))
            self.run(use_stochastic_policy=False, record=record)
            data_set.add_record(record)
            if i % 10 == 0:
                data_set.save(self._conf['human_play_data_path'])

        data_set.save(self._conf['human_play_data_path'])

    def collect_self_play_data(self):
        name = os.getenv('computername')
        for epoch in range(self._epoch):
            print('> epoch = ' + str(epoch + 1))
            data_set = DataSet()
            path = self._conf['self_play_data_path'] + str(epoch + 1) + '_' + str(name) + '_'
            for i in range(self._games_num):
                record = GameRecord()
                print('> game num = ' + str(i + 1))
                self.run(use_stochastic_policy=True, record=record)
                data_set.add_record(record)
                data_set.save(path)
            data_set.save(path)

    def train_on_external_data(self):
        root, prefix = os.path.split(self._conf['self_play_data_path'])
        postfix_pattern = r'self\_play\_8\_\d+\_[0-9a-zA-Z\_\-]+\_col\.npy'
        last_path = ''
        external_data_set = DataSet()
        count = 0
        for filename in os.listdir(root):
            if re.match(postfix_pattern, filename):
                path = root + '/' + filename
                path = path[0:-7]
                if path != last_path:
                    print('> data no.' + str(count + 1))
                    count += 1
                    print('> external data path = ' + path)
                    last_path = path
                    external_data_set.load(path)
                    obs, col, last_move, pi, z = external_data_set.get_sample(1)
                    self._agent_1.train(obs, col, last_move, pi, z)
                    external_data_set.clear()
        self.evaluate()
        self._agent_1.save_model()
        self._network_version += 1
        print('> network version = ' + str(self._network_version))

    def _obs(self):
        return self._board.board()

    def _current_agent(self):
        if self._board.current_player() == BLACK:
            return self._agent_1
        else:
            return self._agent_2

    def _check_rules(self, action):
        return self._rules.check_rules(self._board.board(), action, self._board.current_player())
