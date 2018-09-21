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

        if conf['mode'] == 11:
            self._agent_1 = MCTSAgent(conf, color=BLACK, use_stochastic_policy=True)
            self._agent_2 = FastAgent(color=WHITE)

        if conf['mode'] == 12:
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
        max_score = 0

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
                try:
                    legal_moves = list(value.keys())  # here value is score_dict
                    score_list = [value[legal_moves[i]] for i in range(len(legal_moves))]
                    self._board.show_scores(action_list=legal_moves, score_list=score_list)
                    prior_prob, value = None, None
                except AttributeError:
                    print('> using evaluator agent')
                    legal_moves = self._evaluator_agent.generate(obs=self._obs(), all=True)
                    score_list = list()
                    for i in range(len(legal_moves)):
                        x, y = legal_moves[i]
                        temp_board = np.copy(self._obs())
                        temp_board[x][y] = self._current_agent().color
                        self._evaluator_agent.color = self._current_agent().color
                        score_atk, score_def = self._evaluator_agent.evaluate(temp_board)
                        print('pos:', (x, y), ' atk:', score_atk, ' def:', score_def)
                        score = score_atk if score_atk > score_def else -score_def
                        score_list.append(score)
                    self._board.show_scores(action_list=legal_moves, score_list=score_list)
                try:
                    max_score = max(max(score_list), -min(score_list))
                except:
                    max_score = 0
            else:
                prior_prob, value = None, None

            # show info
            if prior_prob is None:
                info = '1_2'
            else:
                prior_prob = str(round(float(prior_prob), 3))
                value = str(round(-value, 3))
                # now value indicates the winning rate of the last player of the current observation
                info = prior_prob + '_' + value

            result = self._check_rules(action)

            if self._conf['mode'] == 12 and self._board.stone_num() >= 30 and max_score < score_3_live:
                result = 'draw'

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
                time.sleep(0.1)
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
                pass
                # end = True
                # break
            if old_model_wins_num > (total_num - draw_num) / 2:
                pass
                # end = True
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
        postfix_pattern = r'self\_play\_15\_\d+\_[0-9a-zA-Z\_\-]+\_col\.npy'
        last_path = ''
        external_data_set = DataSet()
        count = 0
        obs, col, last_move, pi, z = [], [], [], [], []
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
                    new_obs, new_col, new_last_move, new_pi, new_z = external_data_set.get_sample(1)
                    obs.extend(new_obs)
                    col.extend(new_col)
                    last_move.extend(new_last_move)
                    pi.extend(new_pi)
                    z.extend(new_z)
                    external_data_set.clear()
            if count % 5 == 0 and count != 0:
                self._agent_1.train(obs, col, last_move, pi, z)
                obs, col, last_move, pi, z = [], [], [], [], []
                count = 0
                if self.evaluate():
                    self._agent_1.save_model()
                    self._network_version += 1
                print('> network version = ' + str(self._network_version))
        self._agent_1.save_model()

    def _obs(self):
        return self._board.board()

    def _current_agent(self):
        if self._board.current_player() == BLACK:
            return self._agent_1
        else:
            return self._agent_2

    def _check_rules(self, action):
        return self._rules.check_rules(self._board.board(), action, self._board.current_player())

    # step 1. train on generated game record
    # step 2. train on self-play data generated by fast AI
    # step 3. if MCTS Agent is stronger than fast AI, then begin to train on self-play games
    #         if MCTS Agent degenerated, go back to step 2

    def get_generated_data_set(self, sample_num=10000):
        gen = Generator(self._conf['board_size'], max_noise_stone_num=64)
        gen_data_set = DataSet()

        record_1 = gen.generate_live_3_oo_o_attack(sample_num=sample_num)
        gen_data_set.add_record(record_1)
        record_2 = gen.generate_live_3_oo_o_defend(sample_num=sample_num)
        gen_data_set.add_record(record_2)
        record_3 = gen.generate_live_3_ooo_attack(sample_num=sample_num)
        gen_data_set.add_record(record_3)
        record_4 = gen.generate_live_3_ooo_defend(sample_num=sample_num)
        gen_data_set.add_record(record_4)
        record_5 = gen.generate_live_4_attack(sample_num=sample_num)
        gen_data_set.add_record(record_5)
        record_6 = gen.generate_live_4_defend(sample_num=sample_num)
        gen_data_set.add_record(record_6)
        record_7 = gen.generate_dead_4_oo_oo_defend(sample_num=sample_num)
        gen_data_set.add_record(record_7)
        record_8 = gen.generate_dead_4_ooo_o_defend(sample_num=sample_num)
        gen_data_set.add_record(record_8)

        gen_data_set.save(self._conf['generated_data_path'])
        return gen_data_set

    def train_on_generated_data(self):
        gen_data_set = DataSet()
        gen_data_set.load(self._conf['generated_data_path'])
        for i in range(100):
            print('> epoch = ' + str(i + 1))
            obs, col, last_move, pi, z = gen_data_set.get_sample(0.1, shuffle=True)
            self._agent_1.train(obs, col, last_move, pi, z)
            self._agent_1.save_model()
