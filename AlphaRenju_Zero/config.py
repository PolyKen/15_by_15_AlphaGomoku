class Config(dict):
    def __init__(self, **kwargs):
        # mode   1: training mode, 2: AI vs Human, 3: Human vs Human, 0: Debug
        self['mode'] = 1

        # display mode
        self['display'] = False

        # screen size of renderer
        self['screen_size'] = (720, 720)

        # self play mode
        self['is_self_play'] = True

        # true: 3-3, 4-4, 6+ are not allowed for black
        self['forbidden_moves'] = False

        # PUCT: when c_puct gets smaller, the simulation becomes deeper
        self['c_puct'] = 5

        # simulation times
        self['simulation_times'] = 400

        # initial tau
        self['initial_tau'] = 1

        # proportion of dirichlet noise
        self['epsilon'] = 0.25

        # coef of dirichlet noise
        self['alpha'] = 0.03

        # use dirichlet
        self['use_dirichlet'] = False

        # board size
        self['board_size'] = 8

        # epoch: number of games played to train
        self['epoch'] = 60
        
        # sample percentage
        self['sample_percentage'] = 0.5
        
        # number of games in each training epoch
        self['games_num'] = 50

        # learning rate
        self['learning_rate'] = 2e-3

        # momentum
        self['momentum'] = 9e-1

        # coefficient of l2 penalty
        self['l2'] = 1e-4

        # path of network parameters
        self['net_para_file'] = 'AlphaRenju_Zero/network/model/model_' + str(self['board_size']) + '.h5'

        # path of history of fitting
        self['fit_history_file'] = 'AlphaRenju_Zero/network/history/log_' + str(self['board_size'])

        # human play data path
        self['human_play_data_path'] = 'AlphaRenju_Zero/dataset/human_play_data/human_' + str(self['board_size']) + '_'

        # self play data path
        self['self_play_data_path'] = 'AlphaRenju_Zero/dataset/self_play_data/self_play_' + str(self['board_size']) + '_'
        
        # use previous model
        self['use_previous_model'] = True

        # number of games played for evaluation, must be an even number!!!
        self['evaluate_games_num'] = 10

        # epoch from which evaluation starts
        self['evaluate_start_epoch'] = 1
        
        # Mini-Batch Size
        self['mini_batch_size'] = 512

        # fit epochs, number of each sample used
        self['fit_epochs'] = 20

        # use supervised learning
        self['is_supervised'] = True

        # careful stage
        self['careful_stage'] = 6

        # number of threads
        self['threads_num'] = 32

        # virtual loss
        self['virtual_loss'] = 3

        self.update(**kwargs)

    def update(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]

    def set_mode(self, mode):
        if mode not in [1, 2, 2.5, 3, 4, 5, 6, 7, 0]:
            mode = 1
        if mode == 1:
            self['display'] = False
            self['is_self_play'] = True
            self['mode'] = 1
            print('> Training mode')
        if mode == 2:
            self['display'] = True
            self['is_self_play'] = False
            self['mode'] = 2
            self['simulation_times'] = 3200
            print('> AI vs Human mode')
        if mode == 2.5:
            self['display'] = True
            self['is_self_play'] = False
            self['mode'] = 2.5
            self['simulation_times'] = 3200
            print('> AI vs Human mode')
        if mode == 3:
            self['display'] = True
            self['is_self_play'] = False
            self['mode'] = 3
            print('> Human vs Human mode')
        if mode == 4:
            self['display'] = True
            self['is_self_play'] = False
            self['mode'] = 4
            self['simulation_times'] = 3200
            print('> AI vs AI mode')
        if mode == 5:
            self['display'] = True
            self['is_self_play'] = False
            self['mode'] = 5
            self['games_num'] = 100
            print('> Collect human play data mode')
        if mode == 6:
            self['display'] = False
            self['is_self_play'] = True
            self['mode'] = 6
            self['games_num'] = 30
            self['epoch'] = 20
            print('> Collect self play data mode')
        if mode == 7:
            self['display'] = False
            self['is_self_play'] = True
            self['mode'] = 7
            print('> Train on external data mode')
        if mode == 0:
            self['display'] = True
            self['is_self_play'] = True
            self['mode'] = 0
            self['simulation_times'] = 5
            self['games_num'] = 3
            self['epoch'] = 2
            print('> Debug mode')

    def print_current_config(self):
        print('------------------')
        print('> CURRENT CONFIG:')
        for key in self:
            print('{}: {}'.format(key, self[key]))
        print('------------------')
