class Config(dict):
    def __init__(self, **kwargs):
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

        # the stage after which we set tau = 0
        self['careful_stage'] = 6

        # proportion of dirichlet noise
        self['epsilon'] = 0.25

        # coef of dirichlet noise
        self['dirichlet'] = 0.03

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
        
        # use previous model
        self['use_previous_model'] = True

        # number of games played for evaluation, must be an even number!!!
        self['evaluate_games_num'] = 10

        # epoch from which evaluation starts
        self['evaluate_start_epoch'] = 30
        
        # Mini-Batch Size
        self['mini_batch_size'] = 512

        # fit epochs, number of each sample used
        self['fit_epochs'] = 1

        self.update(**kwargs)

    def update(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]

    def print_current_config(self):
        print('------------------')
        print('CURRENT CONFIG:')
        for key in self:
            print('{}: {}'.format(key, self[key]))
        print('------------------')
