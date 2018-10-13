from AlphaGomoku import *
import warnings
import os
import multiprocessing as mp


def select_mode():
    print('> Please enter the mode:')
    print('> 1: Training (not available)')
    print('> 2: AI vs Human')
    print('> 3: Human vs Human')
    print('> 4: AI vs AI')
    print('> 5: Collect human play data')
    print('> 6: Collect self play data')
    print('> 7: Train on external data')
    print('> 8: Collect human vs AI play data')
    print('> 9: AI(NaiveAgent) vs Human mode')
    print('> 10: AI vs AI(NaiveAgent) mode)')
    print('> 11: Train on generated data')
    print('> 12: Collect self play data(Fast AI)')
    print('> 13: Self play and train')
    _mode = int(input('> mode = '))

    if _mode == 2:
        print('> Please select your color: (1: Black, 0: White)')
        is_black = int(input('> color = '))
        if is_black == 1:
            _mode = 2.5

    return _mode


def start(_mode):
    conf = Config()
    conf.set_mode(_mode)
    _env = Env(conf)
    _env.init_mode(_mode)


if __name__ == '__main__':
    # ignore warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")

    mode = select_mode()
    mp.freeze_support()
    if mode == 13:
        cores_num = mp.cpu_count()
        pool = mp.Pool(processes=cores_num)
        while True:
            pool.map(func=start, iterable=[6] * cores_num)
            start(7)
    else:
        start(mode)
