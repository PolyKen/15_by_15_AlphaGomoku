from AlphaRenju_Zero import *
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


print('> Please enter the mode: '
      '(1: Training, 2: AI vs Human, 3: Human vs Human, '
      '4: AI vs AI, 5: Collect human play data, 6: Collect self play data)')
mode = int(input('> mode = '))
conf = Config()
conf.set_mode(mode)    # 1: training mode, 2: AI vs Human, 3: Human vs Human, 4: AI vs AI
conf.print_current_config()
env = Env(conf)
if mode == 1 or mode == 0:
    env.train()
if mode == 2 or mode == 3 or mode == 4:
    env.run(is_train=False)
if mode == 5:
    env.collect_human_data()
if mode == 6:
    env.collect_self_play_data()


