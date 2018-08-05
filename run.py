from AlphaRenju_Zero import *
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


print('> Please enter the mode: '
      '(1: Training, 2: AI vs Human, 3: Human vs Human, '
      '4: AI vs AI, 5: Collect human play data, 6: Collect self play data, '
      '7: Train on external data)')
mode = int(input('> mode = '))

if mode == 2:
    print('> Please select your color: (1: Black, 0: White)')
    is_black = int(input('> color = '))
    if is_black == 1:
        mode = 2.5

conf = Config()
conf.set_mode(mode)
env = Env(conf)

if mode == 1 or mode == 0:
    env.train()
if mode in [2, 2.5, 3, 4]:
    env.run(is_train=False)
if mode == 5:
    env.collect_human_data()
if mode == 6:
    env.collect_self_play_data()
if mode == 7:
    env.train_on_external_data()