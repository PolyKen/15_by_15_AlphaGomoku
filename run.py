from AlphaGomoku import *
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

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
if mode in [2, 2.5, 3, 9, 10]:
    env.run(use_stochastic_policy=False)
if mode == 4:
    env.mcts_vs_fast(game_num=20)
if mode == 5:
    env.collect_human_data()
if mode in [6, 12]:
    env.collect_self_play_data()
if mode == 7:
    env.train_on_external_data()
if mode == 8:
    env.collect_human_vs_ai_data()
if mode == 11:
    env.train_on_generated_data()
if mode == 13:
    env.self_play_and_train()
