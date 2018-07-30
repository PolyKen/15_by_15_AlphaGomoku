from AlphaRenju_Zero import *
import warnings
warnings.filterwarnings("ignore")


print('> Please enter the mode: (1: training mode, 2: AI vs Human, 3: Human vs Human)')
mode = int(input('> mode = '))
conf = Config()
conf.set_mode(mode)    # 1: training mode, 2: AI vs Human, 3: Human vs Human
conf.print_current_config()
env = Env(conf)
if mode == 1:
    env.train()
if mode == 2 or mode == 3:
    env.run()


