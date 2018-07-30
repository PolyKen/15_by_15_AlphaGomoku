from AlphaRenju_Zero import *
import warnings
warnings.filterwarnings("ignore")


conf = Config()
conf.set_mode(1)    # 1: training mode, 2: AI vs Human, 3: Human vs Human
conf.print_current_config()
env = Env(conf)
env.train()


