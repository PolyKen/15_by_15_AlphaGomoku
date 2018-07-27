from AlphaRenju_Zero import *
import warnings
warnings.filterwarnings("ignore")


conf = Config()
conf.print_current_config()
env = Env(conf)
env.train()
