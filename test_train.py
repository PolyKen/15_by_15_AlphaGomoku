from AlphaGomoku import *
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

conf = Config()
conf.set_mode(7)
env = Env(conf)
env.train()
