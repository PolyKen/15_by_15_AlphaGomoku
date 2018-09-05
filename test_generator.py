from AlphaRenju_Zero import *


conf = Config()
ren = Renderer(conf['screen_size'], conf['board_size'])
board = Board(ren, conf['board_size'])
gen = Generator(conf['board_size'], max_noise_stone_num=32)
dataset = DataSet()

record = gen.generate_live_4_attack(sample_num=10000)
dataset.add_record(record)
