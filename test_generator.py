from AlphaRenju_Zero import *


conf = Config()
ren = Renderer(conf['screen_size'], conf['board_size'])
board = Board(ren, conf['board_size'])
gen = Generator(conf['board_size'], max_noise_stone_num=32)
test_board_list = gen.generate_live_4_attack(sample_num=10000)
for bd in test_board_list:
    board.read(bd)
    time.sleep(3)
