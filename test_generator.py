from AlphaRenju_Zero import *


def test_legal(obs_list, color_list, last_move_list, pi_list, z_list):
    for i in range(100):
        if sum(sum(obs_list[i])) == 0 and color_list[i] == WHITE:
            print('error: network.train')
            read_board(obs_list[i], color_list[i], last_move_list[i], pi_list[i], z_list[i])
        if sum(sum(obs_list[i])) == 1 and color_list[i] == BLACK:
            print('error: network.train')


def read_board(obs, color, last_move, pi, z):
    board.read(obs)
    print(obs)
    print('color =', color, ', last move =', last_move, ', pi ind =',
          index2coordinate(list(np.where(pi > 0)[0])[0], 15), ', z =', z)
    time.sleep(1)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


conf = Config()
ren = Renderer(conf['screen_size'], conf['board_size'])
board = Board(ren, conf['board_size'])
gen = Generator(conf['board_size'], max_noise_stone_num=32)

record_l4a = gen.generate_live_4_attack(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_l4a.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_l4d = gen.generate_live_4_defend(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_l4d.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_d4d = gen.generate_dead_4_oooo_defend(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_d4d.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_d4d = gen.generate_dead_4_ooo_o_defend(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_d4d.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_d4d = gen.generate_dead_4_oo_oo_defend(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_d4d.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_l3a = gen.generate_live_3_ooo_attack(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3a.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_l3a = gen.generate_live_3_oo_o_attack(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3a.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_l3d = gen.generate_live_3_ooo_defend(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3d.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)

record_l3d = gen.generate_live_3_oo_o_defend(sample_num=100)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3d.get_sample(1)
test_legal(obs_list, color_list, last_move_list, pi_list, z_list)
