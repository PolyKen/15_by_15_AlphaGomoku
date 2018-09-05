from AlphaRenju_Zero import *

conf = Config()
ren = Renderer(conf['screen_size'], conf['board_size'])
board = Board(ren, conf['board_size'])
gen = Generator(conf['board_size'], max_noise_stone_num=32)

record_l4a = gen.generate_live_4_attack(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_l4a.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)

record_l4d = gen.generate_live_4_defend(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_l4d.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)

record_d4d = gen.generate_dead_4_oooo_defend(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_d4d.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)
record_d4d = gen.generate_dead_4_ooo_o_defend(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_d4d.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)
record_d4d = gen.generate_dead_4_oo_oo_defend(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_d4d.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)

record_l3a = gen.generate_live_3_ooo_attack(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3a.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)
record_l3a = gen.generate_live_3_oo_o_attack(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3a.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)

record_l3d = gen.generate_live_3_ooo_defend(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3d.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)
record_l3d = gen.generate_live_3_oo_o_defend(sample_num=1)
obs_list, color_list, last_move_list, pi_list, z_list = record_l3d.get_sample(1)
board.read(obs_list[0])
print('color =', color_list[0], ', last move =', last_move_list[0], ', pi ind =',
      index2coordinate(list(np.where(pi_list[0] > 0)[0])[0], 15), ', z =', z_list[0])
time.sleep(5)
