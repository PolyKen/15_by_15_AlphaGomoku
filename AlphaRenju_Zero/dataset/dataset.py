import random
import numpy as np


class DataSet:
    def __init__(self):
        self._game_record = []

    def clear(self):
        self._game_record = []
        
    def add_record(self, record):
        self._game_record.append(record)
        
    def get_sample(self, percentage):
        obs = []
        col = []
        last_move = []
        pi = []
        z = []
        for record in self._game_record:
            a, b, c, d, e = record.get_sample(percentage)
            obs.extend(a)
            col.extend(b)
            last_move.extend(c)
            pi.extend(d)
            z.extend(e)
        return obs, col, last_move, pi, z

    def record_num(self):
        return len(self._game_record)

    def save(self, path):
        obs, col, last_move, pi, z = self.get_sample(1)

        obs_path = path + 'obs'
        np.save(obs_path, obs)

        col_path = path + 'col'
        np.save(col_path, col)

        last_move_path = path + 'last_move'
        np.save(last_move_path, last_move)

        pi_path = path + 'pi'
        np.save(pi_path, pi)

        z_path = path + 'z'
        np.save(z_path, z)

        print(str(len(self._game_record)) + ' games of data saved')

    def load(self, path):
        obs_path = path + 'obs.npy'
        obs = np.load(obs_path)

        col_path = path + 'col.npy'
        col = np.load(col_path)

        last_move_path = path + 'last_move.npy'
        last_move = np.load(last_move_path)

        pi_path = path + 'pi.npy'
        pi = np.load(pi_path)

        z_path = path + 'z.npy'
        z = np.load(z_path)

        size = len(z)
        record = GameRecord()
        for i in range(size):
            record.add(obs[i], col[i], last_move[i], pi[i], z[i])
        self.add_record(record)
    

class GameRecord:
    def __init__(self):
        self._obs_list = []
        self._color_list = []
        self._last_move_list = []
        self._pi_list = []
        self._z_list = []
        self._total_num = 0

    def add(self, obs, color, last_move, pi, z=None):
        self._obs_list.append(obs)
        self._color_list.append(color)
        self._last_move_list.append(last_move)
        self._pi_list.append(pi)
        self._z_list.append(z)
        self._total_num += 1

# the method to define the value of z
    def set_z(self, result):
        if result == 0:
            self._z_list = [0 for i in range(self._total_num)]
            return
        for i in range(self._total_num):
            if result == self._color_list[i]:
                self._z_list[i] = 1
            else:
                self._z_list[i] = -1

    def get_sample(self, percentage):
        sample_num = int(self._total_num * percentage)
        indices = random.sample([i for i in range(self._total_num)], sample_num)
        obs_sample = [self._obs_list[index] for index in indices]
        color_sample = [self._color_list[index] for index in indices]
        last_move_sample = [self._last_move_list[index] for index in indices]
        pi_sample = [self._pi_list[index] for index in indices]
        z_sample = [self._z_list[index] for index in indices]
        return obs_sample, color_sample, last_move_sample, pi_sample, z_sample

