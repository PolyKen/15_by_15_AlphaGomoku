import random

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
        pi = []
        z = []
        for record in self._game_record:
            a, b, c, d = record.get_sample(percentage)
            obs.extend(a)
            col.extend(b)
            pi.extend(c)
            z.extend(d)
        return obs, col, pi, z

    def record_num(self):
        return len(self._game_record)
    

class GameRecord:
    def __init__(self):
        self._obs_list = []
        self._color_list = []
        self._pi_list = []
        self._z_list = []
        self._total_num = 0

    def add(self, obs, color, pi, z=None):
        self._obs_list.append(obs)
        self._color_list.append(color)
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
        pi_sample = [self._pi_list[index] for index in indices]
        z_sample = [self._z_list[index] for index in indices]
        return obs_sample, color_sample, pi_sample, z_sample

