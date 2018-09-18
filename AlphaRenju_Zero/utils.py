import time
import numpy as np


def log(func):
    def wrapper(*args, **kwargs):
        start = time.clock()
        print('>> calling %s()' % func.__name__)
        result = func(*args, **kwargs)
        end = time.clock()
        print('>> %s() time = %s' % (func.__name__, str(end - start)))
        return result

    return wrapper


def index2coordinate(index, size):
    row = index // size
    col = index % size
    return int(row), int(col)


def coordinate2index(cor, size):
    return size * cor[0] + cor[1]


def board2legalvec(board):
    vec = np.array(np.array(board) == 0, dtype=np.int)
    return vec.flatten()
