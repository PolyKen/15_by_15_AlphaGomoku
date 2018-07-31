import time


def log(func):
    def wrapper(*args, **kwargs):
        start = time.clock()
        print('>> calling %s()' % func.__name__)
        result = func(*args, **kwargs)
        end = time.clock()
        print('>> %s() time = %s' % (func.__name__, str(end-start)))
        return result
    return wrapper
