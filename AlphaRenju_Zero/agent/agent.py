from abc import abstractmethod
from ..rules import *


class Agent:
    def __init__(self, color):
        if color != BLACK or color != WHITE:
            self._color = BLACK
        else:
            self._color = color

    @abstractmethod
    def play(self, *args, **kwargs):
        pass

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if value != BLACK and value != WHITE:
            self._color = BLACK
        else:
            self._color = value
