from .agent import Agent


class HumanAgent(Agent):
    def __init__(self, renderer, color):
        self._renderer = renderer

    def play(self, *args):
        x, y = self._renderer.ask_for_click()
        return (x, y), None
