from sys import exit
from ..rules import *
import time
display_mode = True
try:
    import pygame
except ImportError:
    print('ERROR: module [pygame] not found')
    display_mode = False
try:
    import threading
except ImportError:
    print('ERROR: module [threading] not found')
    display_mode = False


image_path = 'AlphaRenju_Zero/ui/image/'


class Renderer(threading.Thread):

    # Noted that some functions have both public and private versions such as 'move', 'read', 'paint_background'
    # private ones are for Renderer thread, which will finish the rendering while the public func play the role in
    # sending signals to Renderer thread. (by updating some boolean variables, since Renderer Thread is listening
    # these variables in an endless loop)

    # Since all rendering must be done in Renderer thread, we have to take an indirect way.

    def __init__(self, screen_size, board_size=15):
        super(Renderer, self).__init__()
        self._screen_size = screen_size
        self._board_size = board_size
        self._spacing = int(self._screen_size[1] / (board_size + 1))
        self._screen = None
        self._background = None
        self._stone_black = None
        self._stone_white = None

        self._init = False

        self._update_move = False
        self._next_pos = None
        self._next_player = 0

        self._update_read = False
        self._new_board = None

        self._update_clear = False

        self._update_info = False
        self._info_surface_cache = []
        self._info_rect_cache = []

        self._is_waiting_for_click = False
        self._mouse_click_pos = None

        self.setDaemon(True)
        self.start()

    def run(self):
        pygame.init()
        self._screen = pygame.display.set_mode(self._screen_size, 0, 32)
        self._background = pygame.image.load(image_path + 'desk.jpg').convert()
        self._stone_black = pygame.image.load(image_path + 'black.png').convert_alpha()
        self._stone_white = pygame.image.load(image_path + 'white.png').convert_alpha()
        self._stone_black = pygame.transform.smoothscale(self._stone_black, (self._spacing, self._spacing))
        self._stone_white = pygame.transform.smoothscale(self._stone_white, (self._spacing, self._spacing))
        self.paint_background()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("exit")
                    pygame.quit()
                    exit()
                if self._is_waiting_for_click and event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()
                    y = int(mouse_position[0] / self._spacing - 0.5)
                    x = int(mouse_position[1] / self._spacing - 0.5)
                    if x in range(self._board_size) and y in range(self._board_size):
                        self._is_waiting_for_click = False
                        self._mouse_click_pos = (x, y)
                    print("click" + str(self._mouse_click_pos))
            if self._update_clear:
                self._paint_background()
            if self._update_move:
                self._move(self._next_player, self._next_pos)
            if self._update_info:
                self._show_info()
            if self._update_read:
                self._read(self._new_board)

    def paint_background(self):
        self._update_clear = True
        self._update_move = False
        self._update_read = False
        self._init = False

    def _paint_background(self):
        self._screen.blit(self._background, (0, 0))

        for i in range(1, self._board_size + 1):
            start_horizontal = (self._spacing, i * self._spacing)
            end_horizontal = (self._screen_size[1] - self._spacing, i * self._spacing)
            start_vertical = (i * self._spacing, self._spacing)
            end_vertical = (i * self._spacing, self._screen_size[1] - self._spacing)

            if i == 1 or i == self._board_size + 1:
                pygame.draw.line(self._screen, (0, 0, 0), start_horizontal, end_horizontal, 3)
                pygame.draw.line(self._screen, (0, 0, 0), start_vertical, end_vertical, 3)
            else:
                pygame.draw.line(self._screen, (0, 0, 0), start_horizontal, end_horizontal, 2)
                pygame.draw.line(self._screen, (0, 0, 0), start_vertical, end_vertical, 2)

        pygame.display.update()
        self._update_clear = False
        self._init = True

    def move(self, player, action, info=None):
        while self._update_move:
            time.sleep(.1)
        self._next_player = player
        self._next_pos = action
        self._update_move = True
        if info is not None:
            self.show_info(info, player, action)

    def _move(self, player, action):
        position = (int((action[1] + 0.5) * self._spacing), int((action[0] + 0.5) * self._spacing))
        if player == BLACK:
            self._screen.blit(self._stone_black, position)
        elif player == -1:
            self._screen.blit(self._stone_white, position)

        pygame.display.update()
        self._update_move = False

    def read(self, new_board):
        while self._update_read:
            time.sleep(.1)
        self._new_board = new_board
        self._update_read = True

    def _read(self, new_board):
        self._paint_background()
        for row in range(self._board_size):
            for col in range(self._board_size):
                if new_board[row][col] == 1:
                    self._move(1, (row, col))
                elif new_board[row][col] == -1:
                    self._move(-1, (row, col))

        pygame.display.update()
        self._update_read = False

    def ask_for_click(self):
        self._is_waiting_for_click = True
        while self._is_waiting_for_click:
            time.sleep(.01)
        return self._mouse_click_pos

    def show_info(self, info, player, action):
        infos = info.split('_')
        # p = 'p = ' + infos[0]
        v = 'v = ' + infos[1]
        num = infos[2]

        # position_1 = (int((action[1] + 0.63) * self._spacing), int((action[0] + 0.76) * self._spacing))
        if float(infos[1]) >= 0:
            position_2 = (int((action[1] + 0.62) * self._spacing), int((action[0] + 0.78) * self._spacing))
        else:
            position_2 = (int((action[1] + 0.61) * self._spacing), int((action[0] + 0.78) * self._spacing))

        if int(num) < 10:
            position_3 = (int((action[1] + 0.90) * self._spacing), int((action[0] + 0.96) * self._spacing))
        else:
            position_3 = (int((action[1] + 0.82) * self._spacing), int((action[0] + 0.96) * self._spacing))

        small_font = pygame.font.SysFont('Calibri', size=16)
        large_font = pygame.font.SysFont('Calibri', size=32)

        color = (255, 0, 0)
        if player == BLACK:
            color = (255, 255, 255)
        if player == WHITE:
            color = (0, 0, 0)

        # self._info_surface_cache.append(small_font.render(p, True, color))
        # self._info_rect_cache.append(position_1)

        if infos[1] != '2':
            self._info_surface_cache.append(small_font.render(v, True, color))
            self._info_rect_cache.append(position_2)

        self._info_surface_cache.append(large_font.render(num, True, color))
        self._info_rect_cache.append(position_3)
        self._update_info = True

    def _show_info(self):
        size = len(self._info_rect_cache)
        for i in range(size):
            self._screen.blit(self._info_surface_cache[i], self._info_rect_cache[i])
        self._info_surface_cache = []
        self._info_rect_cache = []

        pygame.display.update()
        self._update_info = False

    def is_initialized(self):
        return self._init