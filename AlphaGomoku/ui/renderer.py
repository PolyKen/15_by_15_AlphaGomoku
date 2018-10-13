from sys import exit
from ..rules import *
import time
import threading

display_mode = True
use_dialog = True
try:
    import easygui
except ImportError:
    print('> error: module [easygui] not found')
    use_dialog = False
try:
    import pygame
except ImportError:
    print('> error: module [pygame] not found')
    display_mode = False

image_path = 'AlphaGomoku/ui/image/'


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
        self._update_score = False
        self._info_surface_cache = []
        self._info_rect_cache = []
        self._score_surface_cache = []
        self._score_rect_cache = []

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
                    print("> exit")
                    pygame.quit()
                    exit()
                if self._is_waiting_for_click and event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()
                    y = int(mouse_position[0] / self._spacing - 0.5)
                    x = int(mouse_position[1] / self._spacing - 0.5)
                    if x in range(self._board_size) and y in range(self._board_size):
                        self._is_waiting_for_click = False
                        self._mouse_click_pos = (x, y)
                    print("> click " + str(self._mouse_click_pos))
            if self._update_clear:
                self._paint_background()
            if self._update_read:
                self._read(self._new_board)
            if self._update_move:
                self._move(self._next_player, self._next_pos)
            if self._update_info:
                self._show_info()
            if self._update_score:
                self._show_score()

    def paint_background(self):
        self._update_clear = True
        self._update_move = False
        self._update_read = False
        self._init = False

    def _paint_background(self):
        self._screen.blit(self._background, (0, 0))
        black_color = (0, 0, 0)

        for i in range(1, self._board_size + 1):
            start_horizontal = (self._spacing, i * self._spacing)
            end_horizontal = (self._screen_size[1] - self._spacing, i * self._spacing)
            start_vertical = (i * self._spacing, self._spacing)
            end_vertical = (i * self._spacing, self._screen_size[1] - self._spacing)

            if i == 1 or i == self._board_size + 1:
                pygame.draw.line(self._screen, black_color, start_horizontal, end_horizontal, 3)
                pygame.draw.line(self._screen, black_color, start_vertical, end_vertical, 3)
            else:
                pygame.draw.line(self._screen, black_color, start_horizontal, end_horizontal, 2)
                pygame.draw.line(self._screen, black_color, start_vertical, end_vertical, 2)

        if self._board_size % 2 == 1:
            mid = (self._board_size + 1) / 2
            start_pos = (self._spacing * int(mid) - 2, self._spacing * int(mid) - 2)
            size = (6, 6)
            pygame.draw.rect(self._screen, black_color, pygame.rect.Rect(start_pos, size))

        pygame.display.update()
        self._update_clear = False
        self._init = True

    def move(self, player, action, info=None):
        while self._update_move:
            time.sleep(1e-4)
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

        self._update_move = False

    def read(self, new_board):
        while self._update_read:
            time.sleep(1e-4)
        self._new_board = new_board
        self._update_read = True

    def _read(self, new_board):
        self._paint_background()
        self._update_read = False
        for row in range(self._board_size):
            for col in range(self._board_size):
                if new_board[row][col] == 1:
                    self._move(1, (row, col))
                elif new_board[row][col] == -1:
                    self._move(-1, (row, col))

    def ask_for_click(self):
        self._is_waiting_for_click = True
        while self._is_waiting_for_click:
            time.sleep(1e-4)
        return self._mouse_click_pos

    def show_score(self, board, action_list, score_list):
        self.read(board)
        time.sleep(1e-2)
        large_font = pygame.font.SysFont('Calibri', size=20)
        red = (255, 0, 0)

        for a_s in list(zip(action_list, score_list)):
            action, score = a_s[0], a_s[1]
            if self._board_size == 8:
                position = (int((action[1] + 0.63) * self._spacing), int((action[0] + 0.76) * self._spacing))
            if self._board_size == 15:
                position = (int((action[1] + 0.80) * self._spacing), int((action[0] + 0.72) * self._spacing))

            self._score_surface_cache.append(large_font.render(str(round(score, 2)), True, red))
            self._score_rect_cache.append(position)

        self._update_score = True

    def _show_score(self):
        size = len(self._score_rect_cache)
        for i in range(size):
            self._screen.blit(self._score_surface_cache[i], self._score_rect_cache[i])
        self._score_surface_cache = []
        self._score_rect_cache = []

        pygame.display.update()
        self._update_score = False

    def show_info(self, info, player, action):
        infos = info.split('_')
        # p = 'p = ' + infos[0]
        v = infos[1]
        num = infos[2]

        if self._board_size == 8:
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

        if self._board_size == 15:
            # position_1 = (int((action[1] + 0.63) * self._spacing), int((action[0] + 0.76) * self._spacing))
            if float(infos[1]) >= 0:
                position_2 = (int((action[1] + 0.72) * self._spacing), int((action[0] + 0.75) * self._spacing))
            else:
                position_2 = (int((action[1] + 0.70) * self._spacing), int((action[0] + 0.75) * self._spacing))

            if int(num) < 10:
                position_3 = (int((action[1] + 0.90) * self._spacing), int((action[0] + 0.96) * self._spacing))
            else:
                position_3 = (int((action[1] + 0.82) * self._spacing), int((action[0] + 0.96) * self._spacing))

            small_font = pygame.font.SysFont('Calibri', size=10)
            large_font = pygame.font.SysFont('Calibri', size=20)

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


def ask_for_draw():
    if display_mode and use_dialog:
        return easygui.ccbox(title='Request', msg='AlphaRenju requests a draw.', choices=['draw', 'continue'])
    else:
        print('> AlphaRenju requests a draw.')
        return 0


def show_result(mode, result):
    if display_mode and use_dialog and mode in [2, 2.5, 3, 9]:
        info = ''
        if result == 'blackwins':
            info = 'Black wins!'
        if result == 'whitewins':
            info = 'White wins!'
        if result == 'draw':
            info = 'Draw!'
        easygui.msgbox(title='Result', msg=info)
    else:
        print(result)
