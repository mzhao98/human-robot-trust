import sys, time, random
sys.path.append('./mission/js/py_files')

# import pygame as pg
# # define some colors (R, G, B)
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# DARKGREY = (40, 40, 40)
# LIGHTGREY = (100, 100, 100)
# GREEN = (0, 255, 0)
# RED = (255, 0, 0)
# YELLOW = (255, 255, 0)
# WHITE = pg.Color('white')
# BLACK = pg.Color('black')
# LIGHT_GREY = pg.Color('grey59')
# GREY = pg.Color('grey50')
# RED = pg.Color('red')
# GREEN = pg.Color('green')
# DARK_GREEN = pg.Color('green4')
# BLUE = pg.Color('blue')
# MIDNIGHT_BLUE = pg.Color('midnightblue')
# MARINE_BLUE = pg.Color('royalblue4')
# ORANGE = pg.Color('orange')
# YELLOW = pg.Color('yellow4')
# LIGHT_YELLOW = pg.Color('yellow')
# GOLD = pg.Color('gold3')
# BROWN = pg.Color('brown4')
# BROWN_RED = pg.Color('brown')
# MAROON = pg.Color('maroon')
# TURQUOISE = pg.Color('turquoise')

# game settings
# WIDTH = 1116   # 93 * 12
# HEIGHT = 600  # 50 * 12
WIDTH = 660   # 11 * 60
HEIGHT = 660  # 11 * 60

FPS = 60
TITLE = "Search and Rescue"
# BGCOLOR = WHITE

TILESIZE = 60
GRIDWIDTH = WIDTH / TILESIZE # 10
GRIDHEIGHT = HEIGHT / TILESIZE # 10

SLIGHTLY_INJURED = 0
SEVERELY_INJURED = 1
TRIAGED = 2
DEAD = 3

PLAYER_IMG = 'triangle.png'

CORNER_MAP_IMG = 'emptymap.png'

NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4

LEFT = 0
RIGHT = 1
AROUND = 2
BEHIND = 3
AHEAD = 4