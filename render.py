import pygame
import numpy as np
import sys
from cnf import WIDTH, HEIGHT, col_threshold, BLACK, bg_alpha, move_without_render, t, framerate, density, drag_coeff
import time
from visual import button
import json
from time import sleep
pygame.init()
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = input()
sleep(2)

pause = False
screen = pygame.display.set_mode((WIDTH, HEIGHT))
surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
surface.convert()
screen.fill(BLACK)
np.set_printoptions(suppress=True)
clock = pygame.time.Clock()

with open(path) as f:
    content = json.load(f)
all_x = content["x"]
all_m = content["m"]
all_c = content["c"]


# x_pre = all_x[0]
for i in range(len(all_x)):
    surface.fill((0, 0, 0, bg_alpha))
    x = all_x[i]
    m = all_m[i]
    color = all_c[i]
    # print(x)
    clock.tick(framerate)
    out = ""
    for i in range(len(x)):
        px, py = x[i]
        # px_p, py_p = x_pre[i]
        
        if m[i] > 0 and x[i][0] > 0 and x[i][1] > 0:
            out += str(x[i])
            r = int((m[i] ** (1 / 3)) * density)
            pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(px - r / 2, py - r / 2, r, r))
            # pygame.draw.line(surface, (255, 255, 255), (px, py), (px_p, py_p), r)
    print(out)
    # x_pre = x
    screen.blit(surface, (0, 0))
    pygame.display.update()
