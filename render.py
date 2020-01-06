import pygame
import numpy as np
import sys
from cnfrender import (
    WIDTH,
    HEIGHT,
    col_threshold,
    BLACK,
    bg_alpha,
    move_without_render,
    t,
    framerate,
    density,
    drag_coeff,
)
import time
from visual import button
import json
from time import sleep, time
import sqlite3

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

conn = sqlite3.connect(path)
cur = conn.cursor()

ix = 0
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            conn.close()
            sys.exit()
    surface.fill((0, 0, 0, bg_alpha))

    cur.execute("SELECT * FROM sim WHERE ix=?", (ix,))
    result = cur.fetchone()
    if not result:
        break
    ix += move_without_render
    _, x, _, m, color, x_pre = result
    x = json.loads(x)
    m = json.loads(m)
    color = json.loads(color)
    x_pre = json.loads(x_pre)

    # print(x)
    clock.tick(framerate)
    out = ""
    for j in range(len(x)):
        px, py = x[j]
        px_p, py_p = x_pre[j]

        if m[j] > 0 and x[j][0] > 0 and x[j][1] > 0:
            out += str(x[j])
            r = int((m[j] ** (1 / 3)) * density)
            pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(px - r / 2, py - r / 2, r, r))
            pygame.draw.line(surface, (255, 255, 255), (px, py), (px_p, py_p), r)
    # print(out)
    # print(i)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

conn.close()
