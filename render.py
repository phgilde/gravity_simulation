import pygame
import numpy as np
import sys
from cnfrender import (
    WIDTH,
    HEIGHT,
    bg_alpha,
    move_without_render,
    framerate,
    density,
    batch,
)
import json
import sqlite3
import threading
import queue
import time

BLACK = (0, 0, 0)

pygame.init()
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = input("Path: ")
# time.sleep(2)

pause = False
screen = pygame.display.set_mode((WIDTH, HEIGHT))
surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
surface.convert()
screen.fill(BLACK)
np.set_printoptions(suppress=True)
clock = pygame.time.Clock()

for i in range(0 * 60):
    clock.tick(60)
    pygame.display.update()


def load(ix):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    result = cur.execute(
        "SELECT x, m, color, x_pre FROM (SELECT x, m, color, x_pre FROM sim WHERE (ix, 0) in (SELECT ix, ix % ?)) LIMIT ? OFFSET ?",
        (move_without_render, batch, ix),
    )
    res = result.fetchall()
    conn.close()
    return res


def enthread(target, args):
    q = queue.Queue()

    def wrapper():
        q.put(target(*args))

    t = threading.Thread(target=wrapper)
    t.start()
    return q


ix = 0
lst = load(ix)
ix += batch
q1 = enthread(load, (ix,))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    surface.fill((0, 0, 0, bg_alpha))

    if len(lst) == batch:
        lst += q1.get()
        ix += batch
        print("{}        ".format(ix * move_without_render))
        q1 = enthread(load, (ix,))
    try:
        result = lst.pop(0)
    except IndexError:
        break

    x, m, color, x_pre = result
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
            pygame.draw.circle(surface, (255, 255, 255), (px, py), int(r / 2))
            pygame.draw.line(surface, (255, 255, 255), (px, py), (px_p, py_p), r)
    # print(out)
    # print(i)
    screen.blit(surface, (0, 0))
    pygame.display.flip()
