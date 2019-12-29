import sys
import time

import numpy as np
import pygame
from visual import button
from cnf import WIDTH, HEIGHT, col_threshold, BLACK, bg_alpha, move_without_render, t, framerate, density, drag_coeff
import cProfile

import pstats

from environment import V, X, M, COLOR, DO_LOCK, LOCK


def main():
    lock = LOCK
    n_bodies = M.shape[0]

    # Velocity
    v = np.copy(V)

    # Position
    x = np.copy(X)

    # Mass
    m = np.copy(M)

    # Color
    color = np.copy(COLOR)
    cp = np.copy
    """
    def a(x):
        a_ = np.ndarray((n_bodies, 2))
        for i in range(n_bodies):
            d = x - x[i]
            a_i = (m.reshape(-1, 1) * (x - x[i]))\
                    /\
                        (np.sqrt(d[:, 0]**2 + d[:, 1]**2) ** 3).reshape(-1, 1)
            
            a_i[i] = 0
            a_[i] = np.sum(a_i, axis=0)
                    # print("Acceleration", a_[i], m[i])

        a_[(a_[:,0]>max_acc) | (a_[:, 1]>max_acc)] = 0,0
        return a_
    """

    def a(x):
        x_j = x.reshape(-1, 1, 2)
        x_i = x.reshape(1, -1, 2)
        d = x_j - x_i

        a_ = (m.reshape(-1, 1, 1) * (d)) / (np.sqrt(d[:, :, 0] ** 2 + d[:, :, 1] ** 2) ** 3).reshape(
            n_bodies, n_bodies, 1
        )
        r = np.arange(a_.shape[0])
        a_[r, r] = 0, 0
        return np.sum(a_, axis=0)

    # When two objects collide, their force and weight adds up
    def collision(m, p, v, n, lock):
        for i in range(n):
            if m[i] > 0:
                diff = p - p[i]
                r = m[i] ** (1 / 3)
                distance = np.linalg.norm(diff, axis=1)
                collisions = (distance < (r * col_threshold)) & (m > 0)
                collisions[i] = False
                m_col = m[collisions]
                v_col = v[collisions]
                p_col = p[collisions]

                m[collisions] = 0

                m_i_pre = m[i]

                m[i] += np.sum(m_col)

                v[i] *= m_i_pre

                v[i] += np.sum(v_col * m_col.reshape(-1, 1), axis=0)

                v[i] /= m[i]

                p[i] *= m_i_pre
                p[i] += np.sum(p_col * m_col.reshape(-1, 1), axis=0)

                p[i] /= m[i]

                if lock in collisions:
                    lock = i

        return m, p, v, lock

    def sim_runge_kutter(m, x, v, step):
        k0 = step * v
        l0 = step * a(x)

        k1 = step * (v + l0 * 0.5)
        l1 = step * a(x + k0 * 0.5)

        k2 = step * (v + l1 * 0.5)
        l2 = step * a(x + l1 * 0.5)

        k3 = step * (v + l2)
        l3 = step * a(x + k2)
        x = x + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3)

        v = v + (1.0 / 6) * (l0 + 2 * l1 + 2 * l2 + l3)

        return x, v

    def kill_empty(m, x, v, n):
        empty = m == 0
        m = m[~empty]
        x = x[~empty]
        v = v[~empty]
        n = np.sum(~empty)
        return m, x, v, n

    pygame.init()
    pause = False
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    surface.convert()
    screen.fill(BLACK)
    np.set_printoptions(suppress=True)
    clock = pygame.time.Clock()

    while True:
        clock.tick(framerate)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if not pause:
            surface.fill((0, 0, 0, bg_alpha))
            # collide objects
            m, x, v, _ = collision(m, x, v, n_bodies, lock)
            # remove mass=0 objects
            m, x, v, n_bodies = kill_empty(m, x, v, n_bodies)

            x_pre = cp(x)
            # simulate
            for i in range(move_without_render):
                x, v = sim_runge_kutter(m, x, v, t)
            v = v * drag_coeff
            # change position of objects so locked object is always in the middle of the screen
            if DO_LOCK:
                x = x - x[lock] + (WIDTH / 2, HEIGHT / 2)

            # render objects
            for i in range(x.shape[0]):
                px, py = x[i]
                px_p, py_p = x_pre[i]
                if m[i] > 0 and x[i, 0] > 0 and x[i, 1] > 0:
                    r = int((m[i] ** (1 / 3)) * density)
                    pygame.draw.rect(surface, color[i], pygame.Rect(px - r / 2, py - r / 2, r, r))
                    pygame.draw.line(surface, color[i], (px, py), (px_p, py_p), r)

        # pause button
        if button(surface, "PAUSE", 5, 5, 80, 20, (50, 50, 50, 100), (100, 100, 100, 100)):
            print("pause", not pause)
            pause = not pause
            time.sleep(0.1)

        screen.blit(surface, (0, 0))
        pygame.display.update()


cProfile.run("main()", "restats")

p = pstats.Stats("restats")
p.strip_dirs().sort_stats("time").print_stats(10)
