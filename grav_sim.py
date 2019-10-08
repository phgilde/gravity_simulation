import sys
import time

import numpy as np
import pygame
from visual import *
from cnf import *
import tensorflow as tf
autograph = tf.contrib.autograph

def main():

    # Velocity
    # v = np.random.uniform(low=-1, high=1,size=(NUM_OF_BODIES, 2))
    v = np.zeros(shape=(NUM_OF_BODIES, 2))
    v[0] = 0, 0

    # Position
    x = np.random.uniform(low=10, high=WIDTH-10,size=(NUM_OF_BODIES, 2))
    x[0] = WIDTH/2, HEIGHT/2

    # Mass
    m = np.random.randint(MIN_SIZE,MAX_SIZE,size=NUM_OF_BODIES)
    m[0] = m0mass
    
    # Color
    color = np.random.randint(0, 255, size=(NUM_OF_BODIES, 3))
    cp = np.copy
    '''
    def a(x):
        a_ = np.ndarray((NUM_OF_BODIES, 2))
        for i in range(NUM_OF_BODIES):
            d = x - x[i]
            a_i = (m.reshape(-1, 1) * (x - x[i]))\
                    /\
                        (np.sqrt(d[:, 0]**2 + d[:, 1]**2) ** 3).reshape(-1, 1)
            
            a_i[i] = 0
            a_[i] = np.sum(a_i, axis=0)
                    # print("Acceleration", a_[i], m[i])

        a_[(a_[:,0]>max_acc) | (a_[:, 1]>max_acc)] = 0,0
        return a_
    '''
    def a(x):
        x_j = x.reshape(-1, 1, 2)
        x_i = x.reshape(1, -1, 2)
        d = x_j - x_i

        a_ = (m.reshape(-1, 1, 1) * (d))\
                    /\
                        (np.sqrt(d[:, :, 0]**2 + d[:, :, 1]**2) ** 3).reshape(NUM_OF_BODIES, NUM_OF_BODIES, 1)
        r = np.arange(a_.shape[0])
        a_[r, r] = 0, 0
        return np.sum(a_, axis=0)
    # When two objects collide, their force and weight adds up
    def collision(m, p, v, n):
        for i in range(n):
            if m[i] > 0:
                diff = (p - p[i])
                r = m[i] ** (1/3)
                distance = np.linalg.norm(diff, axis=1)
                collisions = (distance < (r * col_threshold)) & (m>0)
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

        return m, p, v



    pygame.init()
    size = WIDTH, HEIGHT
    pause = False
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    surface.convert()
    font = pygame.font.SysFont('Arial', 16)
    screen.fill(BLACK)
    np.set_printoptions(suppress=True)

    while True:
        surface.fill((0, 0, 0, bg_alpha))
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()


        if not pause:
            m, x, v = collision(m, x, v, NUM_OF_BODIES)
            x_pre = cp(x)
            for i in range(move_without_render):
                # m, x, v = collision(m, x, v)
                
                in_t = time.time()

                k0 = t * v
                l0 = t * a(x)

                k1 = t * (v + l0 * 0.5)
                l1 = t * a(x + k0 * 0.5)

                k2 = t * (v + l1 * 0.5)
                l2 = t * a(x + l1 * 0.5)

                k3 = t * (v + l2)
                l3 = t * a(x + k2)
                x = (x + (1 / 6) * (k0 + 2*k1 + 2*k2 + k3))

                v = v + (1. / 6) * (l0 + 2*l1 + 2*l2 + l3)
                # print(v[0] * m[0], v[1] * m[1])
            if lock0:
                x = x - x[0] + (WIDTH/2, HEIGHT/2)
            
            for i in range(x.shape[0]):
                px, py = x[i]
                px_p, py_p = x_pre[i]
                if m[i] > 0 and x[i, 0] > 0 and x[i, 1] > 0:
                    r = int(m[i] ** (1/3))
                    pygame.draw.rect(surface, color[i], pygame.Rect(px-r/2, py-r/2, r,r))
                    pygame.draw.line(surface, color[i], (px, py), (px_p, py_p), r)
            
        if button(surface,"PAUSE", 5, 5, 80, 20, (50, 50, 50, 100), (100, 100, 100, 100)):
            print("pause", not pause)
            pause = not pause
            time.sleep(0.1)
        
        screen.blit(surface, (0, 0))
        pygame.display.update()

import cProfile
cProfile.run('main()', 'restats')

import pstats
p = pstats.Stats('restats')
p.strip_dirs().sort_stats("time").print_stats(10)