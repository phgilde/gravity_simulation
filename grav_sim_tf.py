import tensorflow as tf
from cnf import *
import numpy as np
import pygame

n_bodies_range = range(NUM_OF_BODIES)

pygame.init()
size = WIDTH, HEIGHT
pause = False
screen = pygame.display.set_mode((WIDTH, HEIGHT))
surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
surface.convert()
font = pygame.font.SysFont('Arial', 16)
screen.fill(BLACK)
np.set_printoptions(suppress=True)

def acceleration(x, m, n, z):
    with tf.name_scope('Acceleration'):
        x_j = tf.reshape(x, (-1, 1, 2))
        x_i = tf.reshape(x, (1, -1, 2))
        d = x_j - x_i
        a_ = (tf.reshape(m, (-1, 1, 1)) * d) \
            /\
                 tf.reshape((tf.math.sqrt(d[:, :, 0]**2 + d[:, :, 1]**2) ** 3), (n, n, 1))
        a_ = tf.transpose(tf.matrix_set_diag(tf.transpose(a_,(2, 0, 1)), z), (1, 2, 0))
        return tf.reduce_sum(a_, axis=0)

# Velocity
# v = np.random.uniform(low=-1, high=1,size=(NUM_OF_BODIES, 2))
v = np.zeros(shape=(NUM_OF_BODIES, 2))
v[0] = 0, 0
v = tf.Variable(v, dtype=tf.float64)

# Position
x = np.random.uniform(low=10, high=WIDTH-10,size=(NUM_OF_BODIES, 2))
x[0] = WIDTH/2, HEIGHT/2
x = tf.Variable(x, dtype=tf.float64)

# Mass
m = np.random.randint(MIN_SIZE,MAX_SIZE,size=NUM_OF_BODIES)
m[0] = m0mass
m = tf.Variable(m, name="Mass", dtype=tf.float64)

# Color
color = np.random.randint(0, 255, size=(NUM_OF_BODIES, 3))

z = tf.zeros((2, NUM_OF_BODIES), dtype=tf.float64)

k0 = t * v
l0 = t * acceleration(x, m, NUM_OF_BODIES, z)

k1 = t * (v + l0 * 0.5)
l1 = t * acceleration(x + k0 * 0.5, m, NUM_OF_BODIES, z)

k2 = t * (v + l1 * 0.5)
l2 = t * acceleration(x + l1 * 0.5, m, NUM_OF_BODIES, z)

k3 = t * (v + l2)
l3 = t * acceleration(x + k2, m, NUM_OF_BODIES, z)

new_x = x + (1 / 6) * (k0 + 2*k1 + 2*k2 + k3)
new_v = v + (1 / 6) * (l0 + 2*l1 + 2*l2 + l3)

update_x = tf.assign(x, new_x)
update_v = tf.assign(v, new_v)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    x_prev = x.eval()
    while True:
        # print(".")
        for i in range(move_without_render):
            sess.run((update_x, update_v))
        x_render = x.eval()
        m_render = m.eval()
        
        for i in range(NUM_OF_BODIES):
            if m_render[i] > 0:
                px, py = x_render[i]
                px_p, py_p = x_prev[i]
                r = int(m_render[i] ** (1/3))
                pygame.draw.rect(surface, color[i], pygame.Rect(px-r/2, py-r/2, r,r))
                # pygame.draw.line(surface, color[i], (px, py), (px_p, py_p), r)
        screen.blit(surface, (0, 0))
        pygame.display.update()