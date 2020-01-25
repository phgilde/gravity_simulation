import numpy as np

NUM_OF_BODIES = 1000
WIDTH = 900
HEIGHT = 900

MIN_SIZE = 1
MAX_SIZE = 3

# mass of first body
m0mass = 2

n_bodies = NUM_OF_BODIES
# Velocity
V = np.random.uniform(low=-1, high=1, size=(n_bodies, 2))
# V = np.zeros(shape=(n_bodies, 2))
V[0] = 0, 0


# Position
X = np.random.uniform(low=10, high=min(WIDTH, HEIGHT), size=(n_bodies, 2))
X[0] = WIDTH / 2, HEIGHT / 2
X[1] = WIDTH / 2, HEIGHT - 100
X[2] = WIDTH / 2, HEIGHT / 3
# Mass
M = np.random.randint(MIN_SIZE, MAX_SIZE, size=n_bodies)
M[0] = m0mass

# Color
COLOR = np.full([n_bodies, 3], 255)

# LOCK is always center
DO_LOCK = False
LOCK = 0
