import numpy as np

NUM_OF_BODIES = 100
WIDTH = 800
HEIGHT = 800

MIN_SIZE = 5
MAX_SIZE = 6


m0mass = 5

n_bodies = NUM_OF_BODIES
# Velocity
V = np.random.uniform(low=-1, high=1, size=(n_bodies, 2))
# V = np.zeros(shape=(n_bodies, 2))
V[0] = 0, 0


# Position
X = np.random.uniform(low=10, high=WIDTH - 10, size=(n_bodies, 2))
X[0] = WIDTH / 2, HEIGHT / 2
X[1] = WIDTH / 2, HEIGHT - 100
X[2] = WIDTH / 2, HEIGHT / 3
# Mass
M = np.random.randint(MIN_SIZE, MAX_SIZE, size=n_bodies)
M[0] = m0mass

# Color
COLOR = np.full([n_bodies, 3], 255)

DO_LOCK = False
LOCK = 0
