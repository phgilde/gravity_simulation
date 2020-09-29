import numpy as np

NUM_OF_BODIES = 2
WIDTH = 900
HEIGHT = 900

MIN_SIZE = 1
MAX_SIZE = 3

# mass of first body
m0mass = 2

n_bodies = NUM_OF_BODIES
# Velocity
# V = np.random.uniform(low=-1, high=1, size=(n_bodies, 2))
V = np.zeros(shape=(n_bodies, 2))
V[0] = 0, 0

density = 10
# Position
X = np.array([[0, 400], [500, 400]])
# Mass
M = np.array([10, 10])

# Radius
R = ((3 * M * density) / (4 * np.pi)) ** (1 / 3)

# Color
COLOR = np.full([n_bodies, 3], 255)

# LOCK is always center
DO_LOCK = False
LOCK = 0