
# precision of the simulation, lower = higher precision
t = 0.001

# simulation save rate
move_without_render = 8

# multiply particle radius by this, collide when close enough
col_threshold = 1.2


max_acc = 50


density = 1

drag_coeff = 1
# stop when reaching this number of simulation steps
max_steps = float("inf")

# save every n steps
save_steps = 2000

# if less bodies than this, the simulation will stop
min_bodies = 2

path = "D:/simulations/{}.db"
log_path = "logs/{}.csv"
do_log = False