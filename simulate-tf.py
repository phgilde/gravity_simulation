import time

import numpy as np
import tensorflow as tf
from cnf import (
    max_steps,
    col_threshold,
    move_without_render,
    t,
    drag_coeff,
    min_bodies,
    save_steps,
    path,
    log_path,
    do_log,
    theta,
    use_barnes_hut,
    density,
)
import cProfile

import pstats

from environment import V, X, M, R, COLOR, DO_LOCK, LOCK, WIDTH, HEIGHT
from datetime import datetime, timedelta

import json
import sqlite3
import numba
import sys

tf.compat.v1.enable_eager_execution()


def main():
    now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # create database
    conn = sqlite3.connect(path.format(now))
    cur = conn.cursor()
    cur.execute("CREATE TABLE sim (ix INT PRIMARYKEY, x JSON, v JSON, m JSON, color JSON, x_pre JSON)")
    conn.commit()

    lock = LOCK
    n_bodies = M.shape[0]

    # Velocity
    v = tf.convert_to_tensor(V, dtype=tf.float32)

    # Position
    x = tf.convert_to_tensor(X, dtype=tf.float32)

    # Mass
    m = tf.convert_to_tensor(M, dtype=tf.float32)

    r = tf.convert_to_tensor(R, dtype=tf.float32)

    # Color
    color = np.copy(COLOR)
    cp = np.copy

    sys.setrecursionlimit(1500)

    def a(x, m, n_bodies):
        distance = tf.transpose(x) - x
        l2distance = tf.norm(distance, ord=2, axis=2)

        g = tf.transpose(m) / (l2distance ** 2)
        k = tf.transpose(m) / (((l2distance / (r + tf.transpose(r))) ** 3) * l2distance ** 2)
        a = tf.reduce_sum((g - k) * (distance / l2distance), axis=1)

        return a

    def sim_runge_kutter(m, x, v, step, n_bodies):
        k0 = step * v
        l0 = step * a(x, m, n_bodies)

        k1 = step * (v + l0 * 0.5)
        l1 = step * a(x + k0 * 0.5, m, n_bodies)

        k2 = step * (v + l1 * 0.5)
        l2 = step * a(x + l1 * 0.5, m, n_bodies)

        k3 = step * (v + l2)
        l3 = step * a(x + k2, m, n_bodies)
        x = x + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3)

        v = v + (1.0 / 6) * (l0 + 2 * l1 + 2 * l2 + l3)

        return x, v

    np.set_printoptions(suppress=True)

    start = time.time()
    last = start
    steps = 0
    try:
        while (steps < max_steps) and (n_bodies >= min_bodies):

            x_pre = cp(x)
            # simulate
            x, v = sim_runge_kutter(m, x, v, t, n_bodies)
            v = v * drag_coeff

            # change position of objects so locked object is always in the middle of the screen
            if DO_LOCK:
                x = x - x[lock] + (WIDTH / 2, HEIGHT / 2)
            # put state into database
            if steps % move_without_render == 0:
                cur.execute(
                    "INSERT INTO sim VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        steps,
                        json.dumps(x.astype(int).tolist()),
                        json.dumps(v.astype(int).tolist()),
                        json.dumps(m.astype(int).tolist()),
                        json.dumps(color.astype(int).tolist()),
                        json.dumps(x_pre.astype(int).tolist()),
                    ),
                )
            print(
                "{:>10} {} {} {}           ".format(
                    steps, timedelta(seconds=time.time() - last), timedelta(seconds=time.time() - start), n_bodies
                ),
                end="\r",
            )
            if do_log:
                with open(log_path.format(now), "a") as f:
                    f.write("{},{},{},{}\n".format(steps, time.time() - last, time.time(), n_bodies))
            last = time.time()
            steps += 1
            if steps % save_steps == 0:
                print("\nAutosaving...")
                conn.commit()
                print("Done!")

            # pause button
    finally:
        print("Saving...")
        conn.commit()
        print("Done!")
        conn.close()


cProfile.run("main()", "restats")

p = pstats.Stats("restats")
p.strip_dirs().sort_stats("time").print_stats(10)
