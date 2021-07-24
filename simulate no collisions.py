import time

import numpy as np
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
import tensorflow as tf
import pstats

from environment import V, X, M, COLOR, DO_LOCK, LOCK, WIDTH, HEIGHT
from datetime import datetime, timedelta

import json
import sqlite3
import progressbar
import sys


def main():
    now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # create database
    conn = sqlite3.connect(path.format(now))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE sim (ix INT PRIMARYKEY, x JSON, v JSON, m JSON, color JSON, x_pre JSON)"
    )
    conn.commit()

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

    sys.setrecursionlimit(1500)
    tf.compat.v1.enable_eager_execution()

    def a(x, m, n_bodies):
        x_j = tf.reshape(x, (-1, 1, 2))
        x_i = tf.reshape(x, (1, -1, 2))
        d = x_j - x_i

        a_ = tf.math.divide_no_nan(
            (tf.reshape(m, (-1, 1, 1)) * (d)),
            tf.clip_by_value(tf.reshape(
                tf.sqrt(d[:, :, 0] ** 2 + d[:, :, 1] ** 2) ** 3,
                (n_bodies, n_bodies, 1),
            ), 1, float("inf")),
        )
        return tf.reduce_sum(a_, axis=0)

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
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m = tf.convert_to_tensor(m, dtype=tf.float32)
        v = tf.convert_to_tensor(v, dtype=tf.float32)

        for steps in progressbar.progressbar(range(max_steps)):
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
                        json.dumps(x.numpy().astype(int).tolist()),
                        json.dumps(v.numpy().astype(int).tolist()),
                        json.dumps(m.numpy().astype(int).tolist()),
                        json.dumps(color.astype(int).tolist()),
                        json.dumps(x_pre.astype(int).tolist()),
                    ),
                )
            
            if do_log:
                with open(log_path.format(now), "a") as f:
                    f.write(
                        "{},{},{},{}\n".format(
                            steps, time.time() - last, time.time(), n_bodies
                        )
                    )
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
