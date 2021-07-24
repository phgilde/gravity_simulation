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
import numba
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
    # @lru_cache(maxsize=256)
    def quadtree(m, x):
        if m.shape[0] == 1:
            return {"center": x[0], "mass": m[0], "width": 0}
        if m.shape[0] == 0:
            return {"mass": 0, "width": 0}
        min_x = np.min(x[:, 0])
        min_y = np.min(x[:, 1])
        max_x = np.max(x[:, 0])
        max_y = np.max(x[:, 1])
        # print(min_x, max_x)
        div_x = (min_x + max_x) / 2
        div_y = (min_y + max_y) / 2

        quadrant1_bool = (x[:, 0] < div_x) & (x[:, 1] < div_y)
        quadrant2_bool = (x[:, 0] >= div_x) & (x[:, 1] < div_y)
        quadrant3_bool = (x[:, 0] < div_x) & (x[:, 1] >= div_y)
        quadrant4_bool = (x[:, 0] >= div_x) & (x[:, 1] >= div_y)

        quadrant1 = quadtree(m[quadrant1_bool], x[quadrant1_bool])
        quadrant2 = quadtree(m[quadrant2_bool], x[quadrant2_bool])
        quadrant3 = quadtree(m[quadrant3_bool], x[quadrant3_bool])
        quadrant4 = quadtree(m[quadrant4_bool], x[quadrant4_bool])

        return {
            "center": np.sum(x * m.reshape(-1, 1), axis=0) / np.sum(m),
            "mass": np.sum(m),
            "sub": [quadrant1, quadrant2, quadrant3, quadrant4],
            "width": (((max_x - min_x) ** 2) + ((max_y - min_y) ** 2)) ** 0.5,
        }

    if use_barnes_hut:

        def a(x, m, n_bodies, theta=theta):
            a_ = []
            # generate quadtree
            tree = quadtree(m, x)
            for i in range(len(m)):
                to_check = [tree]
                x_cur, m_cur = [], []
                while len(to_check):
                    curr_quad = to_check.pop()
                    # empty external quadrant
                    if curr_quad["mass"] == 0:
                        continue
                    # not empty external quadrant
                    if curr_quad["width"] == 0:
                        # external quadrant is not current particle
                        if (curr_quad["center"] != x[i]).any():
                            x_cur.append(curr_quad["center"])
                            m_cur.append(curr_quad["mass"])
                        continue
                    # width/distance from center of gravity < theta
                    if (
                        curr_quad["width"]
                        / np.sqrt(
                            (x[i, 0] - curr_quad["center"][0]) ** 2
                            + (x[i, 1] - curr_quad["center"][0]) ** 2
                        )
                        < theta
                    ):
                        x_cur.append(curr_quad["center"])
                        m_cur.append(curr_quad["mass"])
                        continue
                    # open quadrant
                    to_check += curr_quad["sub"]
                m_cur = np.array(m_cur)
                x_cur = np.array(x_cur)

                # calculate gravitational forces and add them to the corresponding array
                a_.append(
                    np.sum(
                        (m_cur.reshape(-1, 1) * (x_cur - x[i]))
                        / np.sqrt(
                            (x_cur[:, 0] - x[i, 0]) ** 2
                            + (x_cur[:, 1] - x[i, 1]) ** 2
                        ).reshape(-1, 1)
                        ** 3,
                        axis=0,
                    )
                )

            return np.array(a_)

    else:

        def a(x, m, n_bodies):
            x_j = tf.reshape(x, (-1, 1, 2))
            x_i = tf.reshape(x, (1, -1, 2))
            d = x_j - x_i

            a_ = tf.math.divide_no_nan(
                (tf.reshape(m, (-1, 1, 1)) * (d)),
                tf.reshape(
                    tf.sqrt(d[:, :, 0] ** 2 + d[:, :, 1] ** 2) ** 3,
                    (n_bodies, n_bodies, 1),
                ),
            ) - tf.math.divide_no_nan(
                (tf.reshape(m, (-1, 1, 1)) * (d)),
                tf.reshape(
                    tf.sqrt(d[:, :, 0] ** 2 + d[:, :, 1] ** 2) ** 5,
                    (n_bodies, n_bodies, 1),
                ),
            )
            return tf.reduce_sum(a_, axis=0)

    # When two objects collide, their force and weight adds up

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

    def kill_empty(m, x, v, n):
        empty = m == 0
        m = m[~empty]
        x = x[~empty]
        v = v[~empty]
        n = np.sum(~empty)
        return m, x, v, n

    np.set_printoptions(suppress=True)

    start = time.time()
    last = start
    steps = 0
    try:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        m = tf.convert_to_tensor(m, dtype=tf.float32)
        v = tf.convert_to_tensor(v, dtype=tf.float32)

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
                        json.dumps(x.numpy().astype(int).tolist()),
                        json.dumps(v.numpy().astype(int).tolist()),
                        json.dumps(m.numpy().astype(int).tolist()),
                        json.dumps(color.astype(int).tolist()),
                        json.dumps(x_pre.astype(int).tolist()),
                    ),
                )
            print(
                "{:>10} {} {} {}           ".format(
                    steps,
                    timedelta(seconds=time.time() - last),
                    timedelta(seconds=time.time() - start),
                    n_bodies,
                ),
                end="\r",
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
