"""Our rig / fake-bridge transition chunks (p0_grasping deploy/transitions.py, 50 Hz rows) -> HIL-SERL demo transitions
at 10 Hz (the pickle list record_demos.py writes). Same rules as the client's HIL mode (p0_grasping deploy/hil.py):
a decision = 5 consecutive ball ticks, its action = the mean of their executed actions, reward 1 and terminal at a low
put-back, terminal 0 at a failed attempt, truncated at 200 decisions or where the ball mode ended. Like record_demos.py,
only episodes that end in a put-back are kept.
  .venv/bin/python examples/experiments/p0_ball/convert_chunks.py OUT.pkl SESSION_DIR [SESSION_DIR ...] [--episodes 20]
--episodes: keep the first N put-back episodes in total (record_demos.py's default is 20 demos)."""

import argparse
import glob
import os
import pickle

import numpy as np

REPEAT, MAX_STEPS = 5, 200
OK = 0  # attempts.REASONS[0]


def rows(session):
    fs = sorted(glob.glob(os.path.join(session, "transitions", "chunk_*.npz")))
    zs = [np.load(f, allow_pickle=True) for f in fs]  # our own files (string columns)
    keys = ("obs", "next_obs", "action", "put", "reason", "tick", "done", "excluded")
    return {k: np.concatenate([z[k] for z in zs]) for k in keys} if zs else None


def episodes(r):
    """Yield (start, end) row ranges of HIL episodes: split at ball-mode gaps (tick jumps / done rows) and attempt ends."""
    n, s = len(r["tick"]), 0
    for i in range(n):
        gap = i + 1 == n or r["tick"][i + 1] != r["tick"][i] + 1 or r["done"][i]
        if gap or r["put"][i] or r["reason"][i] >= 0:
            yield s, i + 1
            s = i + 1


def convert(r, limit=None):
    out, n_ep, n_ok = [], 0, 0
    for s, e in episodes(r):
        n_ep += 1
        ok = bool(r["put"][e - 1])
        if not ok or (r["excluded"][s:e] != 0).any():  # record_demos.py keeps successes only; no HOLD-substituted actions
            continue
        if limit is not None and n_ok >= limit:
            break
        n_ok += 1
        idx = list(range(s, e, REPEAT))[:MAX_STEPS]
        for j, i in enumerate(idx):
            last = min(i + REPEAT, e)
            final = last == e
            nxt = r["next_obs"][e - 1] if final else r["obs"][last]
            out.append(dict(
                observations={"state": r["obs"][i][None].astype(np.float32)},
                actions=r["action"][i:last].mean(0).astype(np.float32),
                next_observations={"state": nxt[None].astype(np.float32)},
                rewards=float(final),
                masks=0.0 if final else 1.0,
                dones=final,
            ))
    return out, n_ep, n_ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--episodes", type=int)
    a = p.parse_args()
    demos, left = [], a.episodes
    for sess in a.sessions:
        r = rows(sess)
        if r is None or left == 0:
            continue
        d, n_ep, n_ok = convert(r, left)
        demos += d
        left = None if left is None else left - n_ok
        print(f"{sess}: {n_ep} episodes, {n_ok} put-backs kept, {len(d)} transitions")
    pickle.dump(demos, open(a.out, "wb"))
    print(f"{len(demos)} transitions -> {a.out}")
