"""Plumbing test without the robot: plays our client's side of socket_env.py at 10 Hz with random observations and
measures the actor's reply latency (obs sent -> action received). --write-demos PATH: a small random demo file instead.
  .venv/bin/python examples/experiments/p0_ball/fake_client.py --steps 600"""

import argparse
import pickle
import time

import numpy as np
import zmq

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=5590)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--episode", type=int, default=50)
p.add_argument("--write-demos")
a = p.parse_args()
rng = np.random.default_rng(0)
obs = lambda: rng.normal(size=112).astype(np.float32)

if a.write_demos:
    o, demos = obs(), []
    for i in range(200):
        n = obs()
        done = i % a.episode == a.episode - 1
        demos.append(dict(observations={"state": o[None]}, actions=rng.uniform(-1, 1, 9).astype(np.float32),
                          next_observations={"state": n[None]}, rewards=float(done), masks=1.0 - done, dones=done))
        o = n
    pickle.dump(demos, open(a.write_demos, "wb"))
    raise SystemExit

s = zmq.Context().socket(zmq.PAIR)
s.connect(f"tcp://127.0.0.1:{a.port}")
lat, t_ep = [], 0
s.send(pickle.dumps({"obs": obs(), "first": True}))
for k in range(a.steps):
    t = time.perf_counter()
    reply = pickle.loads(s.recv())
    lat.append(time.perf_counter() - t)
    assert reply["action"].shape == (9,)
    time.sleep(0.1)
    t_ep += 1
    end = t_ep == a.episode
    iv = bool(rng.random() < 0.2)
    msg = {"obs": obs(), "first": False, "reward": float(end), "terminated": end, "truncated": False,
           "intervened": iv, "action": rng.uniform(-1, 1, 9).astype(np.float32), "info": {}}
    s.send(pickle.dumps(msg))
    if end:
        t_ep = 0
        s.send(pickle.dumps({"obs": obs(), "first": True}))
big = [(k, round(1e3 * v)) for k, v in enumerate(lat) if v > 0.02]
print("slow replies (step, ms):", big[:40])
lat = 1e3 * np.array(lat[5:])
print(f"actor reply latency ms: median {np.median(lat):.1f} p99 {np.percentile(lat, 99):.1f} max {lat.max():.1f} (n={len(lat)})")
