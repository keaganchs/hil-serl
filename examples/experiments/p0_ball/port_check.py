"""P0 setup check: JAX devices, then a ZMQ REP echo bound to one interface (the Tailscale IP) for --seconds.
From the robot computer: .venv/bin/python examples/experiments/p0_ball/port_check.py --connect <ip> (prints the round trip)."""
import argparse
import time

import zmq

p = argparse.ArgumentParser()
p.add_argument("--bind", help="IP to listen on (Sinabox's Tailscale IP)")
p.add_argument("--connect", help="IP to ping")
p.add_argument("--port", type=int, default=5588)
p.add_argument("--seconds", type=float, default=600)
a = p.parse_args()
s = zmq.Context().socket(zmq.REP if a.bind else zmq.REQ)
if a.bind:
    import jax

    print("jax", jax.__version__, jax.devices(), flush=True)
    s.bind(f"tcp://{a.bind}:{a.port}")
    print(f"listening on {a.bind}:{a.port}", flush=True)
    end = time.time() + a.seconds
    while time.time() < end:
        if s.poll(1000):
            s.send(s.recv())
            print("pinged", flush=True)
else:
    s.setsockopt(zmq.RCVTIMEO, 5000)
    s.connect(f"tcp://{a.connect}:{a.port}")
    for _ in range(5):
        t = time.perf_counter()
        s.send(b"ping")
        s.recv()
        print(f"round trip {1e3 * (time.perf_counter() - t):.1f} ms")
