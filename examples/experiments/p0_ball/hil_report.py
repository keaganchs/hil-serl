"""Summarise a HIL session's decision log (the client's <session>/hil.jsonl, p0_grasping deploy/hil.py) in time bins:
episodes and put-backs, interventions (the step share, the paper's per-episode rate, takeovers, their mean duration overall
and per trigger, put-backs reached while the intervener had the arm), clipped stored actions, `near` (wide guard) steps,
press peaks, late actions, the learner's action size and pad motion.
  .venv/bin/python examples/experiments/p0_ball/hil_report.py SESSION/hil.jsonl [SESSION2/hil.jsonl ...] [--bin 600]"""

import argparse
import collections
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("logs", nargs="+")
p.add_argument("--bin", type=float, default=600.0, help="[s]")
a = p.parse_args()
rows, ep0 = [], 0
for f in a.logs:  # several sessions (client restarts) in a row: episodes renumbered
    rs = [json.loads(line) for line in open(f)]
    for r in rs:
        r["ep"] += ep0
    ep0 = max([r["ep"] for r in rs] + [ep0])
    rows += rs
t0 = rows[0]["t"]
for prev, r in zip([None] + rows[:-1], rows):  # pad motion over the step [cm / 100 ms] (the pad's end-of-step positions)
    r["jump"] = 100 * float(np.linalg.norm(np.subtract(r["d"], prev["d"]))) if prev and prev["ep"] == r["ep"] else np.nan
    r["bin"] = int((r["t"] - t0) // a.bin)
ivs = collections.defaultdict(list)  # (ep, iv_k) -> rows of one takeover
for r in rows:
    if r.get("iv_k"):
        ivs[(r["ep"], r["iv_k"])].append(r)
eps = collections.defaultdict(list)
for r in rows:
    eps[r["ep"]].append(r)
bins = collections.defaultdict(list)
for r in rows:
    bins[r["bin"]].append(r)
print("min   steps eps put  iv%  iv%/ep  n_iv  dur_s  dur by trigger [s (n)]                       put@iv clip%  near%  g_p99 g_max late%  |a_ik|  pad cm/step p50/p99")
for b, rs in sorted(bins.items()):
    ends = [r for r in rs if r["term"] or r["trunc"]]
    puts = [r for r in ends if r["r"] > 0]
    tk = [v for v in ivs.values() if v[0]["bin"] == b]
    dur = [sum(r.get("iv_ticks", 5) for r in v) / 50 for v in tk]
    by = collections.defaultdict(list)
    for v, d in zip(tk, dur):
        by[v[0]["iv_why"]].append(d)
    by_s = " ".join(f"{k}:{np.mean(x):.1f}({len(x)})" for k, x in sorted(by.items(), key=lambda kv: -len(kv[1])))
    rate = [np.mean([r["iv"] for r in eps[e["ep"]]]) for e in ends]  # the paper's: intervened steps / steps, per episode
    ivr = [r for r in rs if r["iv"]]
    g = np.array([r["g_max"] for r in rs])
    ik = np.array([np.abs(r["a"][:6]).mean() for r in rs if not r["iv"]] or [np.nan])
    j = np.array([r["jump"] for r in rs if not r["iv"]] or [np.nan])
    print(f"{b * a.bin / 60:4.0f}  {len(rs):5d} {len(ends):3d} {len(puts):3d}  {100 * len(ivr) / len(rs):3.0f}  {100 * np.mean(rate or [np.nan]):5.0f}  "
          f"{len(tk):4d}  {np.mean(dur or [np.nan]):5.2f}  {by_s[:44]:44s}  {sum(r['iv'] for r in puts):5d}  "
          f"{100 * np.mean([r.get('clip', 0) > 0 for r in ivr] or [np.nan]):5.1f}  {100 * np.mean([r.get('near', False) for r in rs]):5.1f}  "
          f"{np.percentile(g, 99):5.0f} {g.max():5.0f} {100 * np.mean([r['late'] for r in rs]):5.1f}  {np.nanmedian(ik):.2f}    "
          f"{np.nanmedian(j):.1f}/{np.nanpercentile(j, 99):.1f}")
ends = collections.Counter(r["why"].split(" (")[0][:60] for r in rows if r["term"] or r["trunc"])
print("episode ends:", dict(ends))
rel = collections.Counter(r["rel"] for r in rows if r.get("rel"))
print("releases:", dict(rel), " takeovers:", len(ivs), " clipped IK elements per intervened step:",
      f"{np.mean([r.get('clip', 0) for r in rows if r['iv']] or [np.nan]):.2f} of 6")
