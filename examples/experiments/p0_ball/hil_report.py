"""Summarise a HIL session's decision log (the client's <session>/hil.jsonl, p0_grasping deploy/hil.py) in time bins:
episodes and put-backs, the intervened share of steps and the takeover triggers, press peaks, clamps, late actions.
  .venv/bin/python examples/experiments/p0_ball/hil_report.py SESSION/hil.jsonl [--bin 300]"""

import argparse
import collections
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--bin", type=float, default=300.0, help="[s]")
a = p.parse_args()
rows = [json.loads(line) for line in open(a.log)]
t0 = rows[0]["t"]
for prev, r in zip([None] + rows[:-1], rows):  # pad motion over the step [cm / 100 ms] (the pad's end-of-step positions)
    r["jump"] = 100 * float(np.linalg.norm(np.subtract(r["d"], prev["d"]))) if prev and prev["ep"] == r["ep"] else np.nan
bins = collections.defaultdict(list)
for r in rows:
    bins[int((r["t"] - t0) // a.bin)].append(r)
print("min    steps  eps  put  put/ep  iv%   iv steps by trigger                     g_p50  g_p99  g_max  clamp%  late%  |a_ik|p50  pad cm/step p50/p99 (learner)")
for b, rs in sorted(bins.items()):
    ends = [r for r in rs if r["term"] or r["trunc"]]
    puts = sum(r["r"] > 0 for r in rs)
    g = np.array([r["g_max"] for r in rs])
    why = collections.Counter(w for r in rs if r["iv"] for w in r["iv_why"].split("+"))
    ik = np.array([np.abs(r["a"][:6]).mean() for r in rs if not r["iv"]] or [np.nan])
    print(f"{b * a.bin / 60:5.0f}  {len(rs):5d}  {len(ends):3d}  {puts:3d}  {puts / max(len(ends), 1):6.2f}  "
          f"{100 * np.mean([r['iv'] for r in rs]):4.0f}  {dict(why)!s:38.38}  {np.median(g):5.0f}  {np.percentile(g, 99):5.0f}  "
          f"{g.max():5.0f}  {100 * np.mean([r['clamp'] for r in rs]):6.1f}  {100 * np.mean([r['late'] for r in rs]):5.1f}  {np.nanmedian(ik):.2f}  "
          f"{np.nanmedian(j := np.array([r['jump'] for r in rs if not r['iv']] or [np.nan])):.1f}/{np.nanpercentile(j, 99):.1f}")
ends = collections.Counter(r["why"] for r in rows if r["term"] or r["trunc"])
print("episode ends:", dict(ends))
