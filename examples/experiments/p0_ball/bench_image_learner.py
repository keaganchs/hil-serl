"""How fast HIL-SERL's own (image) learner loop runs: their RAM-insertion setup (2 wrist cameras 128x128, 19-d proprio,
6-d action, frozen ResNet-10, batch 256 half demo / half online, cta_ratio 2), random data, the learner loop of
train_rlpd.py (sample both buffers, cta_ratio - 1 critic updates + 1 full update). Gives their gradient updates per
collected transition at the actor's 10 Hz, the number the P0 state-only learner is capped to (config.py max_utd).
Result (job 982, 2026-10-30): 5.8 learner steps/s = 11.6 updates/s = 1.2 updates per transition on an RTX A5000 shard
while both GPUs ran other jobs at 93-100 % (a lower bound; their RTX 4090 alone: ~3-4x that). Run inside Slurm:
  sbatch -J hilserl_bench -t 15 --gres=shard:1 -c 2 --mem=12G --wrap '... python bench_image_learner.py'"""

import time

import gymnasium as gym
import jax
import numpy as np

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import make_sac_pixel_agent
from serl_launcher.utils.train_utils import concat_batches

KEYS, CTA, BATCH, N = ("wrist_1", "wrist_2"), 2, 256, 300
obs_space = gym.spaces.Dict(
    {"state": gym.spaces.Box(-1, 1, (1, 19), np.float32)} | {k: gym.spaces.Box(0, 255, (1, 128, 128, 3), np.uint8) for k in KEYS}
)
act_space = gym.spaces.Box(-1, 1, (6,), np.float32)
dev = jax.local_devices()[0]
bufs = []
for _ in range(2):  # online, demo
    b = MemoryEfficientReplayBufferDataStore(obs_space, act_space, capacity=20000, image_keys=list(KEYS))
    o = obs_space.sample()
    for i in range(2000):
        n = obs_space.sample()
        b.insert(dict(observations=o, actions=act_space.sample(), next_observations=n, rewards=0.0, masks=1.0, dones=i % 100 == 99))
        o = n
    bufs.append(b.get_iterator(sample_args={"batch_size": BATCH // 2, "pack_obs_and_next_obs": True}, device=dev))
agent = make_sac_pixel_agent(0, obs_space.sample(), act_space.sample(), image_keys=KEYS)
batch = lambda: concat_batches(next(bufs[0]), next(bufs[1]), axis=0)
for k in range(N + 20):
    if k == 20:
        agent = jax.block_until_ready(agent)
        t0 = time.perf_counter()
    for _ in range(CTA - 1):
        agent, _ = agent.update(batch(), networks_to_update=frozenset({"critic"}))
    agent, _ = agent.update(batch(), networks_to_update=frozenset({"critic", "actor", "temperature"}))
agent = jax.block_until_ready(agent)
hz = N / (time.perf_counter() - t0)
print(f"{jax.devices()[0].device_kind}: learner steps {hz:.1f}/s, gradient updates {CTA * hz:.1f}/s, "
      f"updates per transition at 10 Hz: {CTA * hz / 10:.1f}")
