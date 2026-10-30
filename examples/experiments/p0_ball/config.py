"""P0 rig ball task (single left arm, ball in a dish on the load-cell scale; no camera): state-only RLPD.
Observation: the base ball policy's 112-d actor input, normalised with its frozen normaliser (obs_norm_933.npz).
Action: our 9-d interface (6 IK deltas + grasp command [on, force, aperture]) at 10 Hz; the client repeats it for 5 ticks.
Reward: sparse, the client's Attempts outcomes on the scale (put-back ok = 1, terminal; a failed attempt: terminal 0)."""

import os

import numpy as np

from experiments.config import DefaultTrainingConfig
from experiments.p0_ball.socket_env import P0SocketEnv

HERE = os.path.dirname(os.path.abspath(__file__))


class TrainConfig(DefaultTrainingConfig):
    setup_mode = "single-arm-state"
    image_keys = []
    proprio_keys = ["state"]
    max_traj_length = 200  # 20 s at 10 Hz (a 933 cycle takes ~4.5 s; Attempts' lift / put time-outs are 6 / 12 s)
    discount = 0.97
    batch_size = 256
    cta_ratio = 2
    training_starts = 100
    steps_per_update = 500  # publish every ~2 s: the state-only learner runs ~230 steps/s on a shard (the image one ~10/s)
    buffer_period = 1000
    checkpoint_period = 5000
    replay_buffer_capacity = 200000
    actor_port = 5590
    wandb_project = "p0_manipulation"
    wandb_overrides = {"entity": "pink_robotics", "group": "hil_serl"}
    obs_norm = os.path.join(HERE, "obs_norm_933.npz")

    def obs_stats(self):
        n = np.load(self.obs_norm)
        return n["mean"], n["std"]

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # no TimeLimit: the client ends every episode itself (it truncates at max_traj_length), else the two would desync
        return P0SocketEnv(self.actor_port, fake_env=fake_env)
