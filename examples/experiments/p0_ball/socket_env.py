"""The P0 rig as a gym env for train_rlpd.py's actor loop: the robot side is our client (p0_grasping deploy client,
HIL mode), which runs the arm at 50 Hz and talks to this env over local UDP (one pickled dict per datagram; stdlib on
both sides, so the client's venv needs nothing new) at the 10 Hz decision rate.

client -> env, one message per decision step (pickled dict):
  obs [112] f32         the ball policy's actor input at the decision tick (raw)
  first                 True: a new episode starts at this obs (reward/terminated/... are then ignored)
  reward, terminated, truncated, info   the outcome of the step just finished
  intervened, action    the step's stored action when the intervener had the arm (else the policy's own action ran)
env -> client: {"action": [9] f32, "step": n}, the action for the decision just received.
"""

import pickle

import socket

import gymnasium as gym
import numpy as np

OBS_DIM, ACT_DIM = 112, 9


class P0SocketEnv(gym.Env):
    def __init__(self, port=5590, fake_env=False):
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-np.inf, np.inf, (1, OBS_DIM), np.float32)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.sock, self.peer = None, None
        if not fake_env:  # the learner only needs the spaces
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("127.0.0.1", port))  # local only: the client runs on this computer
        self.n = 0

    def _recv(self):
        data, self.peer = self.sock.recvfrom(65536)  # replies go to the client that sent last
        return pickle.loads(data)

    @staticmethod
    def _obs(m):
        return {"state": np.asarray(m["obs"], np.float32).reshape(1, OBS_DIM)}

    def reset(self, **kwargs):
        while not (m := self._recv())["first"]:  # a message of the last episode (sent before it ended) is dropped
            pass
        return self._obs(m), {}

    def step(self, action):
        self.n += 1
        self.sock.sendto(pickle.dumps({"action": np.asarray(action, np.float32), "step": self.n}), self.peer)
        m = self._recv()
        assert not m["first"], "client started a new episode without ending the last one"
        info = dict(m.get("info", {}))
        if m["intervened"]:
            info["intervene_action"] = np.asarray(m["action"], np.float32)
        return self._obs(m), float(m["reward"]), bool(m["terminated"]), bool(m["truncated"]), info
