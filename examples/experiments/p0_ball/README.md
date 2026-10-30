# HIL-SERL on the P0 rig ball task

State-only RLPD (their SAC, learner and actor loops) on the P0 left arm: lift the ball off the load-cell scale, hold,
put it back. No camera and no human: our ball policy (933 + grasp 844) is the intervener, triggered by the scale and the
pad position, and the reward is the scale's put-back rule.

| piece | where |
|---|---|
| learner (JAX, GPU) | Sinabox, Slurm: `run_learner.sbatch` (listens on the Tailscale IP only, ports 5588/5589) |
| actor (JAX, CPU) | robot computer: `run_actor.sh`, their `train_rlpd.py --actor`, env = `socket_env.py` (UDP 127.0.0.1:5590) |
| robot env | our client, `p0_grasping` branch `hil-serl`: `deploy/hil.py`, `client.py --hil KEY=VALUE,...` |
| demos | `convert_chunks.py`: our transition chunks (a 933 session) -> 10 Hz demo transitions, `--episodes 20` |
| numbers | `hil_report.py <session>/hil.jsonl` |

Settings (`config.py`): obs = 933's 112-d actor input, normalised with its frozen normaliser (`obs_norm_933.npz`);
action = our 9-d interface (6 IK deltas, grasp command) at 10 Hz, repeated for 5 ticks of the 50 Hz client; an
intervened step stores the mean of its 5 executed actions; discount 0.97, 200 steps per episode; reward 1 at a low
put-back (terminal), 0 at a failed attempt (terminal) or a press hand-back (terminal).

## Setup

```bash
# robot computer
uv venv -p 3.10 .venv && uv pip install -p .venv/bin/python -r serl_launcher/requirements_p0.txt "jax[cpu]==0.4.35" \
  && uv pip install -p .venv/bin/python --no-deps -e serl_launcher
# Sinabox (inside Slurm: builds .venv with CUDA JAX, then checks the GPU and the Tailscale port for LISTEN_S)
rsync -az --exclude .venv --exclude .git ./ pink-robotics@sinabox.tail51ec2d.ts.net:projects/hil-serl/
ssh pink-robotics@sinabox.tail51ec2d.ts.net 'cd ~/projects/hil-serl && mkdir -p logs && sbatch --export=ALL,BIND_IP=100.122.197.38,LISTEN_S=300 --output=logs/%x-%j.out examples/experiments/p0_ball/setup.sbatch'
.venv/bin/python examples/experiments/p0_ball/port_check.py --connect 100.122.197.38   # while it listens
```

## A run

```bash
# 1. demos: 20 put-backs of a 933 session (fake bridge or rig), copied to Sinabox runs/demos/
.venv/bin/python examples/experiments/p0_ball/convert_chunks.py demos.pkl <p0_grasping runs/deploy*/SESSION> --episodes 20
# 2. learner (from a snapshot of ~/projects/hil-serl, pink-cluster skill), W&B pink_robotics/p0_manipulation group hil_serl
sbatch -J hilserl_<run> -t 4:00:00 --output=$HOME/projects/hil-serl/logs/%x-%j.out \
  --export=ALL,PROJECT=$HOME/projects/hil-serl,RUN=<run>,DEMOS=$HOME/projects/hil-serl/runs/demos/demos.pkl examples/experiments/p0_ball/run_learner.sbatch
# 3. actor (robot computer), once the learner is up
examples/experiments/p0_ball/run_actor.sh --checkpoint_path runs/<run>_actor
# 4. client in HIL mode (p0_grasping, branch hil-serl): the usual ball-mode command plus --hil '' (or overrides),
#    --auto, no --explore-std. Fake-bridge rehearsals only: --hil recover_hold=true (no operator calls).
```
A run resumes from its checkpoint folder (no prompt).
