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
put-back (terminal), 0 at a failed attempt or a press hand-back (terminal).

**Deviations from HIL-SERL** (besides no camera, no reward classifier and a policy instead of a human):
- **IK action scale 0.5** (`socket_env.IK_SCALE`, user 2026-10-30): the client executes the learner's 6 IK actions x 0.5
  (the grasp command unscaled); intervened and demo actions are stored in the learner's units (/ 0.5, clipped to [-1, 1]).
  933's own actions do not always fit: in the 20 fake-bridge demos 25 % of the transitions (8 % of the IK elements; the
  4th IK action 17 %) are clipped. The client logs the clipped elements per intervened step (`hil.jsonl` `clip`).
- **Update rate capped** (`config.max_utd` = 4 gradient updates per online transition; `train_rlpd.py` learner waits
  beyond it). Their learner is free-running but its speed sets the ratio: their image learner (`bench_image_learner.py`,
  the RAM-insertion setup, `cta_ratio` 2) ran 5.8 steps/s = 11.6 updates/s = 1.2 per 10 Hz transition on a busy A5000
  shard; their RTX 4090 alone is ~3-4x that, so ~4. Uncapped, our state-only learner ran ~300 per transition. Our cap
  counts collected transitions only: during resets (~15-20 s here) ours waits, theirs kept training.
- **The intervener's rules** (`deploy/hil.py`, from the paper: brief corrections that steer exploration, rarer as the
  policy improves, not carrying it to success): takeover on SAFETY (press > 200 g, pad box, soft-limit clamp, `dive` = the
  pad < 8 cm above the rest estimate coming down faster than 0.2 m/s (the guard knows the table, not the dish: a 0.35 m/s
  learner descent pressed it 3.2 kg in rehearsal 2 before anything else reacted), `near` = the
  executed target fails a second guard with 3 cm more table margin and min_z, 3 cm less box_tol, so the learner never
  drives into a HOLD; a learner target the guard itself refuses, a one-tick IK jump of up to 5 cm seen in rehearsal 2,
  is replaced by the last safe target and is a takeover too) or STUCK (3 s in LIFT / 6 s in PUT without progress); release once no safety trigger holds, at a
  lift / attempt start or after 1 s with the pad >= 8 cm above the rest estimate (never held to a put-back); at most 3
  takeovers per episode (then stuck triggers are ignored and a safety trigger is a last takeover that ends the episode at
  its release), and a takeover not released within 4 s ends it (truncated, a cut-off as max_steps; the client hands back
  with the press relief lift and `--auto` recovers).

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
#    --auto, no --explore-std. Fake-bridge rehearsals only: --hil recover_hold=true,scene_reset_port=<p> (no operator
#    calls; the fake bridge started with --reset-port <p> puts the ball back in the dish when it is not resting)
# 5. numbers: per 10 min, intervention rate / duration / triggers, put-backs, clipping, presses
.venv/bin/python examples/experiments/p0_ball/hil_report.py <session>/hil.jsonl
```
A run resumes from its checkpoint folder (no prompt).
