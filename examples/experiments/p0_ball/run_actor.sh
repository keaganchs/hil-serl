#!/bin/bash
# HIL-SERL actor for the P0 ball task, on the robot computer: their actor loop (train_rlpd.py --actor) with JAX on the
# CPU, single-threaded and niced, so it neither takes the laptop GPU from the client's mirror nor competes for its cores.
# It binds 127.0.0.1:5590 for the client (HIL mode) and connects to the learner on Sinabox (Tailscale).
#   examples/experiments/p0_ball/run_actor.sh --checkpoint_path runs/<run> [--ip 100.122.197.38]
cd "$(dirname "$0")/../.." || exit 1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1
exec nice -n 5 ../.venv/bin/python train_rlpd.py --exp_name p0_ball --actor --ip 100.122.197.38 "$@"
