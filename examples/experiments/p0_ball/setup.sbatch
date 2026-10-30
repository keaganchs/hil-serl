#!/bin/bash
#SBATCH --job-name=hilserl_setup
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
# Builds .venv (GPU JAX) in the checkout, then the setup check (JAX on the GPU, the Tailscale port). Submit from the checkout.
set -euo pipefail
echo "[$(date)] job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none}"
UV=$HOME/.local/bin/uv
export GIT_CONFIG_GLOBAL=/dev/null  # the account rewrites github https to ssh, which has no key here
[ -x .venv/bin/python ] || $UV venv -q -p 3.10 .venv
$UV pip install -q -p .venv/bin/python -r serl_launcher/requirements_p0.txt -c serl_launcher/constraints_cuda12.txt "jax[cuda12]==0.4.35"
$UV pip install -q -p .venv/bin/python --no-deps -e serl_launcher
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.2
.venv/bin/python examples/experiments/p0_ball/port_check.py --bind "${BIND_IP:?}" --seconds "${LISTEN_S:-600}"
