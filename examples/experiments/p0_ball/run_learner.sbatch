#!/bin/bash
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%j.out
# HIL-SERL learner for the P0 ball task on Sinabox. Submit from a snapshot of the checkout (pink-cluster skill):
#   sbatch -J hilserl_<run> -t <time> --export=ALL,PROJECT=$HOME/projects/hil-serl,RUN=<run>,DEMOS="<pkl> <pkl>" run_learner.sbatch
# Listens on the Tailscale IP only (5588 data / requests, 5589 weight broadcast); the robot computer's actor connects to it.
set -euo pipefail
echo "[$(date)] job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none} code=$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.25 PYTHONPATH="$PWD/serl_launcher:$PWD/examples"
demo_args=(); for d in ${DEMOS:?}; do demo_args+=(--demo_path "$d"); done
cd examples
"${PROJECT:?}/.venv/bin/python" train_rlpd.py --exp_name p0_ball --learner --bind_ip "${BIND_IP:-100.122.197.38}" \
    --checkpoint_path "$PROJECT/runs/${RUN:?}" "${demo_args[@]}" ${EXTRA:-}
