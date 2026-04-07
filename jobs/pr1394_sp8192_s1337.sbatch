#!/bin/bash
#SBATCH -p H100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=pr1394_s1337
#SBATCH --output=/netscratch/%u/parameter-golf/logs/%x-%j.out

set -euo pipefail
HOST_USER="${HOST_USER:-$USER}"
export HOST_USER
cd /netscratch/$HOST_USER/parameter-golf
PYTHON_DEPS="/netscratch/$HOST_USER/parameter-golf/.deps/py312"
mkdir -p "$PYTHON_DEPS"
mkdir -p logs

# Install sentencepiece + brotli once before srun (single-rank)
srun --ntasks=1 \
  --container-image="/netscratch/$HOST_USER/containers/pytorch_25.02_fa3.sqsh" \
  --container-mounts="/netscratch/$HOST_USER/parameter-golf:/netscratch/$HOST_USER/parameter-golf" \
  --container-workdir="/netscratch/$HOST_USER/parameter-golf" \
  bash -lc "
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore
export PYTHONPATH=\"$PYTHON_DEPS\${PYTHONPATH:+:\$PYTHONPATH}\"
python -c 'import sentencepiece, brotli' 2>/dev/null || pip install --quiet --target \"$PYTHON_DEPS\" sentencepiece brotli
python -c 'import sentencepiece, brotli'
"

# Faithful PR #1394 reproduction — seed 1337
srun --kill-on-bad-exit=1 \
  --container-image="/netscratch/$HOST_USER/containers/pytorch_25.02_fa3.sqsh" \
  --container-mounts="/netscratch/$HOST_USER/parameter-golf:/netscratch/$HOST_USER/parameter-golf" \
  --container-workdir="/netscratch/$HOST_USER/parameter-golf" \
  bash -lc '
export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
export PIP_ROOT_USER_ACTION=ignore
export PYTHONPATH=/netscratch/$HOST_USER/parameter-golf/.deps/py312${PYTHONPATH:+:$PYTHONPATH}
python -c "import sentencepiece, brotli"

DATA_DIR=/netscratch/$HOST_USER/parameter-golf/data \
SEED=1337 \
RUN_ID=pr1394_sp8192_s1337 \
python -u records/track_10min_16mb/2026-04-06_pr1394_sp8192_faithful_repro/train_gpt.py
'
