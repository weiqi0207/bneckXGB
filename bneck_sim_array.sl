#!/bin/bash
#SBATCH -J bneck_LD
#SBATCH -p general
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 72:00:00
#SBATCH -o bneck_LD.%A_%a.out
#SBATCH -e bneck_LD.%A_%a.err

set -euo pipefail

SHARD_REPS=400
OUT_PREFIX="train_bneck_LD"
BASE_SEED=7

# fixed by you
WIN_BP=1000000
N_HAP=44
T_MIN=1500; T_MAX=15000
NU_MIN=${NU_MIN:-0.2}
NU_MAX=${NU_MAX:-0.9}
L_BP=10000000
NE_ANC=43635
MU=1.4e-8
RHO=1.5e-8

# fast LD
LD_MAX_SITES=100
LD_EVERY_N_WINDOWS=1

# mixture target
D_GATE=-1.5
TARGET_NEG_FRAC=0.01

module purge
module load gcc
source /nas/longleaf/home/wang0207/miniconda3/etc/profile.d/conda.sh
conda activate demoinf2
export OMP_NUM_THREADS=1

TASK=${SLURM_ARRAY_TASK_ID}
SEED=$((BASE_SEED + TASK))
OUT="${OUT_PREFIX}_${TASK}.csv"

python simulate_bottleneck_features.py \
  --out "${OUT}" \
  --n_reps "${SHARD_REPS}" \
  --n_hap "${N_HAP}" \
  --L "${L_BP}" \
  --Ne_anc "${NE_ANC}" \
  --mu "${MU}" \
  --rho "${RHO}" \
  --win_bp "${WIN_BP}" \
  --T_min "${T_MIN}" --T_max "${T_MAX}" \
  --nu_min "${NU_MIN}" --nu_max "${NU_MAX}" \
  --ld_max_sites "${LD_MAX_SITES}" \
  --ld_every_n_windows "${LD_EVERY_N_WINDOWS}" \
  --D_gate "${D_GATE}" \
  --target_negD_frac "${TARGET_NEG_FRAC}" \
  --seed "${SEED}"