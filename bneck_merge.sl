#!/bin/bash
#SBATCH -J bneck_merge
#SBATCH -o bneck_merge.%j.out
#SBATCH -e bneck_merge.%j.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

OUT_PREFIX="train_bneck_LD"
FINAL_OUT="${OUT_PREFIX}.csv"

# Collect shard CSVs
files=( ${OUT_PREFIX}_*.csv )
# Drop merged file itself if re-running
files=( "${files[@]/$FINAL_OUT}" )

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No shard CSVs found matching ${OUT_PREFIX}_*.csv"
  exit 1
fi

# Merge: keep header from the first only
awk 'FNR==1 && NR>1 {next} {print}' "${files[@]}" > "${FINAL_OUT}"
echo "Merged ${#files[@]} shards into ${FINAL_OUT}"

# (Optional) merge meta JSONs into an array
if command -v jq >/dev/null 2>&1; then
  jq -s '.' ${OUT_PREFIX}_*.meta.json > "${OUT_PREFIX}.all_meta.json"
  echo "Wrote combined metadata to ${OUT_PREFIX}.all_meta.json"
fi