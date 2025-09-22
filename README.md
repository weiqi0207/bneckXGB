# bneckXGB
XGBoost models to detect/quantify population bottlenecks from msprime-simulated genomic features (Tajima’s D, SFS bins, LD, π, θW).

---

## Prerequisites

1. HPC environment with Slurm job scheduler
2. Python with the following packages:
      msprime (for coalescent simulations)
      xgboost
      numpy, pandas, scikit-learn

---

## Problem Definition

This repository implements an **XGBoost-based approach** to detect and characterize **population bottlenecks** from simulated genomic features.  

The organizers specified the following parameters:

Generation time: $1\ \text{year}$

Mutation rate (per bp per generation): $\mu = 1.4 \times 10^{-8}$

Recombination rate (per bp per generation): $r = 1.5 \times 10^{-8}$

Genomic region length: $L = 1.0 \times 10^{8}\ \text{bp}$

Number of haplotypes: $n_{\text{hap}} = 44$


The input test file was provided as a compressed VCF:

GHIST_2025_bottleneck.testing.vcf.gz

yaml
Copy code

### Effective Population Size Estimation

A rough estimate of effective population size was derived using Watterson’s estimator:

\[
N_e \approx \frac{\tfrac{S}{L}}{4 \mu \, a_{n-1}}
\]

where:
- \( S \) = number of segregating sites,  
- \( L \) = sequence length (bp),  
- \( a_{n-1} \) = harmonic number for \( n-1 \),  
- \( \mu \) = mutation rate.  

To select an appropriate **window size**, exploratory analysis was performed in `Decoy.ipynb` by comparing **global summary statistics** (e.g., Tajima’s D vs. genomic coordinate).

---

## Data Generation

Simulations were generated with **msprime**, wrapped via custom scripts:

- `simulate_bottleneck_features.py` — generate feature CSV + JSON per replicate.  
- `bneck_sim_array.sl` — submit HPC job array for batch simulation.  
- `bneck_merge.sl` — merge outputs into a single training set.

This produces a **feature matrix** (`.csv`) and accompanying metadata (`.json`).  
Fixed parameters (e.g., \( n_{\text{hap}} \)) may be dropped from the CSV without loss of predictive performance.

---

## Workflow

1. **Simulate training data**  
   ```bash
   sbatch bneck_sim_array.sl
   sbatch --dependency=afterok:<jobid> bneck_merge.sl
   ```
2. **Train XGBoost model**
   ```bash
    python train_xgb.py --train-csv train_bneck_LD.csv --out model.json
    ```
3. **Predict on new data (VCF → features → XGB prediction)**
   ```bash
    python predict_xgb.py --model model.json \
                          --features GHIST_2025_features.csv \
                          --out predictions.csv

   ```

---

## Limitations

1. Simulation bottleneck: Current msprime setup is extremely slow on large genomic windows.
2. Fragmented pipeline: Feature generation, training, and prediction are spread across multiple scripts.
3. Model interpretability: Raw XGBoost predictions are hard to interpret biologically.

---

## Future Improvements

1. Fuse all steps into a single streamlined pipeline.
2. Integrate LIME/SHAP for feature attribution and model interpretability.
3. Use GHIST leaderboard feedback to improve both model architecture and simulation realism.
4. Optimize HPC job management to reduce walltime for msprime simulations.

---
