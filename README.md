# bneckXGB
XGBoost models to detect/quantify population bottlenecks from msprime-simulated genomic features (Tajima’s D, SFS bins, LD, π, θW).

## Problem Definition

This repository implements an **XGBoost-based approach** to detect and characterize **population bottlenecks** from simulated genomic features.  

The work was developed in the context of the **GHIST 2025 bottleneck challenge**, where the organizers provided the following demographic and sequencing parameters:

- **Generation time**: 1 year  
- **Mutation rate (per generation per base pair)**:  
  \[
  \mu = 1.4 \times 10^{-8}
  \]
- **Recombination rate (per generation per base pair)**:  
  \[
  r = 1.5 \times 10^{-8}
  \]
- **Genomic region length**:  
  \[
  L = 100,000,000 \ \text{bp}
  \]
- **Number of haplotypes**:  
  \[
  n_{\text{hap}} = 44
  \]

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
