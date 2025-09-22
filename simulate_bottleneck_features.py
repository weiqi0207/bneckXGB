#!/usr/bin/env python3
"""
fast bottleneck simulator with targeted negative-D mix and on-demand LD

Key behavior:
- Windows: 1,000,000 bp (over L = 10,000,000 → 10 windows)
- n_hap = 44, Ne_anc = 43635, mu = 1.4e-8, rho = 1.5e-8 (defaults)
- No rejection sampling; keep every row.
- While realized fraction of rows with D_median < D_gate is below target_negD_frac,
  we bias (within the same ranges) toward parameters that more often yield negative D.
- LD is computed only once per kept row with a per-window site cap and sorted subsample,
  returning a single feature r2_1kb_mean (global mean across LD windows).

Usage example:
  python simulation.py --out train_bneck_LD_0.csv --n_reps 400 --seed 7
"""

import argparse, os, math, json, random
import numpy as np
import pandas as pd
import msprime
import tskit

# -------------------------------
# Helpers: Tajima's D and windows
# -------------------------------

def harmonic_number(n: int) -> float:
    if n <= 1:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n, dtype=np.float64)))

def tajimas_d_from_window(S: int, pi: float, n: int) -> float:
    if S <= 0 or n < 2:
        return np.nan
    a1 = harmonic_number(n)
    if a1 == 0.0:
        return np.nan
    a2 = float(np.sum(1.0 / (np.arange(1, n, dtype=np.float64) ** 2)))
    b1 = (n + 1.0) / (3.0 * (n - 1.0))
    b2 = 2.0 * (n * n + n + 3.0) / (9.0 * n * (n - 1.0))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2.0) / (a1 * n) + a2 / (a1 * a1)
    e1 = c1 / a1
    e2 = c2 / (a1 * a1 + a2)
    thetaW = S / a1
    var = e1 * S + e2 * S * (S - 1.0)
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return (pi - thetaW) / math.sqrt(var)

def window_edges(L: float, win_bp: int):
    nwin = int(math.ceil(L / float(win_bp)))
    edges = np.arange(0, nwin * win_bp + 1, win_bp, dtype=np.float64)
    edges[-1] = L  # ensure exact end
    return edges, nwin

# -------------------------------
# Fast screening (no LD, no dense G)
# -------------------------------

def screen_features_no_ld(ts: tskit.TreeSequence, win_bp: int):
    """
    Streaming per-window S, pi_sum, singleton%, doubleton%, Tajima's D, and global aggregates.
    Avoids ts.genotype_matrix() to keep per-row overhead low.
    """
    L = float(ts.sequence_length)
    edges, nwin = window_edges(L, win_bp)
    win_len = np.diff(edges)

    n = ts.num_samples
    a1 = harmonic_number(n)

    S = np.zeros(nwin, dtype=np.int64)
    pi_sum = np.zeros(nwin, dtype=np.float64)
    singletons = np.zeros(nwin, dtype=np.int64)
    doubletons = np.zeros(nwin, dtype=np.int64)

    if n < 2:
        nan = np.full(nwin, np.nan)
        zf = np.zeros(nwin, dtype=np.float64)
        zi = np.zeros(nwin, dtype=int)
        perwin = dict(S=zi, pi_sum=zf, thetaW=zf, pi_bp=zf, thetaW_bp=zf,
                      D=nan, sing_pct=nan, doub_pct=nan, r2mean=nan, win_len=win_len)
        return aggregate_global(perwin)

    n_float = float(n)
    for var in ts.variants():
        pos = var.site.position
        w = int(min(np.searchsorted(edges, pos, side="right") - 1, nwin - 1))
        g = var.genotypes  # length n (haploid here)
        k = int(g.sum())
        if k == 0 or k == n:
            continue
        S[w] += 1
        p = k / n_float
        # unbiased per-site pi = 2p(1-p) * n/(n-1)
        pi_sum[w] += (2.0 * p * (1.0 - p)) * (n_float / (n_float - 1.0))
        mac = k if k <= (n - k) else (n - k)
        if mac == 1:
            singletons[w] += 1
        elif mac == 2:
            doubletons[w] += 1

    # window metrics
    D = np.full(nwin, np.nan, dtype=np.float64)
    thetaW = np.zeros(nwin, dtype=np.float64)
    if a1 > 0:
        thetaW[S > 0] = S[S > 0] / a1
    for w in range(nwin):
        if S[w] > 0:
            D[w] = tajimas_d_from_window(int(S[w]), float(pi_sum[w]), n)

    # per-bp rates
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_bp = np.divide(pi_sum, win_len, out=np.zeros_like(pi_sum), where=win_len > 0)
        thetaW_bp = np.divide(thetaW, win_len, out=np.zeros_like(thetaW), where=win_len > 0)
        S_float = S.astype(np.float64)
        sing_pct = np.where(S > 0, singletons / S_float, np.nan)
        doub_pct = np.where(S > 0, doubletons / S_float, np.nan)

    perwin = dict(S=S.astype(int), pi_sum=pi_sum, thetaW=thetaW,
                  pi_bp=pi_bp, thetaW_bp=thetaW_bp, D=D,
                  sing_pct=sing_pct, doub_pct=doub_pct,
                  r2mean=np.full(nwin, np.nan), win_len=win_len)
    return aggregate_global(perwin)

# -------------------------------
# LD (computed once per kept row)
# -------------------------------

def compute_ld_r2_within_1kb_sorted(positions_bp: np.ndarray, G: np.ndarray) -> float:
    """
    positions_bp sorted ascending, G (sites x haplotypes) 0/1.
    Returns mean r^2 for pairs within 1 kb.
    """
    n_sites = G.shape[0]
    if n_sites < 2:
        return np.nan

    Gf = G.astype(np.float64, copy=False)
    pos = positions_bp.astype(np.float64, copy=False)

    eps = 1e-12
    p_site = Gf.mean(axis=1)
    valid = (p_site > eps) & (p_site < 1.0 - eps)

    r2_vals = []
    j_start = 0
    for i in range(n_sites):
        if not valid[i]:
            while j_start < n_sites and pos[j_start] <= pos[i]:
                j_start += 1
            continue
        while j_start < n_sites and pos[j_start] <= pos[i]:
            j_start += 1
        xi = Gf[i, :]
        pai = p_site[i]
        j = j_start
        while j < n_sites and (pos[j] - pos[i]) <= 1000.0:
            if valid[j]:
                yj = Gf[j, :]
                pbj = p_site[j]
                pAB = (xi * yj).mean()
                D = pAB - pai * pbj
                denom = pai * (1.0 - pai) * pbj * (1.0 - pbj)
                if denom > eps:
                    r2_vals.append((D * D) / denom)
            j += 1

    return float(np.mean(r2_vals)) if r2_vals else np.nan

def ld_feature_for_row(ts: tskit.TreeSequence, win_bp: int, ld_max_sites: int, ld_every_n_windows: int):
    L = float(ts.sequence_length)
    edges, nwin = window_edges(L, win_bp)
    if ts.num_sites == 0:
        return np.nan

    positions_bp = np.fromiter((s.position for s in ts.sites()), dtype=np.float64, count=ts.num_sites)
    G = ts.genotype_matrix()  # (sites x haplotypes), build once

    r2_vals = []
    for w in range(nwin):
        if (w % ld_every_n_windows) != 0:
            continue
        lo, hi = edges[w], edges[w+1]
        mask = (positions_bp >= lo) & (positions_bp < hi)
        if not np.any(mask):
            continue
        pos_w = positions_bp[mask]
        G_w = G[mask, :]

        if ld_max_sites and G_w.shape[0] > ld_max_sites:
            idx = np.random.choice(G_w.shape[0], ld_max_sites, replace=False)
            pos_w = pos_w[idx]; G_w = G_w[idx, :]

        order = np.argsort(pos_w)  # keep positions sorted after subsample
        pos_w = pos_w[order]; G_w = G_w[order, :]

        r2 = compute_ld_r2_within_1kb_sorted(pos_w, G_w)
        if np.isfinite(r2):
            r2_vals.append(r2)

    return float(np.mean(r2_vals)) if r2_vals else np.nan

# -------------------------------
# Global aggregation
# -------------------------------

def aggregate_global(perwin: dict) -> dict:
    D = perwin["D"]
    pi_bp = perwin["pi_bp"]
    thetaW_bp = perwin["thetaW_bp"]
    sing_pct = perwin["sing_pct"]
    doub_pct = perwin["doub_pct"]
    r2mean = perwin["r2mean"]
    win_len = perwin["win_len"]
    S_win = perwin["S"].astype(np.float64)

    valid = np.isfinite(D)
    frac_D_pos = float(np.mean(D[valid] > 0.0)) if np.any(valid) else np.nan
    frac_D_neg15 = float(np.mean(D[valid] < -1.5)) if np.any(valid) else np.nan

    with np.errstate(invalid='ignore'):
        wlen_sum = np.nansum(win_len)
        pi_bp_mean_w = float(np.nansum(pi_bp * win_len) / wlen_sum) if wlen_sum > 0 else np.nan
        thetaW_bp_mean_w = float(np.nansum(thetaW_bp * win_len) / wlen_sum) if wlen_sum > 0 else np.nan

    tot_S = float(np.nansum(S_win))
    tot_singletons = float(np.nansum(np.where(S_win > 0, sing_pct * S_win, 0.0)))
    tot_doubletons  = float(np.nansum(np.where(S_win > 0, doub_pct * S_win, 0.0)))
    singleton_pct_global = (tot_singletons / tot_S) if tot_S > 0 else np.nan
    doubleton_pct_global = (tot_doubletons  / tot_S) if tot_S > 0 else np.nan

    R = (pi_bp_mean_w / thetaW_bp_mean_w) if (thetaW_bp_mean_w is not None and np.isfinite(thetaW_bp_mean_w) and thetaW_bp_mean_w > 0) else np.nan

    out = dict(
        D_median=float(np.nanmedian(D)),
        D_mean=float(np.nanmean(D)),
        D_sd=float(np.nanstd(D)),
        frac_D_pos=frac_D_pos,
        frac_D_neg15=frac_D_neg15,
        pi_bp_mean=pi_bp_mean_w,
        thetaW_bp_mean=thetaW_bp_mean_w,
        R=R,
        singleton_pct=singleton_pct_global,
        doubleton_pct=doubleton_pct_global,
        r2_1kb_mean=float(np.nanmean(r2mean))
    )
    return out

# -------------------------------
# Simulation + main loop
# -------------------------------

def simulate_bottleneck_once(n_hap, L, Ne_anc, mu, rho, T_gen, nu, seed):
    dem = msprime.Demography()
    dem.add_population(name="pop", initial_size=nu * Ne_anc)
    dem.add_population_parameters_change(time=T_gen, initial_size=Ne_anc, population="pop")
    ts = msprime.sim_ancestry(samples=n_hap, sequence_length=L, demography=dem,
                              recombination_rate=rho, random_seed=seed, model="hudson")
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1)
    return ts

def main():
    ap = argparse.ArgumentParser(description="Fast bottleneck sims with targeted negative-D mix and on-demand LD.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_reps", type=int, default=400)

    # Fixed-by-you defaults (can still be overridden via CLI if needed)
    ap.add_argument("--n_hap", type=int, default=44)
    ap.add_argument("--L", type=float, default=10_000_000)
    ap.add_argument("--Ne_anc", type=float, default=43635)
    ap.add_argument("--mu", type=float, default=1.4e-8)
    ap.add_argument("--rho", type=float, default=1.5e-8)
    ap.add_argument("--win_bp", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=7)

    # Priors within your ranges
    ap.add_argument("--T_min", type=int, default=1500)
    ap.add_argument("--T_max", type=int, default=15000)
    ap.add_argument("--nu_min", type=float, default=0.2)
    ap.add_argument("--nu_max", type=float, default=0.9)

    # LD controls (accepted rows only)
    ap.add_argument("--ld_max_sites", type=int, default=100, help="cap sites per window for LD")
    ap.add_argument("--ld_every_n_windows", type=int, default=1, help="compute LD every k-th window")

    # Targeted fraction of very negative D (no rejection; we bias proposals while under target)
    ap.add_argument("--D_gate", type=float, default=-1.5, help="global (median over windows) threshold for 'very negative D'")
    ap.add_argument("--target_negD_frac", type=float, default=0.01, help="target fraction of rows with D_median < D_gate")

    args = ap.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def sample_T(biased: bool):
        # both base and biased use log-uniform; bias handled via decision frequency
        u = rng.uniform(np.log(max(1, args.T_min)), np.log(args.T_max))
        t = int(np.rint(np.exp(u)))
        if biased:
            # slight push even younger: shrink upper bound randomly
            u2 = rng.uniform(np.log(max(1, args.T_min)), np.log(max(1, t)))
            t = int(np.rint(np.exp(u2)))
        return max(args.T_min, min(t, args.T_max))

    def sample_nu(biased: bool):
        if biased:
            # bias toward smaller nu (harsher bottleneck), Beta(0.6,2.0) scaled to [nu_min, nu_max]
            x = rng.beta(0.6, 2.0)
            return float(args.nu_min + x * (args.nu_max - args.nu_min))
        else:
            return float(rng.uniform(args.nu_min, args.nu_max))

    rows = []
    neg_count = 0
    total = 0

    while total < args.n_reps:
        frac_neg = (neg_count / total) if total > 0 else 0.0
        want_biased = (frac_neg < args.target_negD_frac)

        T = sample_T(biased=want_biased)
        nu = sample_nu(biased=want_biased)
        seed_i = random.randrange(1, 10**9)

        ts = simulate_bottleneck_once(
            args.n_hap, args.L, args.Ne_anc, args.mu, args.rho,
            T_gen=T, nu=nu, seed=seed_i
        )

        # fast screen (no LD) to get D_median quickly
        feats = screen_features_no_ld(ts, win_bp=args.win_bp)
        D_med = feats.get("D_median", np.nan)
        is_neg = (np.isfinite(D_med) and D_med < args.D_gate)

        # on-demand LD once per kept row
        r2_mean = ld_feature_for_row(ts, win_bp=args.win_bp,
                                     ld_max_sites=args.ld_max_sites,
                                     ld_every_n_windows=args.ld_every_n_windows)
        feats["r2_1kb_mean"] = r2_mean

        # label/meta
        feats.update(dict(
            T_generations=int(T),
            post_decline_fraction=float(nu),
            n_hap=int(args.n_hap),
            L_bp=float(args.L),
            mu=float(args.mu),
            rho=float(args.rho)
        ))

        rows.append(feats)
        total += 1
        if is_neg:
            neg_count += 1

        if (total % 25) == 0:
            print(f"[{total}/{args.n_reps}] negD so far: {neg_count} ({neg_count/total:.3%})")

    print(f"Done: {total} rows; D_median < {args.D_gate}: {neg_count} ({neg_count/total:.3%})")

    pd.DataFrame(rows).to_csv(args.out, index=False)

    meta = dict(
        n_reps=int(args.n_reps),
        neg_D_gate=float(args.D_gate),
        target_negD_frac=float(args.target_negD_frac),
        realized_negD_frac=(neg_count/total if total else float('nan')),
        neg_count=int(neg_count),
        n_hap=int(args.n_hap),
        L=float(args.L),
        Ne_anc=float(args.Ne_anc),
        mu=float(args.mu),
        rho=float(args.rho),
        win_bp=int(args.win_bp),
        T_prior=[int(args.T_min), int(args.T_max)],
        nu_prior=[float(args.nu_min), float(args.nu_max)],
        ld_max_sites=int(args.ld_max_sites),
        ld_every_n_windows=int(args.ld_every_n_windows),
        seed=int(args.seed)
    )
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()