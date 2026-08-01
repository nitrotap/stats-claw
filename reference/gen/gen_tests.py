"""Golden fixtures for the statistical-test family and the categorical
bootstrap-CI items.

Run via ``python3 -m gen.gen_tests`` from ``stats-claw/reference``. Python
runs here ONLY at generation time; ``cargo test`` reads the committed JSON offline.

Most fixtures record an independent scipy/statsmodels reference in the provenance
block. The two categorical bootstrap-CI fixtures (``cramers_boot``, ``boot_delta``)
are different: their bounds are NOT an independent scipy reference but the output of
stats-claw's OWN SplitMix64 seeded resample, replayed here in Python so the committed
numbers match the Rust path bit-for-bit. Their provenance therefore names the
stats-claw generator and crate version (not scipy), and the Rust-side tests treat
them as determinism/regression fixtures, not equivalence references. (Genuine
scipy equivalence for these statistics is covered elsewhere by Monte-Carlo-error
tests and by truly scipy-derived fixtures such as ``beta_credible``.)
"""

import numpy as np
import scipy
from scipy import stats

from ._common import write_fixture

#: scipy version recorded on fixtures whose values are an independent scipy reference.
VER = scipy.__version__
#: stats-claw crate version, recorded on fixtures whose values are produced by
#: stats-claw's OWN SplitMix64 generator (self-round-trip regression fixtures), so the
#: provenance does not falsely claim a scipy reference.
STATS_CLAW_VER = "0.1.0"


# --------------------------------------------------------------------------- #
# Categorical
# --------------------------------------------------------------------------- #
def gen_chi2():
    table = [[10, 20, 30], [6, 9, 17], [8, 12, 25], [11, 14, 20]]
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    write_fixture(
        "test_chi2_independence",
        {"table": table, "statistic": float(chi2), "p_value": float(p), "df": int(dof)},
        library="scipy.stats.chi2_contingency(correction=False)",
        version=VER,
    )


def gen_cramers_v():
    table = [[10, 20, 30], [6, 9, 17], [8, 12, 25], [11, 14, 20]]
    v = stats.contingency.association(np.array(table), method="cramer")
    write_fixture(
        "test_cramers_v",
        {"table": table, "cramers_v": float(v)},
        library="scipy.stats.contingency.association(method='cramer')",
        version=VER,
    )


def gen_fisher():
    table = [[1, 9], [11, 3]]
    out = {"table": table}
    for alt in ("two-sided", "less", "greater"):
        odds, p = stats.fisher_exact(table, alternative=alt)
        out[f"p_{alt.replace('-', '_')}"] = float(p)
    out["odds_ratio"] = float(stats.fisher_exact(table)[0])
    write_fixture(
        "test_fisher_exact", out,
        library="scipy.stats.fisher_exact", version=VER,
    )


def gen_mcnemar():
    from statsmodels.stats.contingency_tables import mcnemar
    # large discordant counts -> asymptotic; small -> exact.
    large = [[30, 12], [40, 18]]
    small = [[5, 1], [3, 6]]
    r_large = mcnemar(np.array(large), exact=False, correction=False)
    r_large_cc = mcnemar(np.array(large), exact=False, correction=True)
    r_small = mcnemar(np.array(small), exact=True)
    write_fixture(
        "test_mcnemar",
        {
            "large": large,
            "stat_large": float(r_large.statistic),
            "p_large": float(r_large.pvalue),
            "stat_large_cc": float(r_large_cc.statistic),
            "p_large_cc": float(r_large_cc.pvalue),
            "small": small,
            "p_small_exact": float(r_small.pvalue),
        },
        library="statsmodels.stats.contingency_tables.mcnemar", version=VER,
    )


def gen_cochran():
    from statsmodels.stats.contingency_tables import cochrans_q
    # subjects x treatments, binary
    data = np.array([
        [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0],
        [1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1],
    ])
    r = cochrans_q(data)
    write_fixture(
        "test_cochran_q",
        {"data": data.tolist(), "statistic": float(r.statistic),
         "p_value": float(r.pvalue), "df": int(data.shape[1] - 1)},
        library="statsmodels.stats.contingency_tables.cochrans_q", version=VER,
    )


# --------------------------------------------------------------------------- #
# Parametric
# --------------------------------------------------------------------------- #
A = [5.1, 4.9, 6.2, 5.7, 5.0, 6.1, 4.8, 5.5, 5.9, 5.3]
B = [6.2, 6.8, 5.9, 7.1, 6.5, 6.0, 7.3, 6.7, 6.1, 6.9]
C = [4.5, 5.1, 4.8, 5.3, 4.9, 5.0, 4.7, 5.2]


def _ttest_block(name, res, df, alt_results):
    out = {"statistic": float(res.statistic), "df": float(df)}
    for alt, p in alt_results.items():
        out[f"p_{alt}"] = float(p)
    return out


def gen_t_tests():
    a, b = np.array(A), np.array(B)
    # 1-sample vs popmean
    popmean = 5.0
    alts = {}
    for alt in ("two-sided", "less", "greater"):
        alts[alt.replace("-", "_")] = stats.ttest_1samp(a, popmean, alternative=alt).pvalue
    one = _ttest_block("1samp", stats.ttest_1samp(a, popmean), len(a) - 1, alts)
    one["popmean"] = popmean
    one["sample"] = A

    # independent (pooled)
    alts = {alt.replace("-", "_"): stats.ttest_ind(a, b, alternative=alt).pvalue
            for alt in ("two-sided", "less", "greater")}
    ind = _ttest_block("ind", stats.ttest_ind(a, b), len(a) + len(b) - 2, alts)

    # paired
    alts = {alt.replace("-", "_"): stats.ttest_rel(a, b, alternative=alt).pvalue
            for alt in ("two-sided", "less", "greater")}
    paired = _ttest_block("paired", stats.ttest_rel(a, b), len(a) - 1, alts)

    # welch
    rw = stats.ttest_ind(a, b, equal_var=False)
    alts = {alt.replace("-", "_"): stats.ttest_ind(a, b, equal_var=False, alternative=alt).pvalue
            for alt in ("two-sided", "less", "greater")}
    welch = _ttest_block("welch", rw, rw.df, alts)

    write_fixture(
        "test_t_test",
        {"a": A, "b": B, "one_sample": one, "independent": ind,
         "paired": paired, "welch": welch},
        library="scipy.stats.ttest_1samp/ind/rel", version=VER,
    )


def gen_anova():
    a, b, c = np.array(A), np.array(B), np.array(C)
    f, p = stats.f_oneway(a, b, c)
    groups = [A, B, C]
    grand = np.concatenate(groups)
    k = len(groups)
    n = len(grand)
    ss_between = sum(len(g) * (np.mean(g) - grand.mean()) ** 2 for g in groups)
    ss_total = ((grand - grand.mean()) ** 2).sum()
    eta_sq = ss_between / ss_total
    write_fixture(
        "test_anova_oneway",
        {"groups": groups, "statistic": float(f), "p_value": float(p),
         "df_between": float(k - 1), "df_within": float(n - k),
         "eta_squared": float(eta_sq)},
        library="scipy.stats.f_oneway", version=VER,
    )


def gen_variance_tests():
    a, b, c = np.array(A), np.array(B), np.array(C)
    lev_w, lev_p = stats.levene(a, b, c, center="mean")
    bar_w, bar_p = stats.bartlett(a, b, c)
    k = 3
    n = len(a) + len(b) + len(c)
    write_fixture(
        "test_variance",
        {"groups": [A, B, C],
         "levene_statistic": float(lev_w), "levene_p": float(lev_p),
         "levene_df_between": float(k - 1), "levene_df_within": float(n - k),
         "bartlett_statistic": float(bar_w), "bartlett_p": float(bar_p),
         "bartlett_df": float(k - 1)},
        library="scipy.stats.levene(center='mean')/bartlett", version=VER,
    )


# --------------------------------------------------------------------------- #
# Nonparametric
# --------------------------------------------------------------------------- #
def gen_mann_whitney():
    a, b = np.array(A), np.array(B)
    out = {"a": A, "b": B}
    for alt in ("two-sided", "less", "greater"):
        r = stats.mannwhitneyu(a, b, use_continuity=True, alternative=alt)
        out[f"u_{alt.replace('-', '_')}"] = float(r.statistic)
        out[f"p_{alt.replace('-', '_')}"] = float(r.pvalue)
    # statistic is the same U for a (scipy returns U1)
    write_fixture("test_mann_whitney", out,
                  library="scipy.stats.mannwhitneyu(use_continuity=True)", version=VER)


def gen_kruskal():
    h, p = stats.kruskal(np.array(A), np.array(B), np.array(C))
    write_fixture(
        "test_kruskal",
        {"groups": [A, B, C], "statistic": float(h), "p_value": float(p),
         "df": float(2)},
        library="scipy.stats.kruskal", version=VER,
    )


def gen_wilcoxon():
    a, b = np.array(A), np.array(B)
    out = {"a": A, "b": B}
    for alt in ("two-sided", "less", "greater"):
        r = stats.wilcoxon(a, b, correction=True, alternative=alt, mode="approx")
        out[f"w_{alt.replace('-', '_')}"] = float(r.statistic)
        out[f"p_{alt.replace('-', '_')}"] = float(r.pvalue)
    write_fixture("test_wilcoxon", out,
                  library="scipy.stats.wilcoxon(correction=True, mode='approx')", version=VER)


def gen_friedman():
    # repeated measures: rows=subjects, columns=treatments
    m1 = [7, 9, 8, 6, 7, 8, 9, 7]
    m2 = [6, 5, 7, 5, 6, 6, 7, 5]
    m3 = [8, 9, 9, 7, 8, 9, 9, 8]
    stat, p = stats.friedmanchisquare(m1, m2, m3)
    n = len(m1)
    k = 3
    w = stat / (n * (k - 1))
    write_fixture(
        "test_friedman",
        {"measurements": [m1, m2, m3], "statistic": float(stat),
         "p_value": float(p), "df": float(k - 1), "kendalls_w": float(w)},
        library="scipy.stats.friedmanchisquare", version=VER,
    )


# --------------------------------------------------------------------------- #
# Exact rank-test null distributions: small, distinct, untied data so scipy's
# method="exact" is well-defined. Tolerances: p abs <= 1e-8.
# --------------------------------------------------------------------------- #
# Distinct, untied small samples for the exact Mann-Whitney U distribution.
MWU_A = [1.0, 3.0, 5.0, 7.0, 9.0]
MWU_B = [2.0, 4.0, 6.0, 8.0]

# Paired samples whose nonzero differences are distinct in magnitude (no ties,
# no zeros), the domain of the exact Wilcoxon signed-rank distribution.
WIL_A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
WIL_B = [1.7, 2.3, 3.9, 5.3, 3.2, 8.1, 6.0]


def gen_mann_whitney_exact():
    a, b = np.array(MWU_A), np.array(MWU_B)
    out = {"a": MWU_A, "b": MWU_B}
    for alt in ("two-sided", "less", "greater"):
        r = stats.mannwhitneyu(a, b, alternative=alt, method="exact")
        key = alt.replace("-", "_")
        out[f"u_{key}"] = float(r.statistic)
        out[f"p_{key}"] = float(r.pvalue)
    write_fixture("test_mann_whitney_exact", out,
                  library="scipy.stats.mannwhitneyu(method='exact')", version=VER)


def gen_wilcoxon_exact():
    a, b = np.array(WIL_A), np.array(WIL_B)
    out = {"a": WIL_A, "b": WIL_B}
    for alt in ("two-sided", "less", "greater"):
        r = stats.wilcoxon(a, b, alternative=alt, method="exact")
        key = alt.replace("-", "_")
        out[f"w_{key}"] = float(r.statistic)
        out[f"p_{key}"] = float(r.pvalue)
    write_fixture("test_wilcoxon_exact", out,
                  library="scipy.stats.wilcoxon(method='exact')", version=VER)


# Small samples for the exact KS distributions (one- and two-sample).
KS_SAMPLE = [-0.5, 0.1, 0.8, -1.2, 0.3, 1.5, -0.3, 0.9]
KS_SAMPLE_B = [0.2, -0.1, 1.1, -0.7, 0.5]


def gen_ks_exact():
    x = np.array(KS_SAMPLE)
    out = {"sample": KS_SAMPLE, "sample_b": KS_SAMPLE_B}
    for alt in ("two-sided", "less", "greater"):
        r = stats.ks_1samp(x, stats.norm.cdf, alternative=alt, method="exact")
        key = alt.replace("-", "_")
        out[f"one_statistic_{key}"] = float(r.statistic)
        out[f"one_p_{key}"] = float(r.pvalue)
    two = stats.ks_2samp(x, np.array(KS_SAMPLE_B), method="exact")
    out["two_statistic"] = float(two.statistic)
    out["two_p"] = float(two.pvalue)
    write_fixture("test_ks_exact", out,
                  library="scipy.stats.ks_1samp/ks_2samp(method='exact')", version=VER)


# --------------------------------------------------------------------------- #
# Goodness of fit
# --------------------------------------------------------------------------- #
SAMPLE = [
    -0.21, 0.34, 1.15, -0.78, 0.42, -1.30, 0.05, 0.88, -0.46, 1.02,
    0.19, -0.95, 0.61, -0.33, 0.27, 1.44, -0.58, 0.13, -1.10, 0.70,
    0.02, -0.41, 0.99, -0.67, 0.36, 1.21, -0.88, 0.50, -0.24, 0.81,
]


def gen_ks():
    import math
    x = np.array(SAMPLE)
    # stats-claw uses the classic large-sample Smirnov asymptotic Q_KS(√n · D);
    # scipy exposes it directly as kstwobign.sf. (scipy's ks_2samp 'asymp' path
    # instead uses the finite-n kstwo distribution, so we compute the reference
    # from kstwobign to match the implemented asymptotic on both sides.)
    one = stats.kstest(x, "norm", method="asymp")  # one-sample asymp = kstwobign
    b = np.array(SAMPLE) + 0.5
    d_two = stats.ks_2samp(x, b).statistic
    n1, n2 = len(x), len(b)
    en = n1 * n2 / (n1 + n2)
    two_p = float(stats.kstwobign.sf(math.sqrt(en) * d_two))
    write_fixture(
        "test_ks",
        {"sample": SAMPLE, "one_statistic": float(one.statistic),
         "one_p": float(one.pvalue),
         "sample_b": b.tolist(),
         "two_statistic": float(d_two), "two_p": two_p},
        library="scipy.stats.kstest(asymp)/kstwobign.sf", version=VER,
    )


def gen_anderson():
    r = stats.anderson(np.array(SAMPLE), dist="norm")
    write_fixture(
        "test_anderson",
        {"sample": SAMPLE, "statistic": float(r.statistic),
         "critical_values": [float(v) for v in r.critical_values],
         "significance_levels": [float(v) for v in r.significance_level]},
        library="scipy.stats.anderson(dist='norm')", version=VER,
    )


def gen_shapiro():
    r = stats.shapiro(np.array(SAMPLE))
    write_fixture(
        "test_shapiro",
        {"sample": SAMPLE, "statistic": float(r.statistic), "p_value": float(r.pvalue)},
        library="scipy.stats.shapiro", version=VER,
    )


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
X = [1.0, 2.1, 2.9, 4.2, 5.0, 5.8, 7.1, 8.0, 9.2, 10.1]
Y = [2.2, 3.9, 6.1, 7.8, 10.2, 11.7, 14.3, 15.9, 18.1, 20.4]


def gen_correlation():
    x, y = np.array(X), np.array(Y)
    out = {"x": X, "y": Y}
    for alt in ("two-sided", "less", "greater"):
        a = alt.replace("-", "_")
        pr = stats.pearsonr(x, y, alternative=alt)
        sr = stats.spearmanr(x, y, alternative=alt)
        kt = stats.kendalltau(x, y, alternative=alt)
        out[f"pearson_p_{a}"] = float(pr.pvalue)
        out[f"spearman_p_{a}"] = float(sr.pvalue)
        out[f"kendall_p_{a}"] = float(kt.pvalue)
    out["pearson_r"] = float(stats.pearsonr(x, y).statistic)
    out["spearman_r"] = float(stats.spearmanr(x, y).statistic)
    out["kendall_tau"] = float(stats.kendalltau(x, y).statistic)
    write_fixture("test_correlation", out,
                  library="scipy.stats.pearsonr/spearmanr/kendalltau", version=VER)


# --------------------------------------------------------------------------- #
# Categorical bootstrap CIs, coverage, credible interval
# --------------------------------------------------------------------------- #
def _splitmix64_stream(seed, count):
    """Reproduce stats-claw's SplitMix64 next_u64 stream (for fixture parity)."""
    mask = (1 << 64) - 1
    state = seed & mask
    out = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z = z ^ (z >> 31)
        out.append(z & mask)
    return out


def _uniform_index(u, n):
    return u % n


def _cramers_v(table):
    t = np.array(table, dtype=float)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    n = t.sum()
    r, c = t.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def _table_to_observations(table):
    """Flatten a contingency table into one (row, col) cell index per observation.

    Cell (i, j) with count c contributes c copies of the flat index i*ncols+j, in
    row-major order — the canonical observation list both sides resample from.
    """
    obs = []
    ncols = len(table[0])
    for i, row in enumerate(table):
        for j, count in enumerate(row):
            obs.extend([i * ncols + j] * int(count))
    return obs, ncols


def gen_cramers_boot():
    """Cramér's V bootstrap CI by resampling *observations* with stats-claw's RNG.

    Mirrors the Rust path exactly: expand the table to N individual cell-index
    observations (row-major), draw B with-replacement resamples of size N from the
    SplitMix64 stream (seed=42), rebuild each resampled table, recompute Cramér's V,
    then take the 2.5/97.5 percentile CI.
    """
    table = [[10, 20, 30], [6, 9, 17], [8, 12, 25], [11, 14, 20]]
    obs, ncols = _table_to_observations(table)
    nrows = len(table)
    n = len(obs)
    seed, b = 42, 10000
    stream = _splitmix64_stream(seed, b * n)
    vs = []
    k = 0
    for _ in range(b):
        resampled = np.zeros((nrows, ncols), dtype=float)
        for _ in range(n):
            cell = obs[_uniform_index(stream[k], n)]
            k += 1
            resampled[cell // ncols, cell % ncols] += 1.0
        if (resampled.sum(axis=0) == 0).any() or (resampled.sum(axis=1) == 0).any():
            vs.append(0.0)
            continue
        vs.append(_cramers_v(resampled))
    vs = np.sort(vs)
    lo = vs[int(np.floor(0.025 * b))]
    hi = vs[int(np.floor(0.975 * b))]
    write_fixture(
        "test_cramers_boot",
        {"table": table, "seed": seed, "b": b, "alpha": 0.05,
         "point": _cramers_v(np.array(table, dtype=float)),
         "ci_low": float(lo), "ci_high": float(hi)},
        library="stats-claw SplitMix64 observation-bootstrap + percentile CI (self-round-trip; not a scipy reference)",
        version=STATS_CLAW_VER, seed=seed,
    )


def gen_boot_delta():
    """Paired-delta bootstrap CI with stats-claw's RNG."""
    before = [12.0, 15.0, 14.0, 10.0, 13.0, 11.0, 16.0, 9.0, 14.0, 12.0]
    after = [14.0, 16.0, 13.0, 12.0, 15.0, 14.0, 18.0, 11.0, 15.0, 14.0]
    delta = np.array(after) - np.array(before)
    n = len(delta)
    seed, b = 777, 10000
    stream = _splitmix64_stream(seed, b * n)
    stats_out = []
    k = 0
    for _ in range(b):
        s = 0.0
        for _ in range(n):
            idx = _uniform_index(stream[k], n)
            k += 1
            s += delta[idx]
        stats_out.append(s / n)
    stats_out = np.sort(stats_out)
    lo = stats_out[int(np.floor(0.025 * b))]
    hi = stats_out[int(np.floor(0.975 * b))]
    write_fixture(
        "test_boot_delta",
        {"before": before, "after": after, "seed": seed, "b": b, "alpha": 0.05,
         "point": float(delta.mean()), "ci_low": float(lo), "ci_high": float(hi)},
        library="stats-claw SplitMix64 paired-delta bootstrap + percentile CI (self-round-trip; not a scipy reference)",
        version=STATS_CLAW_VER, seed=seed,
    )


# --------------------------------------------------------------------------- #
# Log-space / extreme-p path: scipy <dist>.logsf / logcdf at extreme inputs,
# and test-level extreme-input log p-values. Tolerance: log values rel <= 1e-9
# (abs <= 1e-9 where |log| < 1, since a near-zero log has unbounded relative
# sensitivity to a tiny absolute error).
# --------------------------------------------------------------------------- #
def gen_logsf():
    """Per-distribution logsf/logcdf at body and extreme-tail points."""
    out = {}
    # Standard normal N(0, 1).
    out["normal"] = {
        "x": [0.5, 5.0, 20.0, 40.0],
        "logsf": [float(stats.norm.logsf(x)) for x in (0.5, 5.0, 20.0, 40.0)],
        "logcdf": [float(stats.norm.logcdf(x)) for x in (0.5, 5.0, 20.0, 40.0)],
    }
    # Student's t, df = 5.
    txs = [1.0, 10.0, 50.0]
    out["t"] = {
        "df": 5,
        "x": txs,
        "logsf": [float(stats.t.logsf(x, 5)) for x in txs],
        "logcdf": [float(stats.t.logcdf(x, 5)) for x in txs],
    }
    # Chi-squared, df = 3.
    cxs = [2.0, 50.0, 200.0]
    out["chi2"] = {
        "df": 3,
        "x": cxs,
        "logsf": [float(stats.chi2.logsf(x, 3)) for x in cxs],
        "logcdf": [float(stats.chi2.logcdf(x, 3)) for x in cxs],
    }
    # F, d1 = 3, d2 = 10.
    fxs = [1.0, 30.0, 10000.0]
    out["f"] = {
        "d1": 3,
        "d2": 10,
        "x": fxs,
        "logsf": [float(stats.f.logsf(x, 3, 10)) for x in fxs],
        "logcdf": [float(stats.f.logcdf(x, 3, 10)) for x in fxs],
    }
    write_fixture(
        "test_logsf", out,
        library="scipy.stats.{norm,t,chi2,f}.logsf/logcdf", version=VER,
    )


def gen_extreme_p():
    """Test-level extreme-input log p-values (one-sample t with a huge effect).

    The sample sits tightly around 100 and is tested against a null mean of 0, so
    |t| ~ 2.8e5 on df = 4 — a p-value (~1e-21) where the linear path has lost
    relative precision but the log p-value is finite. The reference log p-values
    come from scipy's t.logsf / logcdf for the three alternatives.
    """
    sample = [100.0, 100.001, 99.999, 100.0005, 99.9995]
    popmean = 0.0
    a = np.array(sample)
    res = stats.ttest_1samp(a, popmean)
    t = float(res.statistic)
    df = len(sample) - 1
    log_two_sided = float(np.log(2.0) + stats.t.logsf(abs(t), df))
    log_greater = float(stats.t.logsf(t, df))
    log_less = float(stats.t.logcdf(t, df))
    write_fixture(
        "test_extreme_p",
        {
            "sample": sample, "popmean": popmean,
            "statistic": t, "df": float(df),
            "p_two_sided": float(res.pvalue),
            "log_p_two_sided": log_two_sided,
            "log_p_greater": log_greater,
            "log_p_less": log_less,
        },
        library="scipy.stats.ttest_1samp + t.logsf/logcdf", version=VER,
    )


def gen_beta_credible():
    """Beta(a0+k, b0+n-k) credible interval via scipy.stats.beta.ppf."""
    a0, b0, k, n = 1.0, 1.0, 7, 10
    a, bb = a0 + k, b0 + (n - k)
    lo = float(stats.beta.ppf(0.025, a, bb))
    hi = float(stats.beta.ppf(0.975, a, bb))
    write_fixture(
        "test_beta_credible",
        {"alpha0": a0, "beta0": b0, "successes": k, "trials": n,
         "alpha": 0.05, "ci_low": lo, "ci_high": hi,
         "posterior_mean": a / (a + bb)},
        library="scipy.stats.beta.ppf", version=VER,
    )


def main():
    gen_chi2()
    gen_cramers_v()
    gen_fisher()
    gen_mcnemar()
    gen_cochran()
    gen_t_tests()
    gen_anova()
    gen_variance_tests()
    gen_mann_whitney()
    gen_mann_whitney_exact()
    gen_kruskal()
    gen_wilcoxon()
    gen_wilcoxon_exact()
    gen_friedman()
    gen_ks()
    gen_ks_exact()
    gen_anderson()
    gen_shapiro()
    gen_correlation()
    gen_cramers_boot()
    gen_boot_delta()
    gen_beta_credible()
    gen_logsf()
    gen_extreme_p()


if __name__ == "__main__":
    main()
