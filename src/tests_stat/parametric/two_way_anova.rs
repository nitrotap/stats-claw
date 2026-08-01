//! Balanced two-way (factorial) analysis of variance with interaction.

use super::{f_upper_log_tail, f_upper_tail, len_f64};
use crate::error::{Error, Result};
use crate::tests_stat::TestResult;

/// Per-effect decomposition of a balanced two-way factorial ANOVA.
#[derive(Debug, Clone)]
pub struct TwoWayAnovaResult {
    /// Main effect of factor A: `F`, p-value, and `df = a − 1`.
    pub factor_a: TestResult,
    /// Main effect of factor B: `F`, p-value, and `df = b − 1`.
    pub factor_b: TestResult,
    /// The A×B interaction effect: `F`, p-value, and `df = (a − 1)(b − 1)`.
    pub interaction: TestResult,
    /// Within-cell (error) sum of squares.
    pub ss_within: f64,
    /// Within-cell (error) degrees of freedom, `a · b · (n − 1)`.
    pub df_within: f64,
}

/// Computes a balanced two-way fixed-effects ANOVA with interaction.
///
/// Partitions the total sum of squares into the two main effects (`SS_A`,
/// `SS_B`), their interaction (`SS_AB`), and the within-cell error (`SS_within`),
/// then forms `F = MS_effect / MS_within` for each effect and takes the upper-tail
/// p-value from the framework F distribution (the same path as
/// [`super::one_way_anova`]). For a balanced design the Type I, II, and III sums
/// of squares coincide, so the results match statsmodels `anova_lm` regardless of
/// its `typ`. The reported effect size is partial `η² = SS / (SS + SS_within)`.
///
/// # Arguments
///
/// * `cells` — a rectangular `a × b` grid where `cells[i][j]` holds the replicate
///   observations at level `i` of factor A and level `j` of factor B. Every cell
///   must carry the same replicate count `n ≥ 2` (balanced design).
///
/// # Returns
///
/// A [`TwoWayAnovaResult`] with one [`TestResult`] per effect plus the shared
/// within-cell error `SS_within` and its degrees of freedom `a · b · (n − 1)`.
///
/// # Errors
///
/// * [`Error::InsufficientData`] — fewer than two levels on either factor, or
///   fewer than two replicates per cell (the interaction is then inestimable).
/// * [`Error::InvalidInput`] — a ragged/unbalanced grid (rows of differing width
///   or cells with unequal replicate counts).
/// * [`Error::DegenerateInput`] — zero within-cell variation (`F` undefined).
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::parametric::two_way_anova;
///
/// let cells = vec![
///     vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
///     vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 13.0]],
/// ];
/// let r = two_way_anova(&cells)?;
/// assert_eq!(r.factor_a.df, Some(1.0));
/// assert_eq!(r.interaction.df, Some(1.0));
/// // Factor A dominates this design: F ≈ 85.56.
/// assert!((r.factor_a.statistic - 85.5625).abs() < 1e-6);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn two_way_anova(cells: &[Vec<Vec<f64>>]) -> Result<TwoWayAnovaResult> {
    let a = cells.len();
    let first_row = cells.first().ok_or(Error::InsufficientData)?;
    let b = first_row.len();
    if a < 2 || b < 2 {
        return Err(Error::InsufficientData);
    }
    let n = first_row.first().map_or(0, Vec::len);
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    for row in cells {
        if row.len() != b {
            return Err(Error::InvalidInput(
                "unbalanced design: factor B levels differ across factor A".to_owned(),
            ));
        }
        if row.iter().any(|cell| cell.len() != n) {
            return Err(Error::InvalidInput(
                "unbalanced design: unequal replicate counts".to_owned(),
            ));
        }
    }

    let (a_f, b_f, n_f) = (len_f64(a), len_f64(b), len_f64(n));
    let cells_per_a = b_f * n_f;
    let cells_per_b = a_f * n_f;
    let total = a_f * b_f * n_f;

    // Marginal sums: one total per factor-A level, and one per factor-B level.
    let row_sums: Vec<f64> = cells.iter().map(|row| row.iter().flatten().sum()).collect();
    let col_sums: Vec<f64> = (0..b)
        .map(|j| cells.iter().filter_map(|row| row.get(j)).flatten().sum())
        .collect();
    let grand_mean = row_sums.iter().sum::<f64>() / total;

    let row_means: Vec<f64> = row_sums.iter().map(|&s| s / cells_per_a).collect();
    let col_means: Vec<f64> = col_sums.iter().map(|&s| s / cells_per_b).collect();

    // SS_A = b·n · Σ_i (ȳ_i.. − ȳ...)²; SS_B = a·n · Σ_j (ȳ_.j. − ȳ...)².
    let ss_a = cells_per_a
        * row_means
            .iter()
            .map(|&rm| (rm - grand_mean) * (rm - grand_mean))
            .sum::<f64>();
    let ss_b = cells_per_b
        * col_means
            .iter()
            .map(|&cm| (cm - grand_mean) * (cm - grand_mean))
            .sum::<f64>();

    // SS_AB = n · Σ_ij (ȳ_ij. − ȳ_i.. − ȳ_.j. + ȳ...)²; SS_within = Σ (y − ȳ_ij.)².
    let mut ss_interaction = 0.0;
    let mut ss_within = 0.0;
    for (row, &rm) in cells.iter().zip(&row_means) {
        for (cell, &cm) in row.iter().zip(&col_means) {
            let cell_mean = cell.iter().sum::<f64>() / n_f;
            let interaction_dev = cell_mean - rm - cm + grand_mean;
            ss_interaction = interaction_dev.mul_add(interaction_dev, ss_interaction);
            ss_within += cell
                .iter()
                .map(|&y| (y - cell_mean) * (y - cell_mean))
                .sum::<f64>();
        }
    }
    ss_interaction *= n_f;

    if ss_within <= 0.0 {
        return Err(Error::DegenerateInput(
            "zero within-cell variation".to_owned(),
        ));
    }

    let (df_a, df_b) = (a - 1, b - 1);
    let df_interaction = df_a * df_b;
    let df_within = a * b * (n - 1);
    let ms_within = ss_within / len_f64(df_within);

    Ok(TwoWayAnovaResult {
        factor_a: effect_result(ss_a, df_a, ss_within, ms_within, df_within),
        factor_b: effect_result(ss_b, df_b, ss_within, ms_within, df_within),
        interaction: effect_result(
            ss_interaction,
            df_interaction,
            ss_within,
            ms_within,
            df_within,
        ),
        ss_within,
        df_within: len_f64(df_within),
    })
}

/// Builds the [`TestResult`] for one effect: `F = MS_effect / MS_within`, its
/// F-distribution upper-tail p-value, and partial `η² = SS / (SS + SS_within)`.
fn effect_result(
    ss: f64,
    df_effect: usize,
    ss_within: f64,
    ms_within: f64,
    df_within: usize,
) -> TestResult {
    let df_num = i64::try_from(df_effect).unwrap_or(i64::MAX);
    let df_den = i64::try_from(df_within).unwrap_or(i64::MAX);
    let f = (ss / len_f64(df_effect)) / ms_within;
    TestResult {
        statistic: f,
        p_value: f_upper_tail(f, df_num, df_den),
        log_p_value: Some(f_upper_log_tail(f, df_num, df_den)),
        df: Some(len_f64(df_effect)),
        effect_size: Some(ss / (ss + ss_within)),
    }
}

/// Kani formal-verification harnesses for the two-way ANOVA input-validation layer.
///
/// Compiled only under `cargo kani`. The harnesses feed grid descriptors that trip
/// the leading structural guards (an empty grid and a single factor-A level), so the
/// function returns `Err` before the sum-of-squares and F-tail interior.
#[cfg(kani)]
mod verification {
    // Harness bodies live in the sibling `two_way_anova_verification.rs` so this
    // file stays within the 500-line `tests/style.rs` limit. `super` here is the
    // `two_way_anova` module. Compiled only under `cargo kani`.
    include!("two_way_anova_verification.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single level on factor A is insufficient for a two-way ANOVA.
    #[test]
    fn one_level_factor_a_is_insufficient() {
        let cells = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
        assert!(
            matches!(two_way_anova(&cells), Err(Error::InsufficientData)),
            "one A level should be InsufficientData"
        );
    }

    /// Cells with unequal replicate counts are an invalid (unbalanced) design.
    #[test]
    fn unbalanced_cells_are_invalid() {
        let cells = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0]],
        ];
        assert!(
            matches!(two_way_anova(&cells), Err(Error::InvalidInput(_))),
            "unequal replicate counts should be InvalidInput"
        );
    }

    /// A single replicate per cell leaves the interaction inestimable.
    #[test]
    fn single_replicate_is_insufficient() {
        let cells = vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]];
        assert!(
            matches!(two_way_anova(&cells), Err(Error::InsufficientData)),
            "one replicate per cell should be InsufficientData"
        );
    }

    /// Asserts `got` is within `rel` relative error of `want` (`want != 0`).
    fn rel_close(got: f64, want: f64, rel: f64) -> bool {
        (got - want).abs() <= rel * want.abs()
    }

    /// Golden 2×2 design with 3 replicates, validated against statsmodels
    /// `anova_lm(ols("y ~ C(a)*C(b)"), typ=2)`. Generating snippet:
    /// `cells = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,13]]]`.
    #[test]
    fn golden_2x2x3_matches_statsmodels() -> Result<()> {
        let cells = vec![
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 13.0]],
        ];
        let r = two_way_anova(&cells)?;

        // statsmodels: ss_within = 10.666666666666666, df_within = 8.0.
        assert!(
            rel_close(r.ss_within, 10.666_666_666_666_666, 1e-12),
            "ss_within {}",
            r.ss_within
        );
        assert!(
            (r.df_within - 8.0).abs() < 1e-12,
            "df_within {}",
            r.df_within
        );

        // C(a): F=85.5625000000001, p=1.514373310913598e-05, df=1.
        assert!(
            rel_close(r.factor_a.statistic, 85.562_500_000_000_1, 1e-9),
            "F_a {}",
            r.factor_a.statistic
        );
        assert!(
            (r.factor_a.p_value - 1.514_373_310_913_598e-5).abs() < 1e-9,
            "p_a {}",
            r.factor_a.p_value
        );
        assert_eq!(r.factor_a.df, Some(1.0), "df_a");
        assert!(
            r.factor_a
                .effect_size
                .is_some_and(|e| rel_close(e, 0.914_495_657_982_632, 1e-9)),
            "eta_a {:?}",
            r.factor_a.effect_size
        );

        // C(b): F=22.562500000000032, p=0.0014452491304369458, df=1.
        assert!(
            rel_close(r.factor_b.statistic, 22.562_500_000_000_032, 1e-9),
            "F_b {}",
            r.factor_b.statistic
        );
        assert!(
            (r.factor_b.p_value - 0.001_445_249_130_436_945_8).abs() < 1e-9,
            "p_b {}",
            r.factor_b.p_value
        );
        assert_eq!(r.factor_b.df, Some(1.0), "df_b");

        // C(a):C(b): F=0.06250000000000204, p=0.8088874454935321, df=1.
        assert!(
            rel_close(r.interaction.statistic, 0.062_500_000_000_002_04, 1e-9),
            "F_ab {}",
            r.interaction.statistic
        );
        assert!(
            (r.interaction.p_value - 0.808_887_445_493_532_1).abs() < 1e-9,
            "p_ab {}",
            r.interaction.p_value
        );
        assert_eq!(r.interaction.df, Some(1.0), "df_ab");
        Ok(())
    }

    /// Golden 3×2 design with 4 replicates, validated against statsmodels
    /// `anova_lm(ols("y ~ C(a)*C(b)"), typ=2)`. Generating snippet:
    /// `cells = [[[2,3,5,4],[11,10,12,9]],[[1,0,2,3],[8,9,7,10]],`
    /// `[[9,11,10,12],[6,5,7,4]]]`.
    #[test]
    fn golden_3x2x4_matches_statsmodels() -> Result<()> {
        let cells = vec![
            vec![vec![2.0, 3.0, 5.0, 4.0], vec![11.0, 10.0, 12.0, 9.0]],
            vec![vec![1.0, 0.0, 2.0, 3.0], vec![8.0, 9.0, 7.0, 10.0]],
            vec![vec![9.0, 11.0, 10.0, 12.0], vec![6.0, 5.0, 7.0, 4.0]],
        ];
        let r = two_way_anova(&cells)?;

        // statsmodels: ss_within = 30.0, df_within = 18.0.
        assert!(
            rel_close(r.ss_within, 30.0, 1e-12),
            "ss_within {}",
            r.ss_within
        );
        assert!(
            (r.df_within - 18.0).abs() < 1e-12,
            "df_within {}",
            r.df_within
        );

        // C(a): F=11.20000000000003, p=0.0006918632457406173, df=2.
        assert!(
            rel_close(r.factor_a.statistic, 11.200_000_000_000_03, 1e-9),
            "F_a {}",
            r.factor_a.statistic
        );
        assert!(
            (r.factor_a.p_value - 0.000_691_863_245_740_617_3).abs() < 1e-9,
            "p_a {}",
            r.factor_a.p_value
        );
        assert_eq!(r.factor_a.df, Some(2.0), "df_a");
        assert!(
            r.factor_a
                .effect_size
                .is_some_and(|e| rel_close(e, 0.554_455_445_544_555_2, 1e-9)),
            "eta_a {:?}",
            r.factor_a.effect_size
        );

        // C(b): F=32.40000000000006, p=2.1301261274685675e-05, df=1.
        assert!(
            rel_close(r.factor_b.statistic, 32.400_000_000_000_06, 1e-9),
            "F_b {}",
            r.factor_b.statistic
        );
        assert!(
            (r.factor_b.p_value - 2.130_126_127_468_567_5e-5).abs() < 1e-9,
            "p_b {}",
            r.factor_b.p_value
        );
        assert_eq!(r.factor_b.df, Some(1.0), "df_b");
        assert!(
            r.factor_b
                .effect_size
                .is_some_and(|e| rel_close(e, 0.642_857_142_857_143_2, 1e-9)),
            "eta_b {:?}",
            r.factor_b.effect_size
        );

        // C(a):C(b): F=57.59999999999998, p=1.5028461477044567e-08, df=2.
        // This deep-tail p routes through the framework F CDF (regularized
        // incomplete beta); the measured relative error against statsmodels is
        // ~1.2e-9, so a 1e-8 relative check is honest and tight.
        assert!(
            rel_close(r.interaction.statistic, 57.599_999_999_999_98, 1e-9),
            "F_ab {}",
            r.interaction.statistic
        );
        assert!(
            rel_close(r.interaction.p_value, 1.502_846_147_704_456_7e-8, 1e-8),
            "p_ab {}",
            r.interaction.p_value
        );
        assert_eq!(r.interaction.df, Some(2.0), "df_ab");
        assert!(
            r.interaction
                .effect_size
                .is_some_and(|e| rel_close(e, 0.864_864_864_864_864_8, 1e-9)),
            "eta_ab {:?}",
            r.interaction.effect_size
        );
        Ok(())
    }

    /// Partial η² of each effect equals `SS_eff / (SS_eff + SS_within)`, where
    /// `SS_eff` is reconstructed from the reported `F`, its df, and `MS_within`.
    #[test]
    fn partial_eta_squared_matches_definition() -> Result<()> {
        let cells = vec![
            vec![vec![2.0, 3.0, 5.0, 4.0], vec![11.0, 10.0, 12.0, 9.0]],
            vec![vec![1.0, 0.0, 2.0, 3.0], vec![8.0, 9.0, 7.0, 10.0]],
            vec![vec![9.0, 11.0, 10.0, 12.0], vec![6.0, 5.0, 7.0, 4.0]],
        ];
        let r = two_way_anova(&cells)?;
        let ms_within = r.ss_within / r.df_within;
        for eff in [&r.factor_a, &r.factor_b, &r.interaction] {
            let df_eff = eff.df.ok_or(Error::InsufficientData)?;
            let ss_eff = eff.statistic * df_eff * ms_within;
            let want = ss_eff / (ss_eff + r.ss_within);
            assert!(
                eff.effect_size.is_some_and(|e| rel_close(e, want, 1e-9)),
                "partial eta^2 {:?} != {want}",
                eff.effect_size
            );
        }
        Ok(())
    }
}
