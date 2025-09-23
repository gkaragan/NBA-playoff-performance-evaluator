from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import numpy as np
import statsmodels.api as sm


# Enhanced stat weights - adjusted for playoff importance
STAT_WEIGHTS = {
    # Scoring & Efficiency (60% total) - most important in playoffs
    "PTS": 0.4,
    "TS_PCT": 0.20,
    # Playmaking & Ball Security (22.5% total)
    "AST": 0.175,
    "PCT_TOV": 0.05,  # Turnovers more costly in playoffs
    # Rebounding (12.5% total)
    "REB": 0.125,
    # Defense/Hustle (5% total)
    "STL": 0.025,  # Reduced weights because defensive stats are noisy in small playoff samples
    "BLK": 0.025,
}

# Tier-based adjustments
TIER_ADJUSTMENTS = {
    "Tier1": {
        "improvement_bonus": 1.3,  # Reward stars who elevate further
        "decline_penalty": 0.7,  # Reduce punishment if still elite
        "elite_threshold": 0.9,  # What percentile counts as "elite" performance
    },
    "Tier2": {"improvement_bonus": 1.2, "decline_penalty": 0.8, "elite_threshold": 0.8},
    "Tier3": {"improvement_bonus": 1.1, "decline_penalty": 0.9, "elite_threshold": 0.7},
    "Tier4": {"improvement_bonus": 0.8, "decline_penalty": 1.0, "elite_threshold": 0.5},
}  # Reduced Tier4 improvement bonus because these players have lower baselines


def main():
    playoff_df = get_merged_df("Playoffs")
    reg_season_df = get_merged_df("Regular Season")
    po_rs_merged_df = reg_season_df.merge(
        playoff_df.drop(columns=["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]),
        on="PLAYER_ID",
        how="inner",
        suffixes=("_RS", "_PO"),
    )
    diffs_merged_df = diff_finder(po_rs_merged_df)

    # Get the df of players ranked by impact score
    results_df = calculate_playoff_impact_scores(diffs_merged_df)

    # Rankings csv
    results_df.to_csv("rankings.csv", index=False)
    print("See 'rankings.csv' for complete rankings.")


def get_merged_df(season_type):  # season_type == 'Regular Season' or 'Playoffs'
    # 1. Get base stats (points, rebounds, assists, etc.) ===
    df_base = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2024-25",
        season_type_all_star=season_type,
        league_id_nullable="00",  # NBA only
        per_mode_detailed="PerGame",  # Per-game averages
        measure_type_detailed_defense="Base",
    )

    df_base = df_base.get_data_frames()[0]
    df_base = df_base[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "GP",
            "MIN",
            "FGA",
            "REB",
            "AST",
            "STL",
            "BLK",
            "PTS",
        ]
    ]

    # 2. get advanced stats
    df_adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2024-25",
        season_type_all_star=season_type,
        league_id_nullable="00",  # NBA only
        per_mode_detailed="PerGame",  # Per-game averages
        measure_type_detailed_defense="Advanced",
    )
    df_adv = df_adv.get_data_frames()[0]
    df_adv = df_adv[
        [
            "PLAYER_ID",
            "TS_PCT",
        ]
    ]

    # 3. Get PCT_TOV stat - More comprehensive than TOV
    df_tov_pct = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2024-25",
        season_type_all_star=season_type,
        league_id_nullable="00",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Usage",
    ).get_data_frames()[0]
    df_tov_pct = df_tov_pct[["PLAYER_ID", "PCT_TOV"]]

    # Return merged df containing all relevant stats
    return df_base.merge(df_adv, on="PLAYER_ID", how="inner").merge(
        df_tov_pct, on="PLAYER_ID", how="inner"
    )


def diff_finder(merged_df):
    """
    Compute playoff-regular season differences for each stat.

    Generates `DIFF_<stat>` columns capturing changes in performance.

    Args:
        diffs_merged_df (pd.DataFrame): Input DataFrame with RS and PO stat columns.

    Returns:
        merged_df (pd.DataFrame): Original DataFrame with difference columns added.
    """

    # Split dataframe into PO and RS blocks
    po = merged_df.filter(regex=r"_PO$").copy()
    rs = merged_df.filter(regex=r"_RS$").copy()

    # Take only columns corresponding to numeric data
    po = po.select_dtypes(include="number")
    rs = rs.select_dtypes(include="number")

    # Normalize column names (drop the suffix)
    po.columns = po.columns.str.replace(r"_PO$", "", regex=True)
    rs.columns = rs.columns.str.replace(r"_RS$", "", regex=True)

    # Align columns to caclulate differences (although, columns should be aligned already by construction)
    common_cols = po.columns.intersection(rs.columns).difference(["GP"])
    diff_df = (po[common_cols] - rs[common_cols]).round(4)

    # Prefix the columns in the new dfs
    diff_df = diff_df.add_prefix("DIFF_")

    # Give diff_df unique identifiers (for standalone use)
    identifiers = merged_df[["PLAYER_ID", "PLAYER_NAME"]].copy()
    diff_df = pd.concat([identifiers, diff_df], axis=1)

    diff_df[
        "DIFF_PCT_TOV"
    ] *= -1  # Keep consistency with other stats: greater diff = greater improvement

    new_merged_df = pd.concat(
        [merged_df, diff_df.drop(columns=["PLAYER_ID", "PLAYER_NAME"])],
        axis=1,
    )

    return new_merged_df


def calculate_playoff_impact_scores(diffs_merged_df):
    """
    Calculate overall playoff impact scores for all players.

    Pipeline:
    1. Precompute robust residual z-scores for each stat (baseline-adjusted).
    2. Compute stat-specific z-thresholds (80th, 90th percentiles of |z|).
    3. For each player, calculate contextual impact per stat using:
    - Tier adjustments (different treatment for stars vs. role players).
    - Elite-performance exceptions (softened penalties for still-dominant stars).
    - Weighted contributions of each stat.
    4. Aggregate into a single impact score per player.

    Args:
        diffs_merged_df (pd.DataFrame): DataFrame with RS, PO, and DIFF stats.

    Returns:
        pd.DataFrame: DataFrame of players with contextual playoff impact scores,
        sorted by impact (descending = biggest risers).
    """

    # Filter players based on minimum requirements
    # We only care about 'meaningful' risers/droppers
    filtered_df = diffs_merged_df[
        (diffs_merged_df["MIN_RS"] >= 20)  # 20+ min in regular season
        & (diffs_merged_df["MIN_PO"] >= 30)  # 30+ min in playoffs
        & (diffs_merged_df["GP_RS"] >= 15)  # 15+ games regular season
        & (diffs_merged_df["GP_PO"] >= 4)  # 4+ games playoffs
    ].copy()

    # Reset index
    filtered_df = filtered_df.reset_index(drop=True)

    # Precompute residual z for each stat once
    residual_z = {}
    for s in STAT_WEIGHTS:
        residual_z[s] = baseline_adjusted_residuals(filtered_df, s).to_numpy()

    # Compute percentile thresholds per stat
    z_thresholds = {}
    for s, z_vals in residual_z.items():
        z_abs = np.abs(z_vals)
        z_thresholds[s] = {
            "80": np.percentile(z_abs, 80),
            "90": np.percentile(z_abs, 90),
        }

    # Create player tier classifications based on regular season performance
    player_tiers, comp_scores = classify_player_tiers(filtered_df)

    # Calculate context-aware impact scores
    impact_scores = []

    for idx, player in filtered_df.iterrows():
        total_score = 0
        player_details = {
            "PLAYER_ID": player["PLAYER_ID"],
            "PLAYER_NAME": player["PLAYER_NAME"],
            "TEAM_ABBREVIATION": player["TEAM_ABBREVIATION"],
            "PLAYER_RS_TIER": player_tiers[idx],
            "COMPOSITE_TIER_SCORE": comp_scores[idx],
            "MIN_RS": round(player["MIN_RS"], 1),
            "MIN_PO": round(player["MIN_PO"], 1),
            "GP_RS": int(player["GP_RS"]),
            "GP_PO": int(player["GP_PO"]),
            "FGA_PO": player["FGA_PO"],
        }

        # Calculate sample size reliability factor
        reliability_factor = calculate_sample_reliability(
            player["GP_PO"], player["MIN_PO"], player_tiers[idx]
        )
        player_details["reliability_factor"] = reliability_factor

        for stat, weight in STAT_WEIGHTS.items():
            diff_col = f"DIFF_{stat}"
            rs_col = f"{stat}_RS"
            po_col = f"{stat}_PO"

            if all(col in player.index for col in [diff_col, rs_col, po_col]):
                change = player[diff_col]
                rs_value = player[rs_col]
                po_value = player[po_col]

                # Context-aware impact calculation
                stat_impact = calculate_contextual_impact(
                    stat,
                    player_tiers[idx],
                    filtered_df,
                    weight,
                    idx,
                    residual_z,
                    z_thresholds,
                )

                # Apply reliability adjustment
                stat_impact *= reliability_factor

                total_score += stat_impact

                # Store details
                player_details[f"{stat}_RS"] = round(rs_value, 3)
                player_details[f"{stat}_PO"] = round(po_value, 3)
                player_details[f"{stat}_change"] = round(change, 3)
                # Add standardized z-score for transparency
                player_details[f"{stat}_z"] = round(residual_z[stat][idx], 3)
                player_details[f"{stat}_impact"] = round(stat_impact, 4)

        player_details["total_impact_score"] = round(total_score, 4)
        impact_scores.append(player_details)

    # Convert to DataFrame and rank
    results_df = pd.DataFrame(impact_scores)
    results_df = results_df.sort_values("total_impact_score", ascending=False)
    results_df["rank"] = range(1, len(results_df) + 1)

    ### DIAGNOSTICS SECTION ###
    # The below code is to check the validity of the regression model for each stat

    """# Test for given stat
    X = results_df[["PTS_RS"]].to_numpy()
    y = results_df["PTS_PO"].to_numpy()

    # More minutes played -> treat data as more reliable 
    reliability = (results_df["GP_PO"] * results_df["MIN_PO"]) / 100
    weights = np.sqrt(reliability)

    res = fit_wls(y, X, weights=weights)


    print("R^2:", res["r_squared"])
    print("Adj R^2:", res["adj_r_squared"])
    print("AIC / BIC:", res["aic"], res["bic"])
    print("Params (const, ...):", res["params"])
    print("Std err:", res["stderr"])
    print("P-values:", res["pvalues"])
    print("Breusch-Pagan test:", res["bp_test"])
    print(res["summary"])   # full pretty summary

    # Display robust results
    if res["robust_stderr"] is not None:
        print("\n=== ROBUST STANDARD ERRORS ===")
        print("Regular SE:", res["stderr"])
        print("Robust SE:", res["robust_stderr"])
        print("Regular P-values:", res["pvalues"])
        print("Robust P-values:", res["robust_pvalues"])
        
        # Compare the key coefficient (slope)
        print(f"\nSlope Coefficient: {res['params'][1]:.4f}")
        print(f"Regular SE: {res['stderr'][1]:.4f}")
        print(f"Robust SE: {res['robust_stderr'][1]:.4f}")
        print(f"Regular p-value: {res['pvalues'][1]:.2e}")
        print(f"Robust p-value: {res['robust_pvalues'][1]:.2e}")


    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    preds = res["preds"]
    resid = res["residuals"]

    # Residuals vs fitted
    plt.scatter(preds, resid)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.show()

    # QQ plot
    sm.qqplot(resid, line="45")
    plt.title("QQ plot of residuals")
    plt.show()

    # Histogram of residuals
    plt.hist(resid, bins=25)
    plt.title("Residual histogram")
    plt.show()

    # Plot scatter with WLS line and label risers/droppers
    # Only for PTS
    plot_pts_scatter_with_wls(results_df, res, top_n=2, bottom_n=2, save_path="pts_scatter.png")"""

    ### DIAGNOSTICS END ###

    return results_df


def fit_wls(y, X, weights=None, add_constant=True):
    """
    Fit an Ordinary Least Squares (OLS) or Weighted Least Squares (WLS) regression
    and return model results along with diagnostics.

    Args:
        y (array-like): Dependent variable.
        X (array-like): Independent variable(s). If 1D, reshaped to 2D.
        weights (array-like, optional): Observation weights for WLS.
            If None, falls back to unweighted OLS.
        add_constant (bool, default=True): Whether to include an intercept term.


    Returns:
        dict: Dictionary of regression outputs with keys:
            - "model": Fitted statsmodels RegressionResultsWrapper (OLS or WLS).
            - "intercept": Intercept term (float).
            - "coefs": Array of fitted coefficients for predictors.
            - "params": Full parameter array (intercept + coefficients).
            - "pvalues": p-values for coefficients.
            - "stderr": Standard errors for coefficients.
            - "r_squared": R² value of the model.
            - "adj_r_squared": Adjusted R² value.
            - "aic": Akaike Information Criterion.
            - "bic": Bayesian Information Criterion.
            - "preds": Fitted values.
            - "residuals": Residuals (y - preds).
            - "bp_test": Breusch-Pagan test results dict
                {"lm_stat", "lm_pvalue", "f_stat", "f_pvalue"}.
            - "summary": Full regression summary (text).
            - "robust_stderr": Robust (HC3) standard errors, if available.
            - "robust_pvalues": Robust (HC3) p-values, if available.
            - "model_robust": RegressionResults with robust covariance, if computed.

    Notes:
        - Uses HC3 robust covariance estimator for robust standard errors/p-values.
        - If weights are provided, the model is fit with WLS; otherwise OLS is used.
        - Breusch-Pagan test is included to check for heteroskedasticity.
    """

    from statsmodels.stats.diagnostic import het_breuschpagan

    # Ensure numpy arrays and 2D X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if add_constant:
        X2 = sm.add_constant(X, has_constant="add")
    else:
        X2 = X
    y = np.asarray(y)

    # Choose OLS vs WLS
    if weights is not None:
        w = np.asarray(weights)
        # Statsmodels expects a weight vector
        model_res = sm.WLS(y, X2, weights=w).fit()

    else:
        model_res = sm.OLS(y, X2).fit()

    params = model_res.params  # Array (const, coef1, coef2, ...)
    intercept = float(params[0])
    coefs = np.asarray(params[1:])

    # Predictions and residuals
    preds = model_res.predict(X2)
    residuals = model_res.resid

    # Breusch-Pagan heteroskedasticity test (LM and F stat)
    try:
        bp = het_breuschpagan(residuals, model_res.model.exog)
        bp_test = {
            "lm_stat": float(bp[0]),
            "lm_pvalue": float(bp[1]),
            "f_stat": float(bp[2]),
            "f_pvalue": float(bp[3]),
        }
    except Exception:
        bp_test = None

    # Add robust results
    try:
        model_robust = model_res.get_robustcov_results(cov_type="HC3")
        robust_stderr = model_robust.bse
        robust_pvalues = model_robust.pvalues
    except Exception:
        robust_stderr = None
        robust_pvalues = None

    out = {
        "model": model_res,
        "intercept": intercept,
        "coefs": coefs,
        "params": params,
        "pvalues": getattr(model_res, "pvalues", None),
        "stderr": getattr(model_res, "bse", None),
        "r_squared": float(getattr(model_res, "rsquared", float("nan"))),
        "adj_r_squared": float(getattr(model_res, "rsquared_adj", float("nan"))),
        "aic": float(getattr(model_res, "aic", float("nan"))),
        "bic": float(getattr(model_res, "bic", float("nan"))),
        "preds": preds,
        "residuals": residuals,
        "bp_test": bp_test,
        "summary": model_res.summary(),
        "robust_stderr": robust_stderr,
        "robust_pvalues": robust_pvalues,
        "model_robust": model_robust if robust_stderr is not None else None,
    }

    return out


def classify_player_tiers(df):
    """
    Classify players into tiers based on regular season performance levels.
    This helps provide context for evaluating improvements/declines.
    """

    # Classify players based on RS performance levels
    comp = (
        df["PTS_RS"] * 0.50
        + df["AST_RS"] * 0.25
        + df["REB_RS"] * 0.15
        + df["TS_PCT_RS"] * 100 * 0.1
    )

    percentiles = comp.rank(pct=True) * 100

    tiers = pd.Series(
        np.select(
            [percentiles >= 85, percentiles >= 70, percentiles >= 50],
            ["Tier1", "Tier2", "Tier3"],
            default="Tier4",
        ),
        index=df.index,
    )

    return tiers, round(comp, 3)


def calculate_contextual_impact(
    stat, player_tier, df, weight, player_idx, residual_z, z_thresholds
):
    """
    Calculate contextual impact score for a single stat and player.

    Incorporates:
    - Residual z-score (baseline-adjusted playoff change).
    - Player tier with tier-specific bonuses/penalties.
    - Playoff percentile performance (to soften penalties for elite performance).
    - Stat weight (importance of each metric in overall impact).

    Args:
        stat (str): Stat name.
        change (float): Raw playoff change for the stat.
        player_tier (str): Player’s tier label (e.g., 'Tier1').
        df (pd.DataFrame): Full DataFrame with player stats.
        weight (float): Weight of the stat in overall impact.
        player_idx (int): Row index of the player in df.
        residual_z (dict): Mapping of stat → z-scores for all players.
        z_thresholds (dict): Mapping of stat → percentile thresholds for |z|.

    Returns:
        float: Weighted contextual impact score for this stat and player.
    """

    # Get stat-specific parameters
    po_col = f"{stat}_PO"

    # Calculate percentile of playoff performance
    if stat == "PCT_TOV":  # Lower is better
        po_percentile = df[po_col].rank(ascending=False, pct=True).iloc[player_idx]
    else:
        po_percentile = df[po_col].rank(pct=True).iloc[player_idx]

    # Baseline-adjusted residual (robust z)
    z_score = residual_z[stat][player_idx]

    adjustment = TIER_ADJUSTMENTS[player_tier]

    # Pull dynamic thresholds for this stat
    stat_thresh = z_thresholds.get(stat)

    if stat in ("STL", "BLK"):
        impact = z_score  # Further control for STL/BLK small sample noise by bypassing tier inflation
    else:
        # Contextual scoring logic
        if z_score > 0:  # Improvement
            # Base improvement score
            impact = z_score * adjustment["improvement_bonus"]

            # Additional bonus for meaningful improvements by higher-tier players
            if player_tier in ["Tier1", "Tier2"] and z_score > stat_thresh["80"]:
                impact *= 1.3
            elif player_tier == "Tier3" and z_score > stat_thresh["90"]:
                impact *= 1.3

        else:  # Decline
            # Check if player is still performing at elite level despite decline
            if po_percentile >= adjustment["elite_threshold"]:
                # Reduced penalty for elite performance despite decline
                impact = z_score * adjustment["decline_penalty"]
            else:
                # Full penalty for poor playoff performance
                impact = z_score

            # Extra penalty for significant declines by high RS performers
            # Tier-specific decline penalties with different thresholds
            if player_tier in ["Tier1", "Tier2"] and abs(z_score) > stat_thresh["80"]:
                impact *= 1.3  # Harsh penalty for stars
            elif player_tier == "Tier3" and abs(z_score) > stat_thresh["90"]:
                impact *= 1.3  # Same penalty for lower RS performers, higher threshold
    if stat == "TS_PCT":  # TS_PCT correction
        fga_po = df.iloc[player_idx]["FGA_PO"]

        if z_score > 0 and fga_po < 10:
            # Down-weight low-volume improvements
            impact *= 0.7
        elif z_score <= 0 and fga_po < 10:
            # Full penalty for low-volume decliners
            impact = z_score

    # Apply base weight
    impact *= weight

    return impact


def calculate_sample_reliability(gp_po, min_po, player_tier):
    """
    Calculate reliability adjustment based on playoff sample size.
    More games = more reliable data = less adjustment needed.
    """

    total_minutes = gp_po * min_po

    base = np.clip(total_minutes / 400, 0.0, 1.0)  # 0–1 scaling
    tier_bonus = {"Tier1": 0.15, "Tier2": 0.1, "Tier3": 0.05, "Tier4": 0.0}.get(
        player_tier, 0.0
    ) # Treat the performance of higher tier players as more reliable 

    return round(min(1.0, base + tier_bonus), 2)


def baseline_adjusted_residuals(df, stat):
    """
    Compute robust residual z-scores for a given stat.

    Fits a regression of playoff performance (stat_PO) on regular-season baseline performance
    to account for regression-to-the-mean and baseline effects.
    Residuals are then scaled using a robust z-score (median and MAD) to reduce sensitivity to outliers.
    """

    po_col = f"{stat}_PO"
    rs_col = f"{stat}_RS"

    X = df[[rs_col]].to_numpy().copy()
    # Reliability weights so players with the most min played influence the regression line more
    reliability = np.minimum((df["GP_PO"] * df["MIN_PO"]) / 100, 5.0)
    weights = np.sqrt(reliability)
    y = df[po_col].to_numpy().copy()

    # Separate treatment since lower is better for PCT_TOV
    if stat == "PCT_TOV":
        X = 1 - X
        y = 1 - y
        # Fit weighted regression
        res = fit_wls(y, X, weights=weights)
    else:
        res = fit_wls(y, X, weights=weights)

    # Residuals = actual - expected
    residuals = res["residuals"]

    # Robust z-scores
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    robust_z = (residuals - med) / (1.4826 * mad + 1e-6)

    # Cap z to reduce single stat domination in the scoring
    robust_z = np.clip(robust_z, -2.5, 2.5)

    return pd.Series(robust_z, index=df.index)


# Use in diagnostics section
def plot_pts_scatter_with_wls(results_df, res, top_n=2, bottom_n=2, save_path=None):
    """
    Scatterplot of PTS_PO vs PTS_RS with WLS regression line from fit_wls.
    Labels top_n risers and bottom_n droppers by impact score.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with PLAYER_NAME, PTS_RS, PTS_PO, and PTS_impact columns.
    res : dict
        Output dictionary from fit_wls() for the PTS model.
    top_n, bottom_n : int
        Number of risers/droppers to label.
    save_path : str or None
        If provided, saves the figure to this file (e.g., "pts_scatter.png").
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    # Scatter points
    plt.scatter(
        results_df["PTS_RS"], results_df["PTS_PO"],
        alpha=0.6, label="Players"
    )

    # Regression line from your WLS model
    x_vals = np.linspace(results_df["PTS_RS"].min(), results_df["PTS_RS"].max(), 100)
    X_line = sm.add_constant(x_vals)
    y_vals = res["model"].predict(X_line)
    plt.plot(x_vals, y_vals, color="red", lw=2, label="WLS fit")

    # Label top risers and droppers
    top_risers = results_df.nlargest(top_n, "PTS_impact")
    top_droppers = results_df.nsmallest(bottom_n, "PTS_impact")

    for _, row in pd.concat([top_risers, top_droppers]).iterrows():
        plt.text(
            row["PTS_RS"], row["PTS_PO"], row["PLAYER_NAME"],
            fontsize=5.5, weight="bold", color="darkblue"
        )

    plt.xlabel("Regular Season PPG")
    plt.ylabel("Playoff PPG")
    plt.title("Playoff vs. Regular Season Scoring")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()