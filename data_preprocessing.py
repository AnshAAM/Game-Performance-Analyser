"""
data_preprocessing.py
=====================
Handles all data loading, cleaning, feature engineering, and preparation
for the Gaming Performance Analyzer Dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ─── Constants ───────────────────────────────────────────────────────────────
DATA_PATH = "csgo_pro_games_data.csv"

SKILL_LABELS = {
    (0,  50): "Beginner",
    (50, 75): "Intermediate",
    (75, 101): "Pro",
}

PERFORMANCE_WEIGHTS = {
    "game_rating": 0.40,
    "kd_ratio":    0.30,
    "kast":        0.15,
    "adr":         0.10,
    "fkdiff":      0.05,
}


# ─── Player record extraction ────────────────────────────────────────────────
def _extract_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the wide-format match DataFrame into one row per player per match.
    Returns a tidy DataFrame with columns:
        player_name, kills, deaths, assists, kast, kddiff, adr,
        fkdiff, game_rating, kills_headshot, assists_flash, team_side
    """
    records = []
    sides   = ["team1", "team2"]
    slots   = range(1, 6)        # p1 … p5

    for _, row in df.iterrows():
        for side in sides:
            for p in slots:
                prefix  = f"{side}_p{p}"
                name    = row.get(f"{prefix}_name")
                kills   = row.get(f"{prefix}_kills")
                deaths  = row.get(f"{prefix}_deaths")
                assists = row.get(f"{prefix}_assists")
                kast    = row.get(f"{prefix}_kast")
                kddiff  = row.get(f"{prefix}_kddiff")
                adr     = row.get(f"{prefix}_adr")
                fkdiff  = row.get(f"{prefix}_fkdiff")
                rating  = row.get(f"{prefix}_game_rating")
                hs      = row.get(f"{prefix}_kills_headshot")
                afl     = row.get(f"{prefix}_assists_flash")

                if pd.isna(name) or pd.isna(kills):
                    continue

                records.append({
                    "player_name":      name,
                    "kills":            kills,
                    "deaths":           deaths,
                    "assists":          assists,
                    "kast":             kast,
                    "kddiff":           kddiff,
                    "adr":              adr,
                    "fkdiff":           fkdiff,
                    "game_rating":      rating,
                    "kills_headshot":   hs,
                    "assists_flash":    afl,
                    "team_side":        side,
                })

    return pd.DataFrame(records)


# ─── Feature engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to a tidy player DataFrame.
    """
    df = df.copy()
    df["deaths"]  = df["deaths"].replace(0, 0.01)         # avoid ÷0
    df["kd_ratio"] = df["kills"] / df["deaths"]

    # Headshot percentage
    df["hs_pct"] = np.where(
        df["kills"] > 0,
        (df["kills_headshot"] / df["kills"]) * 100,
        0
    )

    # Raw performance score (0–100 scale per metric before weighting)
    scaler = MinMaxScaler(feature_range=(0, 100))

    score_cols = ["game_rating", "kd_ratio", "kast", "adr", "fkdiff"]
    df_score   = df[score_cols].copy()

    # Clip extreme outliers to 99th percentile before scaling
    for col in score_cols:
        upper = df_score[col].quantile(0.99)
        lower = df_score[col].quantile(0.01)
        df_score[col] = df_score[col].clip(lower, upper)

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_score),
        columns=[f"{c}_scaled" for c in score_cols],
        index=df.index,
    )
    df = pd.concat([df, df_scaled], axis=1)

    # Weighted performance score
    df["performance_score"] = (
        df["game_rating_scaled"] * PERFORMANCE_WEIGHTS["game_rating"] +
        df["kd_ratio_scaled"]    * PERFORMANCE_WEIGHTS["kd_ratio"]    +
        df["kast_scaled"]        * PERFORMANCE_WEIGHTS["kast"]        +
        df["adr_scaled"]         * PERFORMANCE_WEIGHTS["adr"]         +
        df["fkdiff_scaled"]      * PERFORMANCE_WEIGHTS["fkdiff"]
    )

    # Clip to [0, 100]
    df["performance_score"] = df["performance_score"].clip(0, 100)

    # Skill label
    df["skill_label"] = pd.cut(
        df["performance_score"],
        bins=[0, 50, 75, 100],
        labels=["Beginner", "Intermediate", "Pro"],
        include_lowest=True,
    )

    return df


# ─── Aggregate per-player stats ──────────────────────────────────────────────
def aggregate_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-match player rows into career averages.
    """
    agg = df.groupby("player_name").agg(
        matches        = ("kills", "count"),
        kills          = ("kills", "mean"),
        deaths         = ("deaths", "mean"),
        assists        = ("assists", "mean"),
        kast           = ("kast", "mean"),
        kddiff         = ("kddiff", "mean"),
        adr            = ("adr", "mean"),
        fkdiff         = ("fkdiff", "mean"),
        game_rating    = ("game_rating", "mean"),
        kills_headshot = ("kills_headshot", "mean"),
        assists_flash  = ("assists_flash", "mean"),
    ).reset_index()

    agg["deaths"] = agg["deaths"].replace(0, 0.01)
    agg["kd_ratio"] = agg["kills"] / agg["deaths"]
    agg["hs_pct"]   = np.where(
        agg["kills"] > 0,
        (agg["kills_headshot"] / agg["kills"]) * 100,
        0
    )
    return agg


# ─── Main pipeline ───────────────────────────────────────────────────────────
def load_and_preprocess(data_path: str = DATA_PATH):
    """
    Full pipeline:
      1. Load CSV
      2. Extract player records
      3. Engineer features on per-match rows
      4. Aggregate to career averages
      5. Re-compute performance score on aggregated data

    Returns
    -------
    player_df  : one row per player (career averages + engineered cols)
    per_match  : one row per player per match
    dataset_avg: dict of column averages (used by dashboard comparisons)
    """
    raw = pd.read_csv(data_path)

    # --- per-match player rows ---
    per_match = _extract_players(raw)
    per_match = per_match.dropna(
        subset=["kills", "deaths", "game_rating", "kast", "adr"]
    )
    per_match = engineer_features(per_match)

    # --- career aggregates ---
    player_df = aggregate_players(per_match)

    # Re-compute performance score on aggregates (uses same weights)
    scaler2    = MinMaxScaler(feature_range=(0, 100))
    score_cols = ["game_rating", "kd_ratio", "kast", "adr", "fkdiff"]
    df_s       = player_df[score_cols].copy()
    for col in score_cols:
        upper = df_s[col].quantile(0.99)
        lower = df_s[col].quantile(0.01)
        df_s[col] = df_s[col].clip(lower, upper)

    scaled = pd.DataFrame(
        scaler2.fit_transform(df_s),
        columns=[f"{c}_scaled" for c in score_cols],
        index=player_df.index,
    )
    player_df = pd.concat([player_df, scaled], axis=1)

    player_df["performance_score"] = (
        player_df["game_rating_scaled"] * PERFORMANCE_WEIGHTS["game_rating"] +
        player_df["kd_ratio_scaled"]    * PERFORMANCE_WEIGHTS["kd_ratio"]    +
        player_df["kast_scaled"]        * PERFORMANCE_WEIGHTS["kast"]        +
        player_df["adr_scaled"]         * PERFORMANCE_WEIGHTS["adr"]         +
        player_df["fkdiff_scaled"]      * PERFORMANCE_WEIGHTS["fkdiff"]
    ).clip(0, 100)

    player_df["skill_label"] = pd.cut(
        player_df["performance_score"],
        bins=[0, 50, 75, 100],
        labels=["Beginner", "Intermediate", "Pro"],
        include_lowest=True,
    )

    # --- dataset-wide averages for comparison widget ---
    dataset_avg = {
        "kills":       player_df["kills"].mean(),
        "deaths":      player_df["deaths"].mean(),
        "assists":     player_df["assists"].mean(),
        "kast":        player_df["kast"].mean(),
        "adr":         player_df["adr"].mean(),
        "kd_ratio":    player_df["kd_ratio"].mean(),
        "game_rating": player_df["game_rating"].mean(),
        "fkdiff":      player_df["fkdiff"].mean(),
        "hs_pct":      player_df["hs_pct"].mean(),
    }

    return player_df, per_match, dataset_avg


# ─── Scoring utility (used in live dashboard input) ──────────────────────────
def compute_user_score(
    kills: float,
    deaths: float,
    game_rating: float,
    kast: float,
    adr: float,
    fkdiff: float,
    player_df: pd.DataFrame,
) -> dict:
    """
    Given a user's raw stats, compute their normalised performance score
    by appending them to the existing player distribution and rescaling.
    Returns a dict with score, skill_label, and scaled individual metrics.
    """
    deaths = max(deaths, 0.01)
    kd_ratio = kills / deaths

    # Build a temp row aligned with player_df columns used for scoring
    score_cols = ["game_rating", "kd_ratio", "kast", "adr", "fkdiff"]
    existing   = player_df[score_cols].copy()

    user_row   = pd.DataFrame([{
        "game_rating": game_rating,
        "kd_ratio":    kd_ratio,
        "kast":        kast,
        "adr":         adr,
        "fkdiff":      fkdiff,
    }])

    combined = pd.concat([existing, user_row], ignore_index=True)

    # Clip at 99/1 pct of the existing distribution (not combined)
    for col in score_cols:
        upper = existing[col].quantile(0.99)
        lower = existing[col].quantile(0.01)
        combined[col] = combined[col].clip(lower, upper)

    scaler  = MinMaxScaler(feature_range=(0, 100))
    scaled  = scaler.fit_transform(combined)
    user_sc = scaled[-1]   # last row = user

    score = (
        user_sc[0] * PERFORMANCE_WEIGHTS["game_rating"] +
        user_sc[1] * PERFORMANCE_WEIGHTS["kd_ratio"]    +
        user_sc[2] * PERFORMANCE_WEIGHTS["kast"]        +
        user_sc[3] * PERFORMANCE_WEIGHTS["adr"]         +
        user_sc[4] * PERFORMANCE_WEIGHTS["fkdiff"]
    )
    score = float(np.clip(score, 0, 100))

    if score < 50:
        label = "Beginner"
    elif score < 75:
        label = "Intermediate"
    else:
        label = "Pro"

    return {
        "performance_score":    score,
        "skill_label":          label,
        "kd_ratio":             kd_ratio,
        "game_rating_scaled":   float(user_sc[0]),
        "kd_ratio_scaled":      float(user_sc[1]),
        "kast_scaled":          float(user_sc[2]),
        "adr_scaled":           float(user_sc[3]),
        "fkdiff_scaled":        float(user_sc[4]),
    }


if __name__ == "__main__":
    player_df, per_match, dataset_avg = load_and_preprocess()
    print("Player records:", len(player_df))
    print("Per-match rows:", len(per_match))
    print("Skill distribution:\n", player_df["skill_label"].value_counts())
    print("\nDataset averages:", dataset_avg)
