"""
model_training.py
=================
Trains, evaluates, and saves the ML models used by the dashboard:
  1. XGBoost Classifier  – predicts skill level (Beginner / Intermediate / Pro)
  2. KMeans Clustering   – detects playstyle  (Aggressive / Balanced / Defensive)

Run this script once before launching the Streamlit app:
    python model_training.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection  import train_test_split
from sklearn.metrics          import classification_report, accuracy_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from xgboost                  import XGBClassifier
from sklearn.cluster          import KMeans

from data_preprocessing       import load_and_preprocess

# ─── Paths ───────────────────────────────────────────────────────────────────
MODELS_DIR    = "models"
XGB_PATH      = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
KMEANS_PATH   = os.path.join(MODELS_DIR, "kmeans.pkl")
SCALER_PATH   = os.path.join(MODELS_DIR, "cluster_scaler.pkl")
ENCODER_PATH  = os.path.join(MODELS_DIR, "label_encoder.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Features ────────────────────────────────────────────────────────────────
CLASSIFICATION_FEATURES = [
    "kills", "deaths", "assists",
    "kast", "kddiff", "adr", "fkdiff",
    "game_rating", "kd_ratio", "hs_pct",
]

CLUSTER_FEATURES = [
    "kills",    # aggression indicator
    "deaths",   # recklessness
    "kd_ratio", # efficiency
    "adr",      # damage output
    "fkdiff",   # entry-fragger tendency
    "kast",     # reliability
]

# Cluster label mapping: assigned by inspecting centroid ordering
CLUSTER_STYLE_MAP = {
    0: "Aggressive",
    1: "Defensive",
    2: "Balanced",
}


# ─── XGBoost training ────────────────────────────────────────────────────────
def train_xgb(player_df: pd.DataFrame):
    """Train and save the XGBoost skill-level classifier."""

    df = player_df.dropna(subset=CLASSIFICATION_FEATURES + ["skill_label"])
    X  = df[CLASSIFICATION_FEATURES].values
    y_raw = df["skill_label"].astype(str).values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators       = 300,
        max_depth          = 5,
        learning_rate      = 0.08,
        subsample          = 0.80,
        colsample_bytree   = 0.80,
        use_label_encoder  = False,
        eval_metric        = "mlogloss",
        random_state       = 42,
        n_jobs             = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set              = [(X_test, y_test)],
        verbose               = False,
    )

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[XGBoost] Test accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_))

    joblib.dump(model, XGB_PATH)
    joblib.dump(le,    ENCODER_PATH)
    print(f"  Saved model  → {XGB_PATH}")
    print(f"  Saved encoder→ {ENCODER_PATH}")
    return model, le


# ─── KMeans clustering ───────────────────────────────────────────────────────
def train_kmeans(player_df: pd.DataFrame):
    """Train and save the KMeans playstyle clustering model."""

    df     = player_df.dropna(subset=CLUSTER_FEATURES)
    X_raw  = df[CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    # Elbow method check (k=3 chosen by domain knowledge)
    model  = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X)

    # Assign interpretable labels by centroid means
    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=CLUSTER_FEATURES,
    )
    # Rank clusters by kills (highest = most aggressive)
    order  = centers["kills"].argsort().values  # ascending idx
    label_map = {
        int(order[0]): "Defensive",
        int(order[1]): "Balanced",
        int(order[2]): "Aggressive",
    }

    labels     = model.labels_
    style_col  = [label_map[l] for l in labels]
    player_df  = player_df.copy()
    player_df.loc[df.index, "playstyle"] = style_col

    print(f"\n[KMeans] Cluster distribution:")
    print(pd.Series(style_col).value_counts())
    print("\n  Cluster centroids (inverse-scaled):")
    print(centers.round(2))

    joblib.dump(model,  KMEANS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    # Save the live label_map too so inference is deterministic
    joblib.dump(label_map, os.path.join(MODELS_DIR, "cluster_label_map.pkl"))
    print(f"  Saved KMeans → {KMEANS_PATH}")
    print(f"  Saved scaler → {SCALER_PATH}")
    return model, scaler, label_map


# ─── Inference helpers (used by Streamlit app) ───────────────────────────────
def predict_skill(kills, deaths, assists, kast, kddiff, adr, fkdiff,
                   game_rating, kd_ratio, hs_pct):
    """Load the saved XGBoost model and return the predicted skill label."""
    model = joblib.load(XGB_PATH)
    le    = joblib.load(ENCODER_PATH)
    X     = np.array([[kills, deaths, assists, kast, kddiff,
                        adr, fkdiff, game_rating, kd_ratio, hs_pct]])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    prob_dict = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}
    return label, prob_dict


def predict_playstyle(kills, deaths, kd_ratio, adr, fkdiff, kast):
    """Load the saved KMeans model and return the playstyle cluster label."""
    model     = joblib.load(KMEANS_PATH)
    scaler    = joblib.load(SCALER_PATH)
    label_map = joblib.load(os.path.join(MODELS_DIR, "cluster_label_map.pkl"))
    X         = np.array([[kills, deaths, kd_ratio, adr, fkdiff, kast]])
    X_sc      = scaler.transform(X)
    cluster   = int(model.predict(X_sc)[0])
    return label_map.get(cluster, "Balanced")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading and preprocessing data …")
    player_df, _, _ = load_and_preprocess()
    print(f"  {len(player_df)} unique players loaded.")

    train_xgb(player_df)
    train_kmeans(player_df)
    print("\n✅  All models saved to ./models/  — you can now run the Streamlit app.")
