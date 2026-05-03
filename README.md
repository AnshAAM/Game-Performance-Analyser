# 🎯 Gaming Performance Analyzer Dashboard

A project that analyzes CS:GO gaming performance using
**XGBoost**, **KMeans clustering**, and an interactive **Streamlit** dashboard.

---

## 📁 Project Structure

```
gaming_analyzer/
├── csgo_pro_games_data.csv   ← your dataset (place here)
├── data_preprocessing.py     ← feature engineering & scoring
├── model_training.py         ← XGBoost + KMeans training
├── app.py                    ← Streamlit dashboard (main file)
├── requirements.txt
├── README.md
└── models/                   ← auto-created after training
    ├── xgb_classifier.pkl
    ├── kmeans.pkl
    ├── cluster_scaler.pkl
    ├── label_encoder.pkl
    └── cluster_label_map.pkl
```

---

## ⚡ Quick Start (VS Code)

### 1 — Create & activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Train the models (run once)
```bash
python model_training.py
```
You should see training accuracy and cluster distribution printed to the console.
A `models/` folder will be created automatically.

### 4 — Launch the dashboard
```bash
streamlit run app.py
```
The browser will open at `http://localhost:8501`.

---

## 🗺 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Dataset overview, skill distribution, rating histogram, weight breakdown |
| 📥 Input Stats | Slider-based form to enter your per-game averages |
| 📊 Analysis | Performance gauge, skill label, ML prediction, playstyle badge, metric progress bars |
| 📈 Visualizations | Radar chart, grouped bar chart (you vs avg), K/D histogram, scatter plot |
| 💡 Recommendations | Personalised tips for every weak metric and playstyle type |
| 🏆 Leaderboard | Session-state leaderboard with ranking bar chart |

---

## 🧠 ML Models

### XGBoost Classifier
- **Task**: Predict skill level — *Beginner / Intermediate / Pro*
- **Features**: kills, deaths, assists, KAST, KD diff, ADR, FK diff, game rating, K/D ratio, HS%
- **Training split**: 80 / 20 stratified

### KMeans Clustering (k=3)
- **Task**: Detect playstyle — *Aggressive / Balanced / Defensive*
- **Features**: kills, deaths, K/D ratio, ADR, FK diff, KAST
- **Label assignment**: automatic from centroid kill-ranking

---

## 📊 Performance Score Formula

```
Score = game_rating × 0.40
      + kd_ratio    × 0.30
      + kast        × 0.15
      + adr         × 0.10
      + fkdiff      × 0.05
```
Each metric is first normalized to [0, 100] using MinMaxScaler fit on the dataset.

