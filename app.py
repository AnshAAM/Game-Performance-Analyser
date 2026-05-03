"""
app.py
======
Gaming Performance Analyzer Dashboard
Streamlit multi-page application powered by XGBoost + KMeans ML models.

Run:
    streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express       as px
from   datetime import datetime

import joblib
from   data_preprocessing import (
    load_and_preprocess,
    compute_user_score,
    PERFORMANCE_WEIGHTS,
)
from   model_training import (
    predict_skill,
    predict_playstyle,
    CLASSIFICATION_FEATURES,
    CLUSTER_FEATURES,
    MODELS_DIR,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "CS:GO Performance Analyzer",
    page_icon  = "🎯",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0e1117;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
h1, h2, h3, h4 { color: #e6edf3 !important; }
p, li, label    { color: #8b949e !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background   : #161b22;
    border       : 1px solid #30363d;
    border-radius: 10px;
    padding      : 14px 18px;
}
[data-testid="stMetricValue"]  { color: #58a6ff !important; font-size:1.6rem !important; }
[data-testid="stMetricLabel"]  { color: #8b949e !important; }

/* ── Skill badge ── */
.badge {
    display      : inline-block;
    padding      : 4px 14px;
    border-radius: 20px;
    font-weight  : 700;
    font-size    : 0.85rem;
    margin       : 2px;
}
.badge-beginner     { background:#1c2128; color:#ff7b72; border:1px solid #ff7b72; }
.badge-intermediate { background:#1c2128; color:#e3b341; border:1px solid #e3b341; }
.badge-pro          { background:#1c2128; color:#3fb950; border:1px solid #3fb950; }
.badge-aggressive   { background:#1c2128; color:#f78166; border:1px solid #f78166; }
.badge-defensive    { background:#1c2128; color:#79c0ff; border:1px solid #79c0ff; }
.badge-balanced     { background:#1c2128; color:#a371f7; border:1px solid #a371f7; }

/* ── Section headers ── */
.section-header {
    font-size   : 1.05rem;
    font-weight : 700;
    color       : #58a6ff !important;
    padding-left: 4px;
    border-left : 3px solid #58a6ff;
    margin      : 20px 0 10px;
}

/* ── Tip cards ── */
.tip-card {
    background   : #161b22;
    border       : 1px solid #30363d;
    border-radius: 10px;
    padding      : 14px 18px;
    margin-bottom: 10px;
}
.tip-title { color: #58a6ff !important; font-weight:700; }

/* ── Progress bar override ── */
div[data-testid="stProgress"] > div { background:#30363d; }
div[data-testid="stProgress"] > div > div { background:#58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────
BADGE_CLASS = {
    "Beginner":     "badge-beginner",
    "Intermediate": "badge-intermediate",
    "Pro":          "badge-pro",
    "Aggressive":   "badge-aggressive",
    "Defensive":    "badge-defensive",
    "Balanced":     "badge-balanced",
}

def badge_html(text: str) -> str:
    cls = BADGE_CLASS.get(text, "badge-balanced")
    return f'<span class="badge {cls}">{text}</span>'

def models_exist() -> bool:
    paths = [
        os.path.join(MODELS_DIR, "xgb_classifier.pkl"),
        os.path.join(MODELS_DIR, "kmeans.pkl"),
    ]
    return all(os.path.exists(p) for p in paths)

SKILL_COLOR = {
    "Beginner":     "#ff7b72",
    "Intermediate": "#e3b341",
    "Pro":          "#3fb950",
}

PLOTLY_TEMPLATE = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font_color    = "#8b949e",
    xaxis         = dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis         = dict(gridcolor="#21262d", zerolinecolor="#21262d"),
)

# ─── Data caching ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def get_data():
    return load_and_preprocess()


# ─── Session-state defaults ──────────────────────────────────────────────────
if "leaderboard"    not in st.session_state:
    st.session_state.leaderboard = []
if "user_result"    not in st.session_state:
    st.session_state.user_result = None
if "user_inputs"    not in st.session_state:
    st.session_state.user_inputs = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 CS:GO Analyzer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "📥  Input Stats",
            "📊  Analysis",
            "📈  Visualizations",
            "💡  Recommendations",
            "🏆  Leaderboard",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;color:#484f58;text-align:center;">'
        "Final-Year ML Project<br>CS:GO Pro Games Dataset</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🎯 Gaming Performance Analyzer")
    st.markdown("##### Powered by XGBoost · KMeans · Plotly")
    st.markdown("---")

    player_df, per_match, dataset_avg = get_data()

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Players",   f"{len(player_df):,}")
    col2.metric("Total Matches",   f"{len(per_match):,}")
    col3.metric("Avg Rating",      f"{dataset_avg['game_rating']:.2f}")
    col4.metric("Avg K/D Ratio",   f"{dataset_avg['kd_ratio']:.2f}")

    st.markdown("---")

    # Skill distribution donut
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Skill Distribution</p>', unsafe_allow_html=True)
        skill_counts = player_df["skill_label"].value_counts().reset_index()
        skill_counts.columns = ["Skill", "Count"]
        fig = go.Figure(go.Pie(
            labels=skill_counts["Skill"],
            values=skill_counts["Count"],
            hole=0.55,
            marker_colors=[SKILL_COLOR.get(s, "#666") for s in skill_counts["Skill"]],
        ))
        fig.update_layout(**PLOTLY_TEMPLATE, height=320,
                          showlegend=True,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Rating Distribution</p>', unsafe_allow_html=True)
        fig2 = px.histogram(
            player_df, x="game_rating", nbins=50,
            color_discrete_sequence=["#58a6ff"],
        )
        fig2.update_layout(**PLOTLY_TEMPLATE, height=320,
                           xaxis_title="Game Rating",
                           yaxis_title="Players",
                           bargap=0.05)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Feature weight breakdown
    st.markdown('<p class="section-header">Performance Score Weights</p>', unsafe_allow_html=True)
    weights_df = pd.DataFrame({
        "Feature": list(PERFORMANCE_WEIGHTS.keys()),
        "Weight":  [w * 100 for w in PERFORMANCE_WEIGHTS.values()],
    })
    fig3 = px.bar(weights_df, x="Feature", y="Weight",
                   text="Weight",
                   color="Weight",
                   color_continuous_scale="Blues")
    fig3.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig3.update_layout(**PLOTLY_TEMPLATE, height=280,
                        yaxis_title="Weight (%)",
                        coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("""
**How it works:**
1. Enter your in-game stats on the **Input Stats** page.
2. The system computes your **Performance Score** (0–100) and **Skill Label**.
3. An **XGBoost** model validates the skill prediction.
4. A **KMeans** model detects your **Playstyle** (Aggressive / Balanced / Defensive).
5. Interactive charts compare you against 12 000+ pro-game observations.
6. Personalised **Recommendations** highlight your weak areas.
7. Submit to the **Leaderboard** to see where you rank.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: INPUT STATS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📥  Input Stats":
    st.title("📥  Enter Your Stats")
    st.markdown("Fill in your average per-game stats. All values are **per-map averages**.")
    st.markdown("---")

    player_df, _, dataset_avg = get_data()
    avg = dataset_avg

    with st.form("stat_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="section-header">Combat</p>', unsafe_allow_html=True)
            name        = st.text_input("Player Name / Tag", value="Player1")
            kills       = st.slider("Kills (avg per map)",    0.0, 50.0, float(round(avg["kills"],    1)), 0.5)
            deaths      = st.slider("Deaths (avg per map)",   0.0, 40.0, float(round(avg["deaths"],   1)), 0.5)
            assists      = st.slider("Assists (avg per map)",  0.0, 20.0, float(round(avg["assists"],  1)), 0.5)
            hs_pct      = st.slider("Headshot % ",             0.0, 100.0, float(round(avg["hs_pct"],  1)), 1.0)

        with col2:
            st.markdown('<p class="section-header">Advanced Metrics</p>', unsafe_allow_html=True)
            game_rating = st.slider("Game Rating (HLTV 2.0)", 0.0, 3.0,  float(round(avg["game_rating"], 2)), 0.01)
            kast        = st.slider("KAST %",                  0.0, 100.0, float(round(avg["kast"],       1)), 1.0)
            adr         = st.slider("ADR (Avg Damage / Round)", 0.0, 200.0, float(round(avg["adr"],      1)), 0.5)
            fkdiff      = st.slider("FK Diff (First Kill Diff)", -15.0, 15.0, float(round(avg["fkdiff"], 1)), 0.5)

        submitted = st.form_submit_button("🔍  Analyze My Performance", use_container_width=True)

    if submitted:
        if not models_exist():
            st.error("⚠️  Models not found. Please run `python model_training.py` first.")
        else:
            kddiff = kills - deaths

            # Performance score
            result = compute_user_score(
                kills, deaths, game_rating, kast, adr, fkdiff, player_df
            )
            result["name"]        = name
            result["kills"]       = kills
            result["deaths"]      = deaths
            result["assists"]      = assists
            result["kast"]        = kast
            result["adr"]         = adr
            result["fkdiff"]      = fkdiff
            result["game_rating"] = game_rating
            result["hs_pct"]      = hs_pct
            result["kddiff"]      = kddiff

            # ML predictions
            result["ml_skill"] = predict_skill(
                kills, deaths, assists, kast, kddiff,
                adr, fkdiff, game_rating, result["kd_ratio"], hs_pct
            )[0]
            result["playstyle"] = predict_playstyle(
                kills, deaths, result["kd_ratio"], adr, fkdiff, kast
            )

            st.session_state.user_result = result
            st.session_state.user_inputs = {
                "kills": kills, "deaths": deaths, "assists": assists,
                "kast": kast, "adr": adr, "fkdiff": fkdiff,
                "game_rating": game_rating, "hs_pct": hs_pct,
            }

            st.success("✅  Analysis complete! Navigate to the **Analysis** page.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analysis":
    st.title("📊  Performance Analysis")
    st.markdown("---")

    result = st.session_state.user_result
    if result is None:
        st.info("👈  Please enter your stats on the **Input Stats** page first.")
        st.stop()

    # ── Summary row ──────────────────────────────────────────────────────────
    score   = result["performance_score"]
    skill   = result["skill_label"]
    ml_sk   = result["ml_skill"]
    play    = result["playstyle"]
    kd      = result["kd_ratio"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Performance Score", f"{score:.1f} / 100")
    col2.metric("K/D Ratio",         f"{kd:.2f}")
    col3.metric("Game Rating",        f"{result['game_rating']:.2f}")
    col4.metric("KAST %",             f"{result['kast']:.1f}%")

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<p class="section-header">Skill Level (Score)</p>', unsafe_allow_html=True)
        st.markdown(badge_html(skill), unsafe_allow_html=True)
        st.markdown(f"*Score-based classification*")

    with col_b:
        st.markdown('<p class="section-header">Skill Level (ML)</p>', unsafe_allow_html=True)
        st.markdown(badge_html(ml_sk), unsafe_allow_html=True)
        st.markdown("*XGBoost prediction*")

    with col_c:
        st.markdown('<p class="section-header">Playstyle</p>', unsafe_allow_html=True)
        st.markdown(badge_html(play), unsafe_allow_html=True)
        st.markdown("*KMeans cluster*")

    st.markdown("---")

    # ── Score gauge ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Performance Gauge</p>', unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": 50, "increasing": {"color": "#3fb950"}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar":   {"color": SKILL_COLOR.get(skill, "#58a6ff")},
            "steps": [
                {"range": [0,  50], "color": "#1c2128"},
                {"range": [50, 75], "color": "#1c2128"},
                {"range": [75, 100],"color": "#1c2128"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": score,
            },
        },
        number = {"suffix": " pts", "font": {"color": "#e6edf3"}},
        title  = {"text": "Performance Score", "font": {"color": "#8b949e"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Scaled metric contributions ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Metric Contributions (0–100 scale)</p>',
                unsafe_allow_html=True)

    contrib_data = {
        "Metric": ["Game Rating", "K/D Ratio", "KAST", "ADR", "FK Diff"],
        "Scaled Score": [
            result["game_rating_scaled"],
            result["kd_ratio_scaled"],
            result["kast_scaled"],
            result["adr_scaled"],
            result["fkdiff_scaled"],
        ],
    }
    for m, s in zip(contrib_data["Metric"], contrib_data["Scaled Score"]):
        pct  = min(int(s), 100)
        color = "#3fb950" if pct >= 75 else "#e3b341" if pct >= 50 else "#ff7b72"
        st.markdown(f"**{m}**")
        st.progress(pct / 100)
        st.markdown(f'<span style="color:{color};font-size:0.85rem">{pct:.1f} / 100</span>',
                    unsafe_allow_html=True)

    # ── Leaderboard submit button ─────────────────────────────────────────────
    st.markdown("---")
    if st.button("🏆  Submit to Leaderboard", use_container_width=True):
        entry = {
            "Name":             result["name"],
            "Score":            round(result["performance_score"], 1),
            "Skill":            result["skill_label"],
            "Playstyle":        result["playstyle"],
            "K/D":              round(result["kd_ratio"], 2),
            "Rating":           round(result["game_rating"], 2),
            "Timestamp":        datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.leaderboard.append(entry)
        st.success("✅  Added to leaderboard!")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Visualizations":
    st.title("📈  Visualizations")
    st.markdown("---")

    result    = st.session_state.user_result
    player_df, _, dataset_avg = get_data()

    if result is None:
        st.info("👈  Please enter your stats on the **Input Stats** page first.")
        st.stop()

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Radar Chart – Scaled Metrics (0–100)</p>',
                unsafe_allow_html=True)

    cats   = ["Game Rating", "K/D Ratio", "KAST", "ADR", "FK Diff", "Game Rating"]
    user_v = [
        result["game_rating_scaled"],
        result["kd_ratio_scaled"],
        result["kast_scaled"],
        result["adr_scaled"],
        result["fkdiff_scaled"],
        result["game_rating_scaled"],   # close the polygon
    ]

    # Compute dataset average in the same scaled space
    # (midpoint of distribution ≈ 50 for all after MinMax)
    avg_v  = [50, 50, 50, 50, 50, 50]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=avg_v, theta=cats,
        fill="toself", name="Dataset Average",
        line_color="#8b949e", fillcolor="rgba(139,148,158,0.1)"
    ))
    fig_r.add_trace(go.Scatterpolar(
        r=user_v, theta=cats,
        fill="toself", name=result["name"],
        line_color="#58a6ff", fillcolor="rgba(88,166,255,0.2)"
    ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="#21262d", tickcolor="#8b949e"),
            angularaxis=dict(gridcolor="#21262d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        legend=dict(font_color="#8b949e"),
        height=420,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Bar chart – user vs avg ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Your Stats vs Dataset Average</p>',
                unsafe_allow_html=True)

    compare_metrics = ["kills", "deaths", "assists", "kast", "adr", "game_rating"]
    compare_labels  = ["Kills", "Deaths", "Assists", "KAST %", "ADR", "Rating"]
    user_vals       = [result.get(m, 0) for m in compare_metrics]
    avg_vals        = [dataset_avg.get(m, 0) for m in compare_metrics]

    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        x=compare_labels, y=user_vals,
        name=result["name"], marker_color="#58a6ff",
    ))
    fig_b.add_trace(go.Bar(
        x=compare_labels, y=avg_vals,
        name="Dataset Avg", marker_color="#3fb950",
    ))
    fig_b.update_layout(**PLOTLY_TEMPLATE, barmode="group", height=360,
                         yaxis_title="Value",
                         legend=dict(font_color="#8b949e"))
    st.plotly_chart(fig_b, use_container_width=True)

    # ── K/D distribution with user marker ────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">K/D Ratio Distribution</p>',
                unsafe_allow_html=True)

    fig_kd = px.histogram(
        player_df, x="kd_ratio", nbins=80,
        color_discrete_sequence=["#a371f7"],
        opacity=0.75,
    )
    fig_kd.add_vline(
        x=result["kd_ratio"],
        line_dash="dash", line_color="#ff7b72", line_width=2,
        annotation_text=f"You: {result['kd_ratio']:.2f}",
        annotation_font_color="#ff7b72",
    )
    fig_kd.add_vline(
        x=dataset_avg["kd_ratio"],
        line_dash="dot", line_color="#3fb950", line_width=2,
        annotation_text=f"Avg: {dataset_avg['kd_ratio']:.2f}",
        annotation_font_color="#3fb950",
        annotation_position="top left",
    )
    fig_kd.update_layout(**PLOTLY_TEMPLATE, height=300,
                          xaxis_title="K/D Ratio", yaxis_title="Players")
    st.plotly_chart(fig_kd, use_container_width=True)

    # ── Performance score vs rating scatter ───────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Performance Score vs Game Rating</p>',
                unsafe_allow_html=True)

    sample = player_df.sample(min(2000, len(player_df)), random_state=42)
    fig_sc = px.scatter(
        sample, x="game_rating", y="performance_score",
        color="skill_label",
        color_discrete_map=SKILL_COLOR,
        opacity=0.55, height=380,
        labels={"game_rating": "Game Rating", "performance_score": "Performance Score"},
    )
    fig_sc.add_scatter(
        x=[result["game_rating"]],
        y=[result["performance_score"]],
        mode="markers+text",
        marker=dict(size=14, color="white", symbol="star",
                    line=dict(color="#58a6ff", width=2)),
        text=[f"  {result['name']}"],
        textfont=dict(color="#e6edf3"),
        name=result["name"],
    )
    fig_sc.update_layout(**PLOTLY_TEMPLATE,
                          legend=dict(font_color="#8b949e"))
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡  Recommendations":
    st.title("💡  Personalised Recommendations")
    st.markdown("---")

    result = st.session_state.user_result
    if result is None:
        st.info("👈  Please enter your stats on the **Input Stats** page first.")
        st.stop()

    _, _, dataset_avg = get_data()

    skill    = result["skill_label"]
    play     = result["playstyle"]
    score    = result["performance_score"]
    kd       = result["kd_ratio"]
    kast     = result["kast"]
    adr      = result["adr"]
    fkdiff   = result["fkdiff"]
    game_r   = result["game_rating"]
    hs_pct   = result["hs_pct"]

    st.markdown(
        f'**{result["name"]}** · {badge_html(skill)} · {badge_html(play)} · '
        f'Score: **{score:.1f}**',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tips = []

    # K/D advice
    if kd < dataset_avg["kd_ratio"] * 0.80:
        tips.append(("🔴 K/D Ratio is Below Average",
            f"Your K/D is **{kd:.2f}** vs dataset avg **{dataset_avg['kd_ratio']:.2f}**. "
            "Focus on positioning — peek angles only when you have information, "
            "and avoid aggressive early pushes without utility."))
    elif kd >= dataset_avg["kd_ratio"] * 1.20:
        tips.append(("🟢 Excellent K/D Ratio",
            f"Your K/D of **{kd:.2f}** is well above average. "
            "Try converting that efficiency into more clutch rounds by "
            "saving when appropriate and calling strats for your team."))

    # KAST advice
    if kast < dataset_avg["kast"] * 0.85:
        tips.append(("🔴 Low KAST % — Inconsistency Detected",
            f"KAST **{kast:.1f}%** vs avg **{dataset_avg['kast']:.1f}%**. "
            "KAST penalises rounds where you contribute nothing. "
            "Prioritise assist/trade kills and don't isolate yourself on bombsites."))

    # ADR advice
    if adr < dataset_avg["adr"] * 0.85:
        tips.append(("🟡 Low Average Damage Per Round",
            f"ADR **{adr:.1f}** vs avg **{dataset_avg['adr']:.1f}**. "
            "Use grenades more proactively, chip enemies through smokes, "
            "and fight for low-hp trades."))
    elif adr > dataset_avg["adr"] * 1.20:
        tips.append(("🟢 High ADR",
            f"Great damage output of **{adr:.1f}**. "
            "Make sure those damage rounds convert to kills — "
            "try to get the finishing shot more often."))

    # FK Diff advice
    if fkdiff < -2:
        tips.append(("🔴 Negative First Kill Differential",
            f"FK Diff **{fkdiff:.1f}** — you're being first-killed more than you "
            "open duels. If you're entry-fragging, use more utility. "
            "If you're support, avoid passive positions that put you alone."))
    elif fkdiff >= 3:
        tips.append(("🟢 Strong First Kill Differential",
            f"FK Diff of **{fkdiff:.1f}** — you're consistently opening duels. "
            "Channel this to set up your team by calling post-plant positions."))

    # Headshot %
    if hs_pct < 30:
        tips.append(("🟡 Low Headshot Percentage",
            f"HS% **{hs_pct:.1f}%** — practice aim on headshot-only DM servers "
            "or Aim Lab 'Micro-Shot' routines to raise crosshair placement."))
    elif hs_pct > 65:
        tips.append(("🟢 High Headshot Percentage",
            f"HS% **{hs_pct:.1f}%** — excellent. Consider learning quick-scope "
            "AWP flicks to diversify your impact."))

    # Game rating
    if game_r < 1.0:
        tips.append(("🔴 Below-Average Game Rating",
            f"Rating **{game_r:.2f}** is below 1.0. The single biggest driver of "
            "rating is surviving rounds (KAST) and winning duels. Work on "
            "game sense — don't take duels unless you're winning them."))

    # Playstyle-specific advice
    if play == "Aggressive":
        tips.append(("⚡ Aggressive Playstyle Detected",
            "You have a high-kill, high-risk style. Make sure to: "
            "(1) coordinate pushes with teammates, (2) flash before peeking, "
            "(3) reset to eco if you lose multiple duels in a row."))
    elif play == "Defensive":
        tips.append(("🛡 Defensive Playstyle Detected",
            "You play passively and value survival. Expand your impact by: "
            "(1) info calls for your team, (2) setting up crossfires, "
            "(3) rotating earlier on CT side."))
    else:
        tips.append(("⚖️ Balanced Playstyle Detected",
            "You adapt well. Refine by: (1) choosing one specialist role, "
            "(2) building a utility arsenal around that role, "
            "(3) reviewing demo footage for positioning errors."))

    # Skill-level path forward
    if skill == "Beginner":
        tips.append(("📚 Next Step: Reach Intermediate",
            "Reach 50+ performance score by: improving crosshair placement, "
            "learning basic grenade lineups for 2 maps, and muting toxic "
            "teammates to stay focused."))
    elif skill == "Intermediate":
        tips.append(("📚 Next Step: Reach Pro",
            "Push past 75 by: mastering mid-round calling, developing a "
            "consistent warmup routine, and reviewing your own demos weekly."))
    else:
        tips.append(("🏆 You're Already Pro Level!",
            f"Score **{score:.1f}/100** — consider creating content, "
            "coaching teammates, or submitting to regional leagues."))

    # Render tips
    for title, body in tips:
        color = "#ff7b72" if "🔴" in title else "#3fb950" if "🟢" in title else "#e3b341"
        st.markdown(
            f'<div class="tip-card">'
            f'<p class="tip-title" style="color:{color}!important">{title}</p>'
            f'<p>{body}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not tips:
        st.success("🎉  All your stats are within a healthy range — keep it up!")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆  Leaderboard":
    st.title("🏆  Leaderboard")
    st.markdown("---")

    lb = st.session_state.leaderboard

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🗑  Clear Leaderboard"):
            st.session_state.leaderboard = []
            st.rerun()

    if not lb:
        st.info("No entries yet. Analyze your stats and submit from the **Analysis** page.")
        st.stop()

    df_lb = pd.DataFrame(lb).sort_values("Score", ascending=False).reset_index(drop=True)
    df_lb.index += 1   # start rank at 1

    # Colour-code the Skill column
    def colour_skill(val):
        c = {"Pro": "#3fb950", "Intermediate": "#e3b341", "Beginner": "#ff7b72"}.get(val, "")
        return f"color: {c}" if c else ""

    styled = (
        df_lb.style
             .applymap(colour_skill, subset=["Skill"])
             .format({"Score": "{:.1f}", "K/D": "{:.2f}", "Rating": "{:.2f}"})
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # Bar chart of top scores
    st.markdown("---")
    st.markdown('<p class="section-header">Top Scores</p>', unsafe_allow_html=True)
    fig = px.bar(
        df_lb.head(10), x="Name", y="Score",
        color="Skill",
        color_discrete_map={
            "Pro": "#3fb950", "Intermediate": "#e3b341", "Beginner": "#ff7b72"
        },
        text="Score",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(**PLOTLY_TEMPLATE, height=360,
                       legend=dict(font_color="#8b949e"),
                       xaxis_title="", yaxis_title="Performance Score")
    st.plotly_chart(fig, use_container_width=True)
