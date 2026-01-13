import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MONTE_SCRIPT = BASE / "monte_carlo_simulation.py"
BART_SCRIPT = BASE / "game_simulation_bart.py"
TORVIK = BASE / "Data/latest_torvik.csv"
MATCHUPS = BASE / "Data/matchups_today.csv"
MONTE_OUT = BASE / "matchups_sim_results.csv"
BART_OUT = BASE / "matchups_bart_results.csv"
ODDS_PATH = BASE / "Data/odds_today.csv"

st.set_page_config(page_title="Basketball MC — Matchups", layout="wide")

st.sidebar.title("Controls")
n_sims = st.sidebar.slider("Monte Carlo sims per matchup (quick)", 500, 5000, 2000, step=250)
run_monte = st.sidebar.button("Run Monte Carlo script")
run_bart = st.sidebar.button("Run BART script")
run_both = st.sidebar.button("Run Both")
refresh = st.sidebar.button("Refresh results")

def run_script(path: Path, extra_args=None):
    cmd = [sys.executable, str(path)]
    if extra_args:
        cmd += extra_args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE, check=True)
        return True, proc.stdout + proc.stderr
    except subprocess.CalledProcessError as e:
        return False, (e.stdout or "") + (e.stderr or "")

@st.cache_data
def load_csv_if_exists(p: Path):
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def load_odds_for_books(p: Path, books=("fanduel", "draftkings")):
    if not p.exists():
        return None
    odds = pd.read_csv(p)
    # normalize team column
    if 'Team Modified' in odds.columns:
        odds = odds.rename(columns={'Team Modified': 'team'})
    elif 'Team' in odds.columns and 'team' not in odds.columns:
        odds = odds.rename(columns={'Team': 'team'})

    # ensure Market filter if present
    if 'Market' in odds.columns:
        odds = odds[odds['Market'].str.lower() == 'spreads']

    odds['Point'] = pd.to_numeric(odds.get('Point', pd.Series()), errors='coerce')

    def canon(s: str) -> str:
        return str(s).strip().lower().replace('.', '').replace("'", "").replace("  ", " ")

    odds['team_key'] = odds['team'].apply(canon)
    odds['book_key'] = odds['Bookmaker'].astype(str).str.strip().str.lower()

    # filter to selected bookmakers
    odds = odds[odds['book_key'].isin([b.lower() for b in books])]

    # per-book median per team
    pivot = (
        odds.groupby(['team_key', 'book_key'])['Point']
        .median()
        .unstack(fill_value=np.nan)
        .reset_index()
    )
    # rename columns to explicit book names
    rename_map = {col: f"{col}_point" for col in pivot.columns if col not in ['team_key']}
    pivot = pivot.rename(columns=rename_map)
    return pivot

st.title("Daily Matchup Simulation Dashboard")

# display file statuses in single row and controls in left column
top_col1, top_col2 = st.columns([2, 3])

with top_col1:
    st.markdown("### Files")
    file_status = {
        "torvik": TORVIK.exists(),
        "matchups": MATCHUPS.exists(),
        "monte_out": MONTE_OUT.exists(),
        "bart_out": BART_OUT.exists(),
        "odds": ODDS_PATH.exists()
    }
    # display single-row status table
    status_df = pd.DataFrame([file_status])
    status_df = status_df.rename(columns={k: k for k in status_df.columns})
    st.table(status_df)

with top_col2:
    st.markdown("### Controls")
    st.write(f"Monte Carlo sims (quick): {n_sims}")
    if run_monte or run_both:
        st.info("Running Monte Carlo script...")
        with st.spinner("Running monte_carlo_simulation.py ..."):
            ok, out = run_script(MONTE_SCRIPT, extra_args=[str(n_sims)])
            if ok:
                st.success("Monte Carlo finished")
            else:
                st.error("Monte Carlo failed")
            st.code(out[:10000])

    if run_bart or run_both:
        st.info("Running BART script...")
        with st.spinner("Running game_simulation_bart.py ..."):
            ok, out = run_script(BART_SCRIPT)
            if ok:
                st.success("BART script finished")
            else:
                st.error("BART script failed")
            st.code(out[:10000])

# Results table below
st.markdown("---")
st.header("Combined Results")

df_monte = load_csv_if_exists(MONTE_OUT)
df_bart = load_csv_if_exists(BART_OUT)
odds_books = load_odds_for_books(ODDS_PATH, books=("fanduel", "draftkings"))

if df_monte is None and df_bart is None:
    st.warning("No results CSVs found. Run scripts from controls or place CSVs in the Data folder.")
    st.stop()

# assemble display dataframe (prefer Monte as primary)
if df_monte is not None:
    df_display = df_monte.copy()
    if df_bart is not None:
        df_display = df_display.merge(
            df_bart[['home_team','away_team','mean_spread_away_minus_home']],
            on=['home_team','away_team'], how='left', suffixes=('','_bart')
        )
        # rename bart column to explicit
        df_display = df_display.rename(columns={'mean_spread_away_minus_home_bart':'spread_bart'})
        df_display = df_display.rename(columns={'mean_spread_away_minus_home':'spread_mc'})
    else:
        df_display['spread_bart'] = pd.NA
else:
    df_display = df_bart.copy()
    df_display.rename(columns={'mean_spread_away_minus_home':'spread_bart'}, inplace=True)
    df_display['spread_mc'] = pd.NA

# add canonical key and merge per-book odds
def canon_key(s: str) -> str:
    return str(s).strip().lower().replace('.', '').replace("'", "").replace("  ", " ")

df_display['home_key'] = df_display['home_team'].apply(canon_key)

if odds_books is not None:
    df_display = df_display.merge(odds_books, left_on='home_key', right_on='team_key', how='left')

# compute model spread (avg of monte & bart if both present)
def compute_model_spread(row):
    vals = []
    try:
        v = float(row.get('spread_mc'))
        if not np.isnan(v):
            vals.append(v)
    except Exception:
        pass
    try:
        v2 = float(row.get('spread_bart'))
        if not np.isnan(v2):
            vals.append(v2)
    except Exception:
        pass
    return np.nan if not vals else float(np.mean(vals))

df_display['model_spread'] = df_display.apply(compute_model_spread, axis=1)

# add per-book columns for easier UI (Fanduel/DraftKings)
df_display['fanduel_point'] = pd.to_numeric(df_display.get('fanduel_point', pd.Series()), errors='coerce')
df_display['draftkings_point'] = pd.to_numeric(df_display.get('draftkings_point', pd.Series()), errors='coerce')

# Determine agreement between models on side (HOME vs AWAY)
def agreement_indicator(row):
    m = row.get('spread_mc')
    b = row.get('spread_bart')
    spread = row.get('fanduel_point')
    try:
        m = float(m)
    except Exception:
        m = np.nan
    try:
        b = float(b)
    except Exception:
        b = np.nan
    if np.isnan(m) or np.isnan(b):
        return {"agree": False, "side": None, "emoji": "—"}
    
    sm = (m - spread) > 0
    sb = (b - spread) > 0

    if sm == True and sb == True:
        return {"agree": True, "side": "AWAY", "emoji": "✅"}
    if sm == False and sb == False:
        return {"agree": True, "side": "HOME", "emoji": "✅"}
    else:
        return {"agree": False, "side": None, "emoji": "—"}

agree_results = df_display.apply(agreement_indicator, axis=1)
df_display['models_agree'] = [r['emoji'] for r in agree_results]
df_display['agree_side'] = [r['side'] for r in agree_results]

# compute model - market per book if available
if 'fanduel_point' in df_display.columns:
    df_display['model_minus_fanduel'] = df_display['model_spread'] - df_display['fanduel_point']
else:
    df_display['model_minus_fanduel'] = np.nan
if 'draftkings_point' in df_display.columns:
    df_display['model_minus_draftkings'] = df_display['model_spread'] - df_display['draftkings_point']
else:
    df_display['model_minus_draftkings'] = np.nan

# final display columns and formatting
display_cols = [
    'home_team', 'away_team',
    'spread_mc', 'spread_bart',
    'model_spread', 'models_agree', 'agree_side',
    'fanduel_point', 'model_minus_fanduel',
    'draftkings_point', 'model_minus_draftkings',
    'home_win_prob'
]
# keep only existing columns
display_cols = [c for c in display_cols if c in df_display.columns]

display_df = df_display[display_cols].copy()

# nicer representation for NaN
display_df = display_df.fillna("N/A")

st.dataframe(display_df, use_container_width=True, height=600)

st.markdown("---")
st.header("Inspect a matchup")

if df_display is not None and not df_display.empty:
    choices = df_display.apply(lambda r: f"{r.home_team}  @  {r.away_team}", axis=1).tolist()
    sel = st.selectbox("Choose matchup", ["-- select --"] + choices)
    if sel and sel != "-- select --":
        home, away = [s.strip() for s in sel.split("@")]
        row = df_display[(df_display['home_team']==home) & (df_display['away_team']==away)].iloc[0]
        st.subheader(f"{home} vs {away}")
        st.write("Monte Carlo mean spread (away-home):", row.get('mean_spread_away_minus_home'))
        st.write("BART mean spread (away-home):", row.get('mean_spread_away_minus_home_bart'))
        st.write("Model spread (avg):", row.get('model_spread'))
        st.write("Fanduel spread (median):", row.get('fanduel_point'))
        st.write("DraftKings spread (median):", row.get('draftkings_point'))
        st.write("Models agree:", row.get('models_agree'), "Side:", row.get('agree_side'))
        st.write("Home win prob (monte):", row.get('home_win_prob'))

        # allow running an on-demand single-match sim
        if st.button("Run single-match Monte Carlo (quick, 1000 sims)"):
            with st.spinner("Running single-match sim..."):
                ok, out = run_script(MONTE_SCRIPT, extra_args=["--single", home, away, "1000"])
                if ok:
                    st.success("Single-match sim finished")
                else:
                    st.error("Single-match sim failed")
                st.code(out[:8000])

else:
    st.info("No matchups to inspect")

st.markdown("Run this app from your terminal:")
st.code(r'cd "C:\Users\manas\OneDrive\Documents\College Basketball\App Scripts" && streamlit run streamlit.py')