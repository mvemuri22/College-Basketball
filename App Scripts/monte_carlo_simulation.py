import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
import statsmodels.api as sm
from scipy.stats import beta
import os
from pathlib import Path

# -----------------------------
# Data prep and helper functions
# -----------------------------
def get_adj_metric_pos(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    X = sm.add_constant(df["sos"])
    model = sm.OLS(df[metric], X).fit()
    df["Adj " + metric] = df[metric] + (model.params["sos"] * df["sos"])
    return df


def harmonic_mean(x: float, y: float) -> float:
    x = float(x)
    y = float(y)
    if (x + y) == 0:
        return 0.0
    return (x * y * 2.0) / (x + y)


def get_beta_value(team_stat: float, opp_stat: float, ess: float = 20.0, eps: float = 1e-4):
    th = lambda v: float(np.clip(v, eps, 1.0 - eps))
    m_home = th(team_stat)
    m_away = th(opp_stat)

    a_home = max(m_home * ess, eps)
    b_home = max((1.0 - m_home) * ess, eps)
    a_away = max(m_away * ess, eps)
    b_away = max((1.0 - m_away) * ess, eps)

    h = beta.rvs(a_home, b_home, size=1)[0]
    a = beta.rvs(a_away, b_away, size=1)[0]
    return h, a


def prepare_team_stats_for_mc(df: pd.DataFrame, team_name: str) -> Dict[str, float]:
    team_row = df[df["Team"] == team_name]
    if team_row.empty:
        raise ValueError(f"Team not found: {team_name}")
    team_data = team_row.iloc[0]

    return {
        "Team": team_data["Team"],
        "eFG%": team_data["Adj eFG%"] / 100,
        "eFG% Def": team_data["Adj eFG% Def"] / 100,
        "TOR": team_data["Adj TO%"] / 100,
        "TORD": team_data["Adj TO% Def."] / 100,
        "ORB": team_data["Adj OR%"] / 100,
        "DR%": team_data["Adj DR%"] / 100,
        "3PR": team_data["3P rate"] / 100,
        "3P%": team_data["Adj 3P%"] / 100,
        "3pD%": team_data["Adj 3pD%"] / 100,
        "2P%": team_data["Adj 2p%"] / 100,
        "2pD%": team_data["Adj 2p%D"] / 100,
        "FTR": team_data["FTR"] / 100,
        "FT%": (team_data["ft%"] if "ft%" in df.columns else team_data.get("ft%D", 70.0)) / 100,
        "adjt": float(team_data["adjt"]),
    }


# -----------------------------
# Simulation core
# -----------------------------
@dataclass
class GameResult:
    home_score: float
    away_score: float
    home_fta: float
    away_fta: float
    total_possessions: float
    spread: float
    total: float


class EnhancedBasketballMC:
    def __init__(self, base_variance: float = 4.5):
        self.base_variance = base_variance

    def simulate_possession_outcome(self, team_stats: Dict, opp_stats: Dict, situation: str = "normal") -> Dict:
        to_rate = harmonic_mean(team_stats.get("TOR", 0.0), opp_stats.get("TORD", 0.0))
        ftr = team_stats.get("FTR", 0.0)

        if situation == "close_late":
            ftr *= 1.3
            to_rate *= 1.1
        elif situation == "blowout":
            ftr *= 0.7
            to_rate *= 1.2

        if np.random.random() < to_rate:
            return {"points": 0, "fta": 0, "type": "turnover"}

        if np.random.random() < ftr * 0.44:
            ft_pct = team_stats.get("FT%", 0.0)
            r = np.random.random()
            if r < 0.93:
                fta = 2
            elif r < 0.995:
                fta = 3
            else:
                fta = 1
            made = sum(np.random.random() < ft_pct for _ in range(int(fta)))
            return {"points": made, "fta": fta, "type": "free_throws"}

        three_pt_rate = team_stats.get("3PR", 0.0)
        if np.random.random() < three_pt_rate:
            three_make_prob = harmonic_mean(team_stats.get("3P%", 0.0), 1.0 - opp_stats.get("3pD%", 0.0))
            if np.random.random() < three_make_prob:
                return {"points": 3, "fta": 0, "type": "three_made"}
            else:
                if np.random.random() < harmonic_mean(team_stats.get("ORB", 0.0), 1.0 - opp_stats.get("DR%", 0.0)):
                    if np.random.random() < 0.25:
                        return {"points": 2, "fta": 0, "type": "putback_made"}
                    return {"points": 0, "fta": 0, "type": "putback_miss"}
                return {"points": 0, "fta": 0, "type": "three_miss"}
        else:
            two_make_prob = harmonic_mean(team_stats.get("2P%", 0.0), 1.0 - opp_stats.get("2pD%", 0.0))
            if np.random.random() < two_make_prob:
                return {"points": 2, "fta": 0, "type": "two_made"}
            else:
                if np.random.random() < harmonic_mean(team_stats.get("ORB", 0.0), 1.0 - opp_stats.get("DR%", 0.0)):
                    if np.random.random() < 0.2:
                        return {"points": 2, "fta": 0, "type": "putback_made"}
                    return {"points": 0, "fta": 0, "type": "putback_miss"}
                return {"points": 0, "fta": 0, "type": "two_miss"}

    def simulate_game_possession_by_possession(self, home_stats: Dict, away_stats: Dict, expected_possessions: float) -> GameResult:
        home_score = 0
        away_score = 0
        home_fta = 0
        away_fta = 0

        actual_possessions = int(np.random.normal(expected_possessions, 3))
        actual_possessions = max(60, min(80, actual_possessions))

        for poss in range(actual_possessions):
            score_diff = abs(home_score - away_score)
            poss_remaining = actual_possessions - poss

            if poss_remaining < 10 and score_diff < 6:
                situation = "close_late"
            elif poss_remaining < 15 and score_diff > 15:
                situation = "blowout"
            else:
                situation = "normal"

            home_result = self.simulate_possession_outcome(home_stats, away_stats, situation)
            home_score += home_result["points"]
            home_fta += home_result["fta"]

            if poss < actual_possessions - 1:
                away_result = self.simulate_possession_outcome(away_stats, home_stats, situation)
                away_score += away_result["points"]
                away_fta += away_result["fta"]

        return GameResult(
            home_score=home_score,
            away_score=away_score,
            home_fta=home_fta,
            away_fta=away_fta,
            total_possessions=actual_possessions,
            spread=away_score - home_score,
            total=home_score + away_score,
        )

    def _calculate_ppp(self, offense_stats: Dict, defense_stats: Dict) -> float:
        efg_off = offense_stats.get("eFG%", 0.0)
        efg_def_allowed = 1.0 - defense_stats.get("eFG% Def", 0.0)
        efg = harmonic_mean(efg_off, efg_def_allowed)
        return efg * 2.0 * 0.95

    def _simulate_game_aggregate(self, home_stats: Dict, away_stats: Dict, expected_possessions: float) -> GameResult:
        actual_poss = np.random.normal(expected_possessions, 2.5)
        home_ppp = self._calculate_ppp(home_stats, away_stats)
        away_ppp = self._calculate_ppp(away_stats, home_stats)
        home_score = np.random.normal(home_ppp * actual_poss, self.base_variance)
        away_score = np.random.normal(away_ppp * actual_poss, self.base_variance)
        return GameResult(
            home_score=home_score,
            away_score=away_score,
            home_fta=0,
            away_fta=0,
            total_possessions=actual_poss,
            spread=away_score - home_score,
            total=home_score + away_score,
        )

    def run_monte_carlo(
        self,
        home_stats: Dict,
        away_stats: Dict,
        n_sims: int = 10000,
        possession_level: bool = False,
        use_beta_distr: bool = True,
    ) -> Dict:
        results: List[GameResult] = []
        expected_poss = (home_stats["adjt"] + away_stats["adjt"]) / 2

        if not use_beta_distr:
            home_draws = home_stats.copy()
            away_draws = away_stats.copy()
        else:
            home_draws = {}
            away_draws = {}
            for stat in home_stats:
                if stat in ("Team", "adjt"):
                    home_draws[stat] = home_stats[stat]
                    away_draws[stat] = away_stats.get(stat, home_stats[stat])
                else:
                    t_val = home_stats.get(stat, 0.0)
                    o_val = away_stats.get(stat, t_val)
                    home_draws[stat], away_draws[stat] = get_beta_value(t_val, o_val, ess=20.0)

        for _ in range(n_sims):
            if possession_level:
                result = self.simulate_game_possession_by_possession(home_draws, away_draws, expected_poss)
            else:
                result = self._simulate_game_aggregate(home_draws, away_draws, expected_poss)
            results.append(result)

        spreads = np.array([r.spread for r in results])
        totals = np.array([r.total for r in results])
        home_scores = np.array([r.home_score for r in results])
        away_scores = np.array([r.away_score for r in results])

        return {
            "spreads": spreads,
            "totals": totals,
            "home_scores": home_scores,
            "away_scores": away_scores,
            "results": results,
            "summary": {
                "mean_spread": float(np.mean(spreads)),
                "median_spread": float(np.median(spreads)),
                "std_spread": float(np.std(spreads)),
                "mean_total": float(np.mean(totals)),
                "median_total": float(np.median(totals)),
                "std_total": float(np.std(totals)),
                "home_win_prob": float(np.mean(home_scores > away_scores)),
                "percentiles": {
                    "spread": {
                        "5th": float(np.percentile(spreads, 5)),
                        "25th": float(np.percentile(spreads, 25)),
                        "75th": float(np.percentile(spreads, 75)),
                        "95th": float(np.percentile(spreads, 95)),
                    },
                    "total": {
                        "5th": float(np.percentile(totals, 5)),
                        "25th": float(np.percentile(totals, 25)),
                        "75th": float(np.percentile(totals, 75)),
                        "95th": float(np.percentile(totals, 95)),
                    },
                },
            },
        }


# -----------------------------
# CLI / Streamlit integration
# -----------------------------
def load_and_prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sos" not in df.columns:
        df["sos"] = 0.0
    df = get_adj_metric_pos(df, "OR%")
    df = get_adj_metric_pos(df, "eFG%")
    df = get_adj_metric_pos(df, "2p%")
    df = get_adj_metric_pos(df, "3P%")
    df = get_adj_metric_pos(df, "TO% Def.")
    df = get_adj_metric_pos(df, "DR%")
    df = get_adj_metric_pos(df, "eFG% Def")
    df = get_adj_metric_pos(df, "2p%D")
    df = get_adj_metric_pos(df, "3pD%")
    df = get_adj_metric_pos(df, "TO%")
    return df

def _find_home_away_cols(df: pd.DataFrame):
    """Return (home_col, away_col) for common column names or None."""
    candidates = [
        ('home_team','away_team'), ('Home','Away'), ('home','away'),
        ('team_home','team_away'), ('Home Team','Away Team')
    ]
    for h,a in candidates:
        if h in df.columns and a in df.columns:
            return h,a
    # fallback: try first two columns
    if df.shape[1] >= 2:
        return df.columns[0], df.columns[1]
    return None, None

def main():
    base = Path(__file__).resolve().parent

    torvik_path = base / "Data/latest_torvik.csv"
    matchups_path = base / "Data/matchups_today.csv"
    out_path = base / "matchups_sim_results.csv"

    if not torvik_path.exists():
        print(f"Missing file: {torvik_path}")
        return
    if not matchups_path.exists():
        print(f"Missing file: {matchups_path}")
        return

    torvik = load_and_prepare_dataframe(str(torvik_path))

    matchups = pd.read_csv(matchups_path)

    home_col, away_col = _find_home_away_cols(matchups)
    if home_col is None:
        print("Couldn't find home/away columns in matchups_today.csv")
        return

    simulator = EnhancedBasketballMC(base_variance=4.5)

    rows = []
    for _, r in matchups.iterrows():
        home_name = str(r[home_col]).strip()
        away_name = str(r[away_col]).strip()

        try:
            home_stats = prepare_team_stats_for_mc(torvik, home_name)
            away_stats = prepare_team_stats_for_mc(torvik, away_name)

            # run fewer sims for quick results; increase n_sims for production
            sim = simulator.run_monte_carlo(
                home_stats,
                away_stats,
                n_sims=10000,
                possession_level=True,
                use_beta_distr=False
            )

            s = sim['summary']
            rows.append({
                'home_team': home_name,
                'away_team': away_name,
                'mean_spread_away_minus_home': s['mean_spread'],
                'median_spread': s['median_spread'],
                'std_spread': s['std_spread'],
                'mean_total': s['mean_total'],
                'home_win_prob': s['home_win_prob']
            })
            print(f"{home_name} vs {away_name} -> mean spread (away-home): {s['mean_spread']:.2f}, home win prob: {s['home_win_prob']:.1%}")

        except Exception as exc:
            print(f"Error simulating {home_name} vs {away_name}: {exc}")
            rows.append({
                'home_team': home_name,
                'away_team': away_name,
                'error': str(exc)
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()