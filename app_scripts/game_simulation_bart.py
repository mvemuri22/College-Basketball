import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import beta
import argparse


def get_adj_metric_pos (df, metric):
    X = sm.add_constant(df['sos'])
    model = sm.OLS(df[metric], X).fit()
    df['Adj ' + metric] = df[metric] + (model.params['sos'] * df['sos'])
    return df
def get_adj_metric_neg (df, metric):
    X = sm.add_constant(df['sos'])
    model = sm.OLS(df[metric], X).fit()
    df['Adj ' + metric] = df[metric] - (model.params['sos'] * df['sos'])
    return df
def harmonic_mean(x,y):
    return ((x*y*2)/(x+y))

def get_beta_value (df,metric,home_team,away_team):
    mean_home = df.loc[home_team][metric]
    mean_away = df.loc[away_team][metric]
    alpha_home = 20 * mean_home
    beta_home = 20 * (1 - mean_home)
    alpha_away = 20 * mean_away
    beta_away = 20 * (1 - mean_away)
    home_val = beta.rvs(alpha_home, beta_home, size=1)
    away_val = beta.rvs(alpha_away, beta_away, size=1)
    return home_val[0], away_val[0]

def get_regular_team_stats(home_team,away_team, df):
    df_team = df[(df['Team'] == home_team) | (df['Team'] == away_team)]
    df_team = df_team[['Team', 'barthag', 'Adj OR%', 'Adj DR%', 'Adj eFG%', 'Adj 2p%', 'Adj 3P%', 'Adj TO% Def.', 'Adj eFG% Def', 'Adj TO%', 'Adj 2p%D', 'Adj 3pD%','adjt','3P rate','FTR','ft%D','3P rate D']]
    df_team = df_team.set_index('Team')
    for col in [ 'Adj OR%', 'Adj DR%', 'Adj eFG%', 'Adj 2p%', 'Adj 3P%', 'Adj TO% Def.', 'Adj eFG% Def', 'Adj TO%', 'Adj 2p%D', 'Adj 3pD%','3P rate','FTR','ft%D','3P rate D']:
        df_team[col] = df_team[col] / 100
    return df_team

def get_simulated_team_stats(home_team, away_team, df):
    df_team = df[(df['Team'] == home_team) | (df['Team'] == away_team)]
    df_team = df_team[['Team', 'barthag' ,'Adj OR%', 'Adj DR%', 'Adj eFG%', 'Adj 2p%', 'Adj 3P%', 'Adj TO% Def.', 'Adj eFG% Def', 'Adj TO%', 'Adj 2p%D', 'Adj 3pD%','adjt','3P rate','FTR','ft%D','3P rate D']]
    df_team = df_team.set_index('Team')
    #Reorder rows such that home team is first
    if df_team.index[0] == away_team:
        df_team = df_team.reindex([home_team, away_team])
    df_sim = df_team.copy()
    for col in [ 'Adj OR%', 'Adj DR%', 'Adj eFG%', 'Adj 2p%', 'Adj 3P%', 'Adj TO% Def.', 'Adj eFG% Def', 'Adj TO%', 'Adj 2p%D', 'Adj 3pD%','3P rate','FTR','ft%D','3P rate D']:
        df_team[col] = df_team[col] / 100
        df_sim[col] = df_sim[col] / 100
        df_sim[col] = get_beta_value(df_sim,col,home_team,away_team)
    return df_sim

def game_spread_ppp (home_team, away_team, teams_df, ftr_df, use_bart = True):

    #For each team take the harmonic mean of 
    df_game = pd.DataFrame()
    df_game['Team'] = [home_team, away_team]

    
    #Get free throw rates for home and away teams
    home_ftr = ftr_df[ftr_df['School'] == home_team]['FT%'].values[0]
    away_ftr = ftr_df[ftr_df['School'] == away_team]['FT%'].values[0]

    #Get strength differences
    h_w = 1
    home_advantage = (teams_df.loc[home_team]['barthag'] - teams_df.loc[away_team]['barthag'])/2.5 + 1
    if use_bart:
        home_advantage = home_advantage
    else:
        home_advantage = 1
    a_w = 1#(teams_df.loc[away_team]['barthag'] - teams_df.loc[home_team]['barthag']) + 1
    
    #Get harmonic means for each team
    df_game['ORB'] = [harmonic_mean(teams_df.loc[home_team]['Adj OR%'],teams_df.loc[away_team]['Adj DR%']),
                    harmonic_mean(teams_df.loc[away_team]['Adj OR%'],teams_df.loc[home_team]['Adj DR%'])]
    df_game['2P%'] = [harmonic_mean(teams_df.loc[home_team]['Adj 2p%'],teams_df.loc[away_team]['Adj 2p%D']),
                    harmonic_mean(teams_df.loc[away_team]['Adj 2p%'],teams_df.loc[home_team]['Adj 2p%D'])]
    df_game['3P%'] = [harmonic_mean(teams_df.loc[home_team]['Adj 3P%'],teams_df.loc[away_team]['Adj 3pD%']),
                    harmonic_mean(teams_df.loc[away_team]['Adj 3P%'],teams_df.loc[home_team]['Adj 3pD%'])]
    df_game['TOR'] = [harmonic_mean(teams_df.loc[home_team]['Adj TO%'],teams_df.loc[away_team]['Adj TO% Def.']),
                    harmonic_mean(teams_df.loc[away_team]['Adj TO%'],teams_df.loc[home_team]['Adj TO% Def.'])]
    df_game['FTR'] = [harmonic_mean(teams_df.loc[home_team]['FTR'],teams_df.loc[away_team]['ft%D']),
                    harmonic_mean(teams_df.loc[away_team]['FTR'],teams_df.loc[home_team]['ft%D'])]
    df_game['Adj FTR'] = df_game['FTR'] * .44
    df_game['FT%'] = [home_ftr, away_ftr]
    df_game['3PR'] = [harmonic_mean(teams_df.loc[home_team]['3P rate'],teams_df.loc[away_team]['3P rate D']),
                    harmonic_mean(teams_df.loc[away_team]['3P rate'],teams_df.loc[home_team]['3P rate D'])]
    df_game['ev_2pt'] = df_game['2P%'] * (1 - df_game['3PR']) * 2
    df_game['ev_3pt'] = df_game['3P%'] * df_game['3PR'] * 3
    df_game['Adj ORB'] = ((df_game['2P%'] * (1 - df_game['3PR']) + (df_game['3P%'] * df_game['3PR']))) * df_game['ORB']    
    df_game['shot attempts'] = (((1-df_game['TOR'] - df_game['Adj FTR'] - df_game['Adj ORB']) * 1)) + (df_game['Adj ORB']*2)
    df_game['adjt'] = [teams_df.loc[home_team]['adjt'], teams_df.loc[away_team]['adjt']]
    df_game['PPP'] = df_game['shot attempts'] * df_game['ev_2pt'] + df_game['shot attempts'] * df_game['ev_3pt'] + (df_game['Adj FTR'] * df_game['FT%'] * 1.5)
    df_game['Est FTA'] = (df_game['Adj FTR'] * 1.5 * df_game['adjt']) 

    possessions = (teams_df.loc[home_team]['adjt']+teams_df.loc[away_team]['adjt'])/2

    df_game['Predicted Score'] = df_game['PPP'] * possessions

    #Update home team to multiply by home_advantage if use_bart = True

    df_game.loc[0,'Predicted Score'] = df_game.loc[0,'Predicted Score'] * home_advantage

    #Get projected score for home team

    #print('Projected Spread Home Team: ', df_game.loc[1]['Predicted Score'] - (df_game.loc[0]['Predicted Score'] + 3))
    #print('Projected Total: ', df_game.loc[0]['Predicted Score'] + df_game.loc[1]['Predicted Score'])

    return df_game

def simulate_game(home_team, away_team,num_sims = 1000,home_adjustment = 3,away_adjustment = 0,df=None, ftr_df=None, use_bart = True):
    home_wins = 0
    away_wins = 0
    spread = []
    totals = []
    ties = 0
    for i in range(num_sims):
        df_team = get_simulated_team_stats(home_team, away_team, df)
        df_game = game_spread_ppp(home_team, away_team, df_team,ftr_df, use_bart)
        #home_advantage = (df_team.loc[home_team]['Barthag'] - df_team.loc[away_team]['Barthag'])/2
        if (df_game.loc[0]['Predicted Score'] + home_adjustment)  > df_game.loc[1]['Predicted Score'] + away_adjustment:
            home_wins += 1
        elif (df_game.loc[0]['Predicted Score'] + home_adjustment) < df_game.loc[1]['Predicted Score'] + away_adjustment:
            away_wins += 1
        else:
            ties += 1
        spread.append(df_game.loc[1]['Predicted Score'] + away_adjustment - ((df_game.loc[0]['Predicted Score'] + home_adjustment) ))
        totals.append((df_game.loc[0]['Predicted Score'] + home_adjustment)  + df_game.loc[1]['Predicted Score'])
    
    print(home_team + ' Wins: ', home_wins/num_sims)
    print(away_team + ' Wins: ', away_wins/num_sims)
    print('Ties: ', ties/num_sims)
    print('Median Spread: ', np.median(spread))
    print('Median Total: ', np.median(totals))
    print('Average Spread: ', np.mean(spread))
    print('Average Total: ', np.mean(totals))


    #plt.hist(spread, bins=50)
    #plt.show()
    #plt.hist(totals, bins=50)
    #plt.show()

    return spread,totals

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

# ...existing code...
import os
from pathlib import Path
import pandas as pd

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
    out_path = base / "matchups_bart_results.csv"
    ftr_path = base / "Data/FT 2026.csv"

    if not torvik_path.exists():
        print(f"Missing file: {torvik_path}")
        return
    if not matchups_path.exists():
        print(f"Missing file: {matchups_path}")
        return

    torvik = load_and_prepare_dataframe(torvik_path)
    matchups = pd.read_csv(matchups_path)
    ftr = pd.read_csv(ftr_path)

    #Replace state with St. in ftr if not N.C. State
    ftr['School'] = ftr['School'].str.replace('State','St.')

    #If name N.C. St. replace with N.C. State
    ftr['School'] = ftr['School'].str.replace('N.C. St.','N.C. State')

    #Trim spaces off school names at end
    ftr['School'] = ftr['School'].str.strip()

    home_col, away_col = _find_home_away_cols(matchups)
    if home_col is None:
        print("Couldn't find home/away columns in matchups_today.csv")
        return

    results_rows = []
    for _, row in matchups.iterrows():
        home_name = str(row[home_col]).strip()
        away_name = str(row[away_col]).strip()

        try:
            home_stats = get_regular_team_stats(home_name, away_name, torvik).loc[home_name]  
            away_stats = get_regular_team_stats(home_name, away_name, torvik).loc[away_name] 
            # call your BART simulation function — replace 'simulate_game_bart' with the actual function name
            spreads,totals = simulate_game(home_name, away_name, num_sims=1000, home_adjustment=3, df=torvik, ftr_df=ftr, use_bart=True)

            results_rows.append({
                'home_team': home_name,
                'away_team': away_name,
                'mean_spread_away_minus_home': np.mean(spreads),
                'median_spread': np.median(spreads),
                'std_spread': np.std(spreads),
                'mean_total': np.mean(totals),
                'home_win_prob': np.mean(np.array(spreads) > 0)
            })
            print(f"{home_name} vs {away_name} -> mean spread: {np.mean(spreads)}, home win prob: {np.mean(np.array(spreads) < 0)}")

        except Exception as e:
            print(f"Error simulating {home_name} vs {away_name}: {e}")
            results_rows.append({'home_team': home_name, 'away_team': away_name, 'error': str(e)})

    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote BART simulation results to {out_path}")

if __name__ == "__main__":
    main()