import pandas as pd

df = pd.read_csv('http://barttorvik.com/2026_team_results.csv',index_col=False)
df_results = df[['rank','team', 'conf','record','adjoe','adjde','barthag', 'sos','adjt']]
df_results.head()

df = pd.read_csv('http://barttorvik.com/2026_fffinal.csv',index_col=False)
df.head()

import re

cols = list(df.columns)
new_cols = cols.copy()

for i, col in enumerate(cols):
    # match 'Rk', 'Rk.1', 'rk', 'rk.2', etc. (case-insensitive)
    if re.match(r'^rk(?:\.\d+)?$', str(col), re.I):
        if i == 0:
            # nothing before it to use as metric name
            continue
        metric_name = str(cols[i - 1]).strip()
        # create new rank column name based on previous metric
        new_name = f"{metric_name} Rank"
        # avoid accidental duplicate column names
        if new_name in new_cols:
            suffix = 1
            candidate = f"{new_name} ({suffix})"
            while candidate in new_cols:
                suffix += 1
                candidate = f"{new_name} ({suffix})"
            new_name = candidate
        new_cols[i] = new_name

df.columns = new_cols

#Merge results and df
merged_df = pd.merge(df, df_results, how='left', left_on='TeamName', right_on='team')
merged_df.head()

merged_df = merged_df.apply(pd.to_numeric, errors='ignore')
print(merged_df.head())

#Remove all columns with word rank in it
clean_df = merged_df[merged_df.columns.drop(list(merged_df.filter(regex='Rank')))]
clean_df['Year'] = 2026
#Normalize each metric by calculating the z-score for each year except adj metrics
clean_df['EFG Z'] = clean_df.groupby('Year')['eFG%'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['EFGD Z'] = clean_df.groupby('Year')['eFG% Def'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['TOR Z'] = clean_df.groupby('Year')['TO%'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['TORD Z'] = clean_df.groupby('Year')['TO% Def.'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['ORB Z'] = clean_df.groupby('Year')['OR%'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['DRB Z'] = clean_df.groupby('Year')['DR%'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['FTR Z'] = clean_df.groupby('Year')['FTR'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['FTRD Z'] = clean_df.groupby('Year')['FTR Def'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['2P Z'] = clean_df.groupby('Year')['2p%'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['2PD Z'] = clean_df.groupby('Year')['2p%D'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['3P Z'] = clean_df.groupby('Year')['3P%'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['3PD Z'] = clean_df.groupby('Year')['3pD%'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['3PR Z'] = clean_df.groupby('Year')['3P rate'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['3PRD Z'] = clean_df.groupby('Year')['3P rate D'].transform(lambda x: (x - x.mean()) / x.std())
clean_df['Adj T Z'] = clean_df.groupby('Year')['adjt'].transform(lambda x: (x - x.mean()) / x.std())

#Rename TeamName to Team
clean_df = clean_df.rename(columns={'TeamName': 'Team'})

#Output to csv with name torvik_ and then today's date
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create the filename with the timestamp
filename = f"Data/latest_torvik.csv"

# Save the dataframe to a csv file
clean_df.to_csv(filename, index=False)


## Get matchups
import pandas as pd
import requests
from datetime import date

# 1. Configuration
# Replace this URL with the specific sports schedule page you want to scrape.
# Example: ESPN's Men's College Basketball Schedule for today
# NOTE: The date in the URL might need to be adjusted for 'today'
# The search tool result (if a schedule URL was found) would be ideal here.
today_str = date.today().strftime('%Y%m%d')
SCHEDULE_URL = "https://www.espn.com/mens-college-basketball/schedule/_/date/" + today_str

# Get today's date for the filename
today_str = date.today().strftime("%Y-%m-%d")
OUTPUT_FILENAME = "Data/matchups_today.csv"


def scrape_and_save_schedule(url, filename):
    """
    Fetches the HTML from the given URL and uses pandas to read all tables.
    It combines all found tables into a single DataFrame and saves it as a CSV.
    """
    print(f"-> Attempting to scrape tables from: {url}")
    
    try:
        # Use requests to get the content of the page
        # Note: Some sites might block simple 'requests' and require setting a User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # 2. Use pandas to automatically read all tables in the HTML
        # This is the most powerful part of this script.
        tables = pd.read_html(response.text)

        if not tables:
            print("-> No HTML tables were found on the page.")
            return

        print(f"-> Successfully found {len(tables)} tables on the page.")

        # 3. Combine Tables (if more than one is found)
        # You may need to inspect the tables and filter them. For simplicity, we concatenate all.
        try:
            # Drop the first row of all tables as it is often a repeated header/title
            processed_tables = [table.iloc[1:] for table in tables]
            
            # Concatenate all processed tables into a single DataFrame
            final_df = pd.concat(processed_tables, ignore_index=True)
            
        except Exception as e:
             print(f"-> Error during table processing (concatenation/row drop): {e}")
             # Fallback to just using the largest table if concatenation fails
             final_df = max(tables, key=len)


        # 4. Clean up and Save
        # Remove any columns that are entirely NaN (empty)
        final_df.dropna(axis=1, how='all', inplace=True)
        
        # Save the DataFrame to a CSV file
        final_df.to_csv(filename, index=False, encoding='utf-8')
        
        print("-" * 50)
        print(f"✅ Success! Matchups saved to: {filename}")
        print(f"A total of {len(final_df)} rows were written.")
        print("-" * 50)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching URL or network issue: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

scrape_and_save_schedule(SCHEDULE_URL, OUTPUT_FILENAME)

cbb_matchups = pd.read_csv(OUTPUT_FILENAME)
cbb_matchups = cbb_matchups[['MATCHUP','MATCHUP.1']]
cbb_matchups = cbb_matchups.rename(columns={'MATCHUP': 'Away', 'MATCHUP.1': 'Home'})

#Remove @ from Away and trim
cbb_matchups['Home'] = cbb_matchups['Home'].str.replace('@', '').str.strip()

#Remove digits from home and away
cbb_matchups['Away'] = cbb_matchups['Away'].str.replace(r'\d+', '', regex=True).str.strip()
cbb_matchups['Home'] = cbb_matchups['Home'].str.replace(r'\d+', '', regex=True).str.strip()

#Replate "State" with "St."
cbb_matchups['Away'] = cbb_matchups['Away'].str.replace('State', 'St.')
cbb_matchups['Home'] = cbb_matchups['Home'].str.replace('State', 'St.') 

# Function to remove the last word
def remove_last_word(text,teams):
    if text == 'Over' or text == 'Under':
        return text
    if text in teams:
        return ' '.join(text.split()[:-2])
    else:
        return ' '.join(text.split()[:-1])
    
def convert_decimal_to_american(decimal_odds):
    if decimal_odds < 2:
        return -100 / (decimal_odds - 1)
    else:
        return decimal_odds - 1

def scrape_odds(filename):
    import requests
    import pandas as pd

    # Replace with your actual API key
    API_KEY = "391d2c632b433f4ed3a318ad2bdb814e"
    SPORT = "basketball_ncaab"  # Sport key for NCAAB
    REGION = "us"  # Odds for the US market
    #BOOKMAKERS = ["draftkings", "fanduel", "espn"]  # Add/remove sportsbooks

    # API URL
    URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"

    # Request Parameters
    params = {
        "api_key": API_KEY,
        "regions": REGION,
        "markets": "spreads" #"h2h,spreads,totals"  # Moneyline, spreads, and over/under
        #"bookmakers": ",".join(BOOKMAKERS),
    }

    # Make API request
    response = requests.get(URL, params=params)

    if response.status_code == 200:
        odds_data = response.json()

        # Convert to DataFrame
        games = []
        for game in odds_data:
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        games.append({
                            "Game": f"{game['home_team']} vs {game['away_team']}",
                            "Bookmaker": bookmaker["key"],
                            "Market": market["key"],
                            "Team": outcome.get("name"),
                            "Odds": outcome.get("price"),
                            "Point": outcome.get("point")  # For spreads/totals
                        })

        df = pd.DataFrame(games)

    else:
        print(f"Error: {response.status_code}, {response.text}")

    #Create list of teams with two names
    teams = ['Southern Miss Golden Eagles','Maine Black Bears', 'UMass Lowell River Hawks', 'Albany Great Danes', 'Georgia Tech Yellow Jackets', 'Oakland Golden Grizzlies', 'Kent State Golden Flashes'
         ,'Lehigh Mountain Hawks','North Carolina Tar Heels','Rutgers Scarlet Knights','TCU Horned Frogs','Tulsa Golden Hurricane',"Louisiana Ragin' Cajuns",'Nevada Wolf Pack','Arizona St Sun Devils',
         'DePaul Blue Demons','Alabama Crimson Tide','Central Connecticut St Blue Devils','Tennessee Tech Golden Eagles','Texas Tech Red Raiders','Oral Roberts Golden Eagles'
         ,'St. Francis (PA) Red Flash','Marquette Golden Eagles','Minnesota Golden Gophers','California Golden Bears','Notre Dame Fighting Irish','Illinois Fighting Illini','Texas Tech Red Raiders'
         ,'Duke Blue Devils']

    # Apply the function to the DataFrame column
    df['Team Modified'] = df['Team'].apply(lambda x: remove_last_word(x, teams))

    df['Team Modified'] = df['Team Modified'].str.replace('State', 'St')
    df['Team Modified'] = df['Team Modified'].str.replace('St', 'St.')

    df['Implied Probability'] = 1 / df['Odds']

    #Set vegas line column by converting decimal odds to american odds
    df['Vegas Line'] = df['Odds'].apply(convert_decimal_to_american)
    final_odds = df[['Team Modified', 'Market', 'Bookmaker', 'Point', 'Implied Probability', 'Vegas Line']]
    final_odds.to_csv(filename, index=False) 

scrape_odds("Data/odds_today.csv")

cbb_matchups.to_csv(OUTPUT_FILENAME, index=False)