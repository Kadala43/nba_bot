import logging
import requests
import pandas as pd
import time
import os
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes


# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Config ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
EDGE_THRESHOLD = 0

# --- Global caches ---
odds_cache = {"data": None, "timestamp": 0}
prediction_cache = {}

# --- Build Models ---
def build_models():
    gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00')
    games = gamefinder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games[games['GAME_DATE'].dt.year >= 2023]

    avg_scores = games.groupby('TEAM_ID')['PTS'].mean().to_dict()
    games_home = games[games['MATCHUP'].str.contains('vs.')].copy()
    games_away = games[games['MATCHUP'].str.contains('@')].copy()
    merged = pd.merge(games_home, games_away, on='GAME_ID', suffixes=('_HOME', '_AWAY'))

    merged['HOME_AVG'] = merged['TEAM_ID_HOME'].map(avg_scores)
    merged['AWAY_AVG'] = merged['TEAM_ID_AWAY'].map(avg_scores)
    merged['TOTAL_SCORE'] = merged['PTS_HOME'] + merged['PTS_AWAY']
    merged['HOME_WIN'] = (merged['PTS_HOME'] > merged['PTS_AWAY']).astype(int)

    # Totals model
    X_totals = merged[['HOME_AVG', 'AWAY_AVG']].dropna()
    y_totals = merged['TOTAL_SCORE'].dropna()
    totals_model = RandomForestRegressor(n_estimators=100, random_state=42)
    totals_model.fit(X_totals, y_totals)

    # Winner classifier
    X_winner = merged[['HOME_AVG', 'AWAY_AVG']].dropna()
    y_winner = merged['HOME_WIN']
    winner_model = RandomForestClassifier(n_estimators=200, random_state=42)
    winner_model.fit(X_winner, y_winner)

    team_map = games[['TEAM_ID', 'TEAM_NAME']].drop_duplicates().set_index('TEAM_NAME')['TEAM_ID'].to_dict()
    return totals_model, winner_model, avg_scores, team_map

totals_model, winner_model, avg_scores, team_map = build_models()

# --- Vegas Odds Cache ---
def fetch_vegas_odds():
    if odds_cache["data"] and time.time() - odds_cache["timestamp"] < 600:  # 10 min cache
        return odds_cache["data"]

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []

    data = response.json()
    odds_cache["data"] = data
    odds_cache["timestamp"] = time.time()
    return data

# --- Prediction Cache ---
def get_total_prediction(home_team, away_team):
    key = (home_team, away_team)
    if key in prediction_cache:
        return prediction_cache[key]

    h_id = team_map.get(home_team)
    a_id = team_map.get(away_team)
    if not h_id or not a_id:
        return None

    h_avg = avg_scores.get(h_id, 110)
    a_avg = avg_scores.get(a_id, 110)
    my_pred = totals_model.predict([[h_avg, a_avg]])[0]

    prediction_cache[key] = my_pred
    return my_pred

def get_winner_prediction(home_team, away_team):
    h_id = team_map.get(home_team)
    a_id = team_map.get(away_team)
    if not h_id or not a_id:
        return None

    h_avg = avg_scores.get(h_id, 110)
    a_avg = avg_scores.get(a_id, 110)

    probs = winner_model.predict_proba([[h_avg, a_avg]])[0]
    home_prob, away_prob = probs[1], probs[0]
    winner = home_team if home_prob > away_prob else away_team
    return winner, home_prob, away_prob

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to NBA Predictor Bot 🏀\n"
        "Type /today to see today's matchups with totals vs Vegas and winner predictions."
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vegas_data = fetch_vegas_odds()
    if not vegas_data:
        await update.message.reply_text("Could not fetch Vegas odds. Check API key or limits.")
        return

    output_lines = []
    for game in vegas_data:
        home_team = game['home_team']
        away_team = game['away_team']

        if not game['bookmakers']:
            continue

        odds_line = None
        for bookie in game['bookmakers']:
            for market in bookie['markets']:
                if market['key'] == 'totals':
                    odds_line = market['outcomes'][0]['point']
                    break
            if odds_line:
                break

        if odds_line is None:
            continue

        my_pred = get_total_prediction(home_team, away_team)
        if my_pred is None:
            continue

        diff = my_pred - odds_line
        recommendation = None
        if abs(diff) >= EDGE_THRESHOLD:
            recommendation = "OVER" if diff > 0 else "UNDER"

        winner_info = get_winner_prediction(home_team, away_team)
        if not winner_info:
            continue
        winner, home_prob, away_prob = winner_info

        line = (
            f"{away_team} @ {home_team}: Pred {my_pred:.1f}, Vegas {odds_line:.1f}"
            + (f" → {recommendation} (Edge {diff:+.1f})" if recommendation else "")
            + f"\n🏆 Winner Prediction: {winner} "
            f"(Home {home_prob*100:.1f}% | Away {away_prob*100:.1f}%)"
        )
        output_lines.append(line)

    if output_lines:
        await update.message.reply_text("\n\n".join(output_lines))
    else:
        await update.message.reply_text("No strong edges today.")

# --- Run Bot ---
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("today", today))
app.run_polling()