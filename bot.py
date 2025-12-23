import os
import time
import datetime
import logging
import requests
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Config ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
EDGE_THRESHOLD = 3  # min edge to show OVER/UNDER recommendation

# --- Global caches ---
odds_cache = {"data": None, "timestamp": 0}
prediction_cache = {}
predictions_log = {}  # store predictions and evaluation results

# --- Build Models ---
def build_models():
    logger.info("Building models from historical game data...")
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

    totals_model = RandomForestRegressor(n_estimators=120, random_state=42)
    totals_model.fit(merged[['HOME_AVG', 'AWAY_AVG']], merged['TOTAL_SCORE'])

    winner_model = RandomForestClassifier(n_estimators=240, random_state=42)
    winner_model.fit(merged[['HOME_AVG', 'AWAY_AVG']], merged['HOME_WIN'])

    team_map = games[['TEAM_ID', 'TEAM_NAME']].drop_duplicates().set_index('TEAM_NAME')['TEAM_ID'].to_dict()

    logger.info("Models ready: totals regressor + winner classifier.")
    return totals_model, winner_model, avg_scores, team_map

totals_model, winner_model, avg_scores, team_map = build_models()

# --- Vegas Odds Cache ---
def fetch_vegas_odds():
    if odds_cache["data"] and time.time() - odds_cache["timestamp"] < 600:
        return odds_cache["data"]

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error("Odds API request failed: %s", e)
        return []

    data = response.json()
    odds_cache["data"] = data
    odds_cache["timestamp"] = time.time()
    logger.info("Fetched %d games from Odds API", len(data))
    return data

# --- Predictions ---
def get_total_prediction(home_team, away_team):
    key = (home_team, away_team)
    if key in prediction_cache:
        return prediction_cache[key]

    h_id = team_map.get(home_team)
    a_id = team_map.get(away_team)
    if not h_id or not a_id:
        logger.warning("Team not found in team_map: %s vs %s", home_team, away_team)
        return None

    h_avg = avg_scores.get(h_id, 110)
    a_avg = avg_scores.get(a_id, 110)
    my_pred = float(totals_model.predict([[h_avg, a_avg]])[0])
    prediction_cache[key] = my_pred
    return my_pred

def get_winner_prediction(home_team, away_team):
    h_id = team_map.get(home_team)
    a_id = team_map.get(away_team)
    if not h_id or not a_id:
        logger.warning("Team not found for winner prediction: %s vs %s", home_team, away_team)
        return None

    h_avg = avg_scores.get(h_id, 110)
    a_avg = avg_scores.get(a_id, 110)
    probs = winner_model.predict_proba([[h_avg, a_avg]])[0]
    home_prob, away_prob = float(probs[1]), float(probs[0])
    winner = home_team if home_prob > away_prob else away_team
    return winner, home_prob, away_prob

# --- Helpers for results ---
def fetch_finished_games_for_date(date_obj):
    gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00')
    games = gamefinder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    return games[games['GAME_DATE'].dt.date == date_obj].copy()

def compute_actuals_for_matchup(df_day, home_team, away_team):
    rows = df_day[(df_day['TEAM_NAME'] == home_team) | (df_day['TEAM_NAME'] == away_team)]
    if rows.empty or rows['GAME_ID'].nunique() == 0:
        return None, None
    game_id = rows['GAME_ID'].unique()[0]
    game_rows = rows[rows['GAME_ID'] == game_id]
    actual_total = int(game_rows['PTS'].sum())
    actual_winner = game_rows.loc[game_rows['PTS'].idxmax()]['TEAM_NAME']
    return actual_total, actual_winner

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to NBA Predictor Bot 🏀\n"
        "Commands:\n"
        "/today → predictions vs Vegas + winner probabilities\n"
        "/results → check yesterday’s accuracy\n"
        "/summary → overall accuracy stats"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("User %s requested today's predictions", update.effective_user.username)
    vegas_data = fetch_vegas_odds()
    if not vegas_data:
        await update.message.reply_text("Could not fetch Vegas odds. Check API key or limits.")
        return

    output_lines = []
    for game in vegas_data:
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        if not home_team or not away_team:
            continue

        # Pull a totals line from any bookmaker
        odds_line = None
        for bookie in game.get('bookmakers', []):
            for market in bookie.get('markets', []):
                if market.get('key') == 'totals' and market.get('outcomes'):
                    point = market['outcomes'][0].get('point')
                    if point is not None:
                        odds_line = float(point)
                        break
            if odds_line is not None:
                break
        if odds_line is None:
            logger.warning("No totals line for %s @ %s", away_team, home_team)
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

        # Save prediction for later comparison
        predictions_log[f"{away_team}@{home_team}"] = {
            "home_team": home_team,
            "away_team": away_team,
            "pred_total": my_pred,
            "pred_winner": winner,
            "vegas_line": odds_line,
            "checked": False,
            "totals_correct": None,
            "winner_correct": None
        }

        line = (
            f"{away_team} @ {home_team}: Pred {my_pred:.1f}, Vegas {odds_line:.1f}"
            + (f" → {recommendation} (Edge {diff:+.1f})" if recommendation else "")
            + f"\n🏆 Winner Prediction: {winner} "
            f"(Home {home_prob*100:.1f}% | Away {away_prob*100:.1f}%)"
        )
        output_lines.append(line)

    await update.message.reply_text("\n\n".join(output_lines) if output_lines else "No strong edges today.")

async def results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("User %s requested results check", update.effective_user.username)
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).date()
    df_day = fetch_finished_games_for_date(yesterday)
    if df_day.empty:
        await update.message.reply_text("No completed games found for yesterday.")
        return

    output_lines = []
    checked = 0

    for key, pred in predictions_log.items():
        home_team = pred['home_team']
        away_team = pred['away_team']

        actual_total, actual_winner = compute_actuals_for_matchup(df_day, home_team, away_team)
        if actual_total is None:
            continue

        correct_total = (
            (pred['pred_total'] > pred['vegas_line'] and actual_total > pred['vegas_line']) or
            (pred['pred_total'] < pred['vegas_line'] and actual_total < pred['vegas_line'])
        )
        correct_winner = (pred['pred_winner'] == actual_winner)

        # Save evaluation
        pred["checked"] = True
        pred["totals_correct"] = bool(correct_total)
        pred["winner_correct"] = bool(correct_winner)

        line = (
            f"{away_team} @ {home_team}:\n"
            f"Pred Total {pred['pred_total']:.1f}, Actual {actual_total}\n"
            f"Pred Winner {pred['pred_winner']}, Actual {actual_winner}\n"
            f"✅ Totals Correct: {correct_total}, ✅ Winner Correct: {correct_winner}"
        )
        output_lines.append(line)
        checked += 1

    if output_lines:
        await update.message.reply_text("\n\n".join(output_lines))
        logger.info("Reported results for %d matchups", checked)
    else:
        await update.message.reply_text("No matching predictions found for yesterday’s completed games.")

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("User %s requested summary", update.effective_user.username)
    checked_preds = [p for p in predictions_log.values() if p["checked"]]
    if not checked_preds:
        await update.message.reply_text("No evaluated predictions yet. Run /results after games finish.")
        return

    total_games = len(checked_preds)
    totals_hits = sum(1 for p in checked_preds if p["totals_correct"])
    winner_hits = sum(1 for p in checked_preds if p["winner_correct"])

    totals_acc = totals_hits / total_games * 100
    winner_acc = winner_hits / total_games * 100

    msg = (
        f"📊 Prediction Summary:\n"
        f"Games Evaluated: {total_games}\n"
        f"Totals Accuracy: {totals_acc:.1f}% ({totals_hits}/{total_games})\n"
        f"Winner Accuracy: {winner_acc:.1f}% ({winner_hits}/{total_games})"
    )
    await update.message.reply_text(msg)

# --- Error handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling update %s: %s", update, context.error)

# --- Main ---
def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable not set.")
    if not ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set — odds fetch will fail.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("results", results))
    app.add_handler(CommandHandler("summary", summary))
    app.add_error_handler(error_handler)

    logger.info("Bot starting polling...")
    app.run_polling(stop_signals=None)

if __name__ == "__main__":
    main()