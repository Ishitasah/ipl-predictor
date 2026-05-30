# 🏏 IPL Match Winner Predictor

Predicts which team wins an IPL match — with a **win probability**, not just a
label — using a RandomForest classifier trained on 752 historical matches.

**Live app:** https://ipl-predictor-ishitasah.streamlit.app

## Stack
Python · pandas · scikit-learn · Streamlit

## Run locally
```
pip install -r requirements.txt
python -m streamlit run app.py
```

## How it works
The model predicts a binary outcome — *does team1 beat team2?* — so the
prediction is always one of the two teams actually playing. Rather than feeding
raw team IDs, it uses **engineered features** that carry real signal:

| Feature | Meaning |
|---------|---------|
| `team1_winrate`, `team2_winrate` | each team's historical win rate |
| `winrate_diff` | strength gap between the teams |
| `h2h_team1` | team1's win rate in past head-to-head meetings |
| `toss_team1` | did team1 win the toss? |
| `toss_field` | did the toss winner choose to field? |
| `venue_enc` | the ground |

Pipeline: clean `matches.csv` → engineer features → train RandomForest (80/20
split) → predict win probability via a Streamlit UI with a feature-importance chart.

## Results
~56% test accuracy. Team strength and head-to-head record are the most
important features.

## Limitations & next steps
- Cricket is high-variance; pre-match features have a natural accuracy ceiling.
- Next: recent-form (last-N-games) features, home/away advantage, XGBoost,
  hyperparameter tuning, and a per-season time-aware split to rule out leakage.

## Files
- `predict.py` — data loading, feature engineering, model, prediction
- `app.py` — Streamlit web app
- `test_predict.py` — smoke tests
- `matches.csv` — IPL match data
