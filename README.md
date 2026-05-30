#  IPL Match Winner Predictor

Predicts the winner of an IPL match from pre-match info (teams, toss, venue)
using a RandomForest classifier trained on 752 historical matches.

## Stack
Python · pandas · scikit-learn · Streamlit

## Run locally
pip install -r requirements.txt
streamlit run app.py



## How it works
1. Clean `matches.csv` (drop no-result games, fill missing cities)
2. Encode categorical columns (teams, venue, etc.)
3. Train a RandomForest (80/20 train-test split)
4. Predict via a Streamlit UI

## Limitations & next steps
- ~49% accuracy — pre-match features alone have limited signal.
- Improvements: one-hot encoding for teams, team-form & head-to-head features, XGBoost.