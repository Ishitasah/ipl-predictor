"""IPL match-winner predictor.

Trains a RandomForest classifier on matches.csv to predict the winning team
from pre-match information (the two teams, toss outcome, venue and city).

Usage:
    python predict.py                # train, evaluate, run a demo prediction
"""

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Features known *before* a ball is bowled -- safe to use for prediction.
FEATURES = ["team1", "team2", "toss_winner", "toss_decision", "venue", "city"]
TARGET = "winner"


def find_csv(name):
    """Return a usable path for ``name``.

    The repo ships ``deliveries.csv`` as a *folder* that contains the real
    file (deliveries.csv/deliveries.csv).  This resolves either layout so the
    script works no matter which is on disk.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    direct = os.path.join(here, name)
    if os.path.isfile(direct):
        return direct
    nested = os.path.join(direct, name)  # e.g. deliveries.csv/deliveries.csv
    if os.path.isdir(direct) and os.path.isfile(nested):
        return nested
    raise FileNotFoundError("Could not locate %s near %s" % (name, here))


def load_matches():
    df = pd.read_csv(find_csv("matches.csv"))
    # Drop matches with no result (e.g. abandoned games) -- nothing to learn.
    df = df.dropna(subset=[TARGET]).copy()
    # Fill the few missing cities so the encoder has a value to work with.
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Unknown")
    return df


def build_encoders(df):
    """One LabelEncoder per categorical column, fit on all seen values."""
    encoders = {}
    for col in FEATURES:
        enc = LabelEncoder()
        df[col + "_enc"] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc
    target_enc = LabelEncoder()
    df[TARGET + "_enc"] = target_enc.fit_transform(df[TARGET].astype(str))
    return encoders, target_enc


def train(df, encoders, target_enc):
    x_cols = [c + "_enc" for c in FEATURES]
    X = df[x_cols]
    y = df[TARGET + "_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


def _encode_value(enc, value):
    """Encode a single value, tolerating labels unseen during training."""
    value = str(value)
    if value in enc.classes_:
        return int(enc.transform([value])[0])
    return 0  # fall back to the first known class


def predict_match(model, encoders, target_enc, match):
    row = {c + "_enc": [_encode_value(encoders[c], match.get(c, ""))] for c in FEATURES}
    X = pd.DataFrame(row)
    pred = model.predict(X)[0]
    return target_enc.inverse_transform([pred])[0]


def main():
    df = load_matches()
    print("Loaded %d matches with a recorded winner." % len(df))
    encoders, target_enc = build_encoders(df)
    model, acc = train(df, encoders, target_enc)
    print("Test accuracy: %.1f%%" % (acc * 100))

    demo = {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "bat",
        "venue": "Wankhede Stadium",
        "city": "Mumbai",
    }
    winner = predict_match(model, encoders, target_enc, demo)
    print("\nDemo prediction")
    print("  %s vs %s (toss: %s chose to %s)"
          % (demo["team1"], demo["team2"], demo["toss_winner"], demo["toss_decision"]))
    print("  -> predicted winner: %s" % winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
