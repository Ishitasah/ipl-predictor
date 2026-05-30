"""IPL match-winner predictor (v2).

Predicts whether *team1 beats team2* (a binary question, so the answer is always
one of the two teams actually playing) and returns a win probability.

Instead of feeding raw team IDs to the model, we engineer features that carry
real signal:

    team1_winrate / team2_winrate  - each team's historical win rate
    winrate_diff                   - strength gap between the two
    h2h_team1                      - team1's win rate in past head-to-heads
    toss_team1                     - did team1 win the toss?
    toss_field                     - did the toss winner choose to field?
    venue_enc                      - the ground (label-encoded)

Usage:
    python predict.py            # train, report accuracy, run a demo prediction
"""

import os
import sys
from collections import defaultdict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURE_NAMES = [
    "team1_winrate", "team2_winrate", "winrate_diff",
    "h2h_team1", "toss_team1", "toss_field", "venue_enc",
]
TARGET = "winner"


def find_csv(name):
    """Locate ``name`` on disk (handles deliveries.csv shipped as a folder)."""
    here = os.path.dirname(os.path.abspath(__file__))
    direct = os.path.join(here, name)
    if os.path.isfile(direct):
        return direct
    nested = os.path.join(direct, name)
    if os.path.isdir(direct) and os.path.isfile(nested):
        return nested
    raise FileNotFoundError("Could not locate %s near %s" % (name, here))


def load_matches():
    df = pd.read_csv(find_csv("matches.csv"))
    df = df.dropna(subset=[TARGET]).copy()       # drop abandoned / no-result games
    df["city"] = df["city"].fillna("Unknown")
    return df


def compute_team_winrate(df):
    """Overall win rate for every team across the dataset."""
    games = defaultdict(lambda: [0, 0])          # team -> [wins, played]
    for t1, t2, w in zip(df["team1"], df["team2"], df[TARGET]):
        games[t1][1] += 1
        games[t2][1] += 1
        games[w][0] += 1
    return {t: (g[0] / g[1] if g[1] else 0.5) for t, g in games.items()}


def compute_h2h(df):
    """Head-to-head record: rec[(A, B)] = [A's wins over B, total A-vs-B games]."""
    rec = defaultdict(lambda: [0, 0])
    for t1, t2, w in zip(df["team1"], df["team2"], df[TARGET]):
        rec[(t1, t2)][1] += 1
        rec[(t2, t1)][1] += 1
        if w == t1:
            rec[(t1, t2)][0] += 1
        elif w == t2:
            rec[(t2, t1)][0] += 1
    return rec


def _h2h_rate(rec, a, b):
    wins, total = rec[(a, b)]
    return wins / total if total else 0.5


def _feature_row(team1, team2, toss_winner, toss_decision, venue,
                 winrate, h2h, venue_classes):
    w1 = winrate.get(team1, 0.5)
    w2 = winrate.get(team2, 0.5)
    venue_enc = venue_classes.index(venue) if venue in venue_classes else -1
    return {
        "team1_winrate": w1,
        "team2_winrate": w2,
        "winrate_diff": w1 - w2,
        "h2h_team1": _h2h_rate(h2h, team1, team2),
        "toss_team1": 1 if toss_winner == team1 else 0,
        "toss_field": 1 if toss_decision == "field" else 0,
        "venue_enc": venue_enc,
    }


def build_model(df):
    """Engineer features, train the model, and bundle everything prediction needs."""
    winrate = compute_team_winrate(df)
    h2h = compute_h2h(df)
    venue_classes = sorted(df["venue"].unique())

    X = pd.DataFrame([
        _feature_row(r.team1, r.team2, r.toss_winner, r.toss_decision, r.venue,
                     winrate, h2h, venue_classes)
        for r in df.itertuples()
    ])[FEATURE_NAMES]
    y = (df["team1"] == df[TARGET]).astype(int).values   # 1 = team1 won

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return {
        "model": model,
        "winrate": winrate,
        "h2h": h2h,
        "venue_classes": venue_classes,
        "accuracy": acc,
        "feature_names": FEATURE_NAMES,
    }


def predict_match(bundle, team1, team2, toss_winner, toss_decision, venue):
    """Return (winner, probability_team1_wins)."""
    row = _feature_row(team1, team2, toss_winner, toss_decision, venue,
                       bundle["winrate"], bundle["h2h"], bundle["venue_classes"])
    X = pd.DataFrame([row])[bundle["feature_names"]]
    classes = list(bundle["model"].classes_)
    proba = bundle["model"].predict_proba(X)[0]
    p1 = proba[classes.index(1)] if 1 in classes else 0.0
    winner = team1 if p1 >= 0.5 else team2
    return winner, p1


def main():
    df = load_matches()
    print("Loaded %d matches with a recorded winner." % len(df))
    bundle = build_model(df)
    print("Test accuracy: %.1f%%" % (bundle["accuracy"] * 100))

    winner, p1 = predict_match(
        bundle, "Mumbai Indians", "Chennai Super Kings",
        "Mumbai Indians", "bat", "Wankhede Stadium",
    )
    print("\nDemo: Mumbai Indians vs Chennai Super Kings")
    print("  -> %s wins  (Mumbai win probability: %.0f%%)" % (winner, p1 * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
