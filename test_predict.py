"""Smoke tests for predict.py using the small sample CSVs.

Run:  python test_predict.py
"""

import pandas as pd

import predict


def test_sample_matches_load():
    df = pd.read_csv("sample_matches.csv")
    assert len(df) == 10
    assert "winner" in df.columns
    print("[ok] sample_matches.csv loads (%d rows)" % len(df))


def test_sample_deliveries_load():
    df = pd.read_csv("sample_deliveries.csv")
    assert "total_runs" in df.columns
    assert df["total_runs"].sum() > 0
    print("[ok] sample_deliveries.csv loads (%d rows)" % len(df))


def test_train_and_predict_on_sample():
    df = pd.read_csv("sample_matches.csv").dropna(subset=[predict.TARGET]).copy()
    df["city"] = df["city"].fillna("Unknown")
    encoders, target_enc = predict.build_encoders(df)

    # Tiny dataset: train on all of it just to exercise the pipeline.
    x_cols = [c + "_enc" for c in predict.FEATURES]
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(df[x_cols], df[predict.TARGET + "_enc"])

    match = {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "bat",
        "venue": "Wankhede Stadium",
        "city": "Mumbai",
    }
    winner = predict.predict_match(model, encoders, target_enc, match)
    assert winner in set(df[predict.TARGET])
    print("[ok] predict_match returns a known team: %s" % winner)


def test_unseen_label_is_tolerated():
    df = pd.read_csv("sample_matches.csv").copy()
    df["city"] = df["city"].fillna("Unknown")
    encoders, _ = predict.build_encoders(df)
    # A team never seen in training should encode to the fallback (0), not crash.
    val = predict._encode_value(encoders["team1"], "Some New Team")
    assert val == 0
    print("[ok] unseen labels fall back without error")


if __name__ == "__main__":
    test_sample_matches_load()
    test_sample_deliveries_load()
    test_train_and_predict_on_sample()
    test_unseen_label_is_tolerated()
    print("\nAll tests passed.")
