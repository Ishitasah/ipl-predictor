"""Smoke tests for predict.py (v2).

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


def test_build_and_predict():
    df = predict.load_matches()
    bundle = predict.build_model(df)
    assert 0.0 <= bundle["accuracy"] <= 1.0
    print("[ok] model trains, accuracy = %.1f%%" % (bundle["accuracy"] * 100))

    winner, p1 = predict.predict_match(
        bundle, "Mumbai Indians", "Chennai Super Kings",
        "Mumbai Indians", "bat", "Wankhede Stadium",
    )
    # The predicted winner must be one of the two teams actually playing.
    assert winner in {"Mumbai Indians", "Chennai Super Kings"}
    assert 0.0 <= p1 <= 1.0
    print("[ok] predict returns a playing team: %s (p=%.2f)" % (winner, p1))


def test_winner_matches_probability():
    """If team1's win prob >= 0.5 the winner is team1, else team2 -- always consistent."""
    df = predict.load_matches()
    bundle = predict.build_model(df)
    for t1, t2 in [("Kolkata Knight Riders", "Rajasthan Royals"),
                   ("Kings XI Punjab", "Delhi Daredevils")]:
        winner, p1 = predict.predict_match(bundle, t1, t2, t1, "field", "Eden Gardens")
        expected = t1 if p1 >= 0.5 else t2
        assert winner == expected
        assert winner in {t1, t2}
    print("[ok] winner is always consistent with the probability and is a playing team")


if __name__ == "__main__":
    test_sample_matches_load()
    test_sample_deliveries_load()
    test_build_and_predict()
    test_winner_matches_probability()
    print("\nAll tests passed.")
