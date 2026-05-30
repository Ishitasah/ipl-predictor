import pandas as pd
import streamlit as st

import predict

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="centered")


@st.cache_resource
def load():
    df = predict.load_matches()
    bundle = predict.build_model(df)
    return df, bundle


df, bundle = load()

st.title("🏏 IPL Match Winner Predictor")
st.caption(
    "RandomForest trained on %d historical IPL matches · test accuracy %.1f%%"
    % (len(df), bundle["accuracy"] * 100)
)

teams = sorted(set(df["team1"]) | set(df["team2"]))
venues = sorted(df["venue"].unique())


def _default(name, fallback=0):
    return teams.index(name) if name in teams else fallback


st.subheader("Match details")
c1, c2 = st.columns(2)
team1 = c1.selectbox("Team 1", teams, index=_default("Mumbai Indians", 0))
team2 = c2.selectbox("Team 2", teams, index=_default("Chennai Super Kings", 1))
toss_winner = st.selectbox("Toss winner", [team1, team2])
toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)
venue = st.selectbox("Venue", venues)

if st.button("Predict winner", type="primary", use_container_width=True):
    if team1 == team2:
        st.error("Pick two different teams.")
    else:
        winner, p1 = predict.predict_match(
            bundle, team1, team2, toss_winner, toss_decision, venue
        )
        p2 = 1 - p1
        st.markdown("## 🏆 %s" % winner)
        st.progress(p1 if winner == team1 else p2)

        col1, col2 = st.columns(2)
        col1.metric(team1, "%.0f%%" % (p1 * 100))
        col2.metric(team2, "%.0f%%" % (p2 * 100))

        # Head-to-head record between the two selected teams.
        t1_wins = bundle["h2h"][(team1, team2)][0]
        t2_wins = bundle["h2h"][(team2, team1)][0]
        meetings = bundle["h2h"][(team1, team2)][1]
        if meetings:
            st.caption(
                "Head-to-head: %s %d – %d %s  (%d past meetings)"
                % (team1, t1_wins, t2_wins, team2, meetings)
            )
        else:
            st.caption("These two teams have no recorded head-to-head meetings.")

with st.expander("What drives the prediction? (feature importance)"):
    fi = pd.DataFrame({
        "feature": bundle["feature_names"],
        "importance": bundle["model"].feature_importances_,
    }).sort_values("importance", ascending=True).set_index("feature")
    st.bar_chart(fi)
    st.caption(
        "Higher = the model relies on it more. Team strength (win rate) and "
        "the head-to-head record usually dominate."
    )

st.divider()
st.caption(
    "Predictions are probabilistic, not guarantees — cricket has high variance. "
    "Built with Python, scikit-learn and Streamlit."
)
