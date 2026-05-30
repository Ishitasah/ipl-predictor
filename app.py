import streamlit as st
import predict  # your existing predict.py

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏")
st.title("🏏 IPL Match Winner Predictor")
st.caption("Trained on 752 historical IPL matches with a RandomForest model.")

# Train once and cache it so it doesn't retrain on every click.
@st.cache_resource
def load_model():
    df = predict.load_matches()
    encoders, target_enc = predict.build_encoders(df)
    model, acc = predict.train(df, encoders, target_enc)
    return df, encoders, target_enc, model, acc

df, encoders, target_enc, model, acc = load_model()
st.success(f"Model ready — test accuracy: {acc*100:.1f}%")

# Build dropdown options from the real data (sorted, readable).
teams = sorted(set(df["team1"]) | set(df["team2"]))
venues = sorted(df["venue"].unique())
cities = sorted(df["city"].unique())

st.subheader("Enter match details")
col1, col2 = st.columns(2)
team1 = col1.selectbox("Team 1", teams)
team2 = col2.selectbox("Team 2", teams, index=1)
toss_winner = st.selectbox("Toss winner", [team1, team2])
toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)
venue = st.selectbox("Venue", venues)
city = st.selectbox("City", cities)

if st.button("Predict winner", type="primary"):
    if team1 == team2:
        st.error("Team 1 and Team 2 must be different.")
    else:
        match = {
            "team1": team1, "team2": team2,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "venue": venue, "city": city,
        }
        winner = predict.predict_match(model, encoders, target_enc, match)
        st.markdown(f"### 🏆 Predicted winner: **{winner}**")
        st.caption("Note: ~49% accuracy. Cricket has high variance — treat as a guess, not gospel.")
