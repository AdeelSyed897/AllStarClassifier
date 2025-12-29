import streamlit as st
import pandas as pd
from player import Player

st.set_page_config(page_title="NBA All-Star Predictor")
st.title("🏀 NBA All-Star Predictor")


st.write(
    "Get ready for All-Star voting! This site uses a logistic regression model I trained to estimate "
    "whether a player is likely to make the All-Star team based on their current stats. "
    "Select a player above to see their model score and key indicators."
)

st.markdown(
    """
**Learn more:** 
- ✍️ Blog: https://ade3lsyednba.hashnode.dev/machine-learning-all-star-selection-classifier  
- 💻 GitHub: https://github.com/AdeelSyed897/NBA-All-Star-ML 
- 🎥 YouTube: https://www.youtube.com/watch?v=3MpH5tPh5vs&t=327s  
    """
)

st.caption(
    "Data source: Basketball-Reference : www.basketball-reference.com"
)


df = pd.read_csv("data/2026.csv")
players = sorted(df["Player"].unique())
playerName = st.selectbox("",players,index=None,placeholder="Input Name")


if playerName:
    player = Player(playerName)

    with st.container(border=True):
        st.subheader(playerName)

        top = st.columns(4)
        top[0].metric("PTS", f"{player.pts:.1f}")
        top[1].metric("REB", f"{player.reb:.1f}")
        top[2].metric("AST", f"{player.ast:.1f}")
        top[3].metric("eFG%", f"{player.efg * 100:.1f}%")

        bottom = st.columns(4)
        bottom[0].metric("STL", f"{player.stl:.1f}")
        bottom[1].metric("BLK", f"{player.blk:.1f}")
        bottom[2].metric("MP", f"{player.mp:.1f}")
        bottom[3].metric("GS", f"{player.gs:.0f}")

        st.divider()
    
        st.subheader("All-Star Status")
        status_col, prob_col = st.columns([2, 1])

        prob = player.AllStarProb()
        if prob >= 0.3434:
            status_col.success("⭐️ ALL-STAR!")
        else:
            status_col.error("❌ NOT AN ALL-STAR")

        prob_col.metric(
            "Model Score",
            f"{prob * 100:.1f}%"
        )

st.info(
    "Important note: the output is a model score, not a literal probability. " \
    "Usually, if the model score is above 50% that means they're an all star, " \
    "but that would only result in 15 all stars. I lowered the thresehold to " \
    "34.34% so there's exactly 24 all stars. "
)
