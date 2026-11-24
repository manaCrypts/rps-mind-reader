"""Rock-Paper-Scissors Mind Reader - Streamlit game.  Run: streamlit run app.py"""
import os, random
import pandas as pd
import streamlit as st
from rps_ai.agent import AI_NAMES, make_agent
from rps_ai.game import MOVES, outcome

ICON = ["✊", "✋", "✌️"]
st.set_page_config(page_title="RPS Mind Reader", page_icon="🧠", layout="centered")
S = st.session_state


def new_game(name):
    S.ai_name = name
    S.agent = make_agent(name, seed=random.randrange(10**6))
    S.hist = []
    S.committed = S.agent.next_move()          # AI locks in its move BEFORE the player picks
    S.committed_probs = S.agent.last_probs.copy()


def play(m):
    ai, probs = S.committed, S.committed_probs
    res = outcome(ai, m)                        # from the AI's point of view
    S.hist.append(dict(round=len(S.hist) + 1, you=m, ai=ai, ai_result=res,
                       ai_expected=int(probs.argmax()), p_expected=float(probs.max())))
    S.agent.observe(ai, m)                      # AI learns from your move
    S.committed = S.agent.next_move()
    S.committed_probs = S.agent.last_probs.copy()


with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("AI brain", AI_NAMES, index=AI_NAMES.index("Markov2"),
                          help="Random = baseline (33%). Markov2 / HMM learn your habits.")
    win = st.slider("Rolling window", 5, 50, 20)
    if st.button("🔄 New game"):
        new_game(choice)
    if st.button("💾 Save session to data/"):
        if S.get("hist"):
            os.makedirs("data", exist_ok=True)
            df = pd.DataFrame(S.hist).assign(predictor=S.ai_name)
            f = "data/human_sessions.csv"
            df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
            st.success(f"Saved {len(df)} rounds")
if "agent" not in S or S.ai_name != choice:
    new_game(choice)

st.title("🧠 Rock-Paper-Scissors Mind Reader")
st.caption("The AI has already locked in its move 🔒 - it studies your habits to predict you. "
           "Pure luck gives it 33.3%. Can you stay unpredictable?")

c = st.columns(3)
for i in range(3):
    c[i].button(f"{ICON[i]} {MOVES[i]}", key=f"b{i}", on_click=play, args=(i,), use_container_width=True)

h = S.hist
if h:
    last = h[-1]
    verdict = {1: "🤖 AI wins", 0: "🤝 Tie", -1: "🎉 You win"}[last["ai_result"]]
    st.subheader(f"{verdict}!   You {ICON[last['you']]}  vs  AI {ICON[last['ai']]}")
    st.caption(f"The AI expected you to play **{MOVES[last['ai_expected']]}** "
               f"({last['p_expected']:.0%} confidence).")
else:
    st.info("Pick a move to start. The AI has hidden its choice. 🔒")

ai_w = sum(r["ai_result"] == 1 for r in h); you_w = sum(r["ai_result"] == -1 for r in h)
m = st.columns(4)
m[0].metric("Rounds", len(h)); m[1].metric("You", you_w); m[2].metric("AI", ai_w)
m[3].metric("AI win-rate", f"{ai_w / len(h):.0%}" if h else "-")

if h:
    w = pd.Series([r["ai_result"] == 1 for r in h], dtype=float)
    st.line_chart(pd.DataFrame({f"AI rolling win-rate ({win})": w.rolling(win, min_periods=1).mean(),
                                "AI cumulative win-rate": w.expanding().mean(),
                                "Luck (33.3%)": 1 / 3}), height=260)
    with st.expander("Round history"):
        st.dataframe(pd.DataFrame([dict(round=r["round"], you=MOVES[r["you"]], ai=MOVES[r["ai"]],
                                        result={1: "AI", 0: "Tie", -1: "You"}[r["ai_result"]]) for r in h]),
                     hide_index=True)
