import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Solar Forecast Replay", layout="wide")

@st.cache_data
def load():
    d = pd.read_csv("dashboard_data.csv", parse_dates=["time"])
    return d

df = load()
n = len(df)

st.title("Solar Power Forecasting: 1-Hour-Ahead Replay")
st.caption("Replay of the held-out test period from DKASC site 31 (Alice Springs). "
           "This is a replay of historical data, not live plant telemetry.")

c1, c2, c3 = st.columns([2, 1, 1])
window_h = c2.selectbox("Window shown", [24, 48, 72, 168], index=1, format_func=lambda h: f"{h} hours")
speed = c3.selectbox("Replay speed", ["Slow", "Normal", "Fast"], index=1)
delay = {"Slow": 0.8, "Normal": 0.3, "Fast": 0.05}[speed]

if "pos" not in st.session_state:
    st.session_state.pos = window_h
if "playing" not in st.session_state:
    st.session_state.playing = False

pos = c1.slider("Replay position (hours into the test period)", window_h, n,
                min(st.session_state.pos, n), key="slider_pos")
st.session_state.pos = pos

b1, b2, _ = st.columns([1, 1, 6])
if b1.button("▶ Play"):
    st.session_state.playing = True
if b2.button("⏸ Pause"):
    st.session_state.playing = False

placeholder = st.empty()

def render(pos):
    seen = df.iloc[:pos]
    win = df.iloc[max(0, pos - window_h):pos]
    day = seen[seen.actual_kw > 0.05 * df.actual_kw.max()]
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    r_m, r_p = rmse(seen.actual_kw, seen.forecast_kw), rmse(seen.actual_kw, seen.persist_kw)
    inside = ((day.actual_kw >= day.lo_kw) & (day.actual_kw <= day.hi_kw)).mean() * 100 if len(day) else float("nan")
    last = seen.iloc[-1]

    with placeholder.container():
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Now (forecast for)", last.time.strftime("%d %b %Y %H:%M"))
        k2.metric("Forecast", f"{last.forecast_kw:.2f} kW")
        k3.metric("Running RMSE (model)", f"{r_m:.3f} kW")
        k4.metric("Running RMSE (persistence)", f"{r_p:.3f} kW",
                  delta=f"{(1 - r_m / r_p) * 100:.0f}% better with model" if r_p > 0 else None)
        k5.metric("90% band coverage (daytime)", f"{inside:.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=win.time, y=win.hi_kw, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=win.time, y=win.lo_kw, fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(255,152,0,0.25)", name="90% interval"))
        fig.add_trace(go.Scatter(x=win.time, y=win.actual_kw, name="Actual", line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=win.time, y=win.forecast_kw, name="LightGBM forecast",
                                 line=dict(color="#F44336", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=win.time, y=win.persist_kw, name="Persistence",
                                 line=dict(color="#90A4AE", width=1.2, dash="dot")))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="AC power (kW)",
                          legend=dict(orientation="h", y=1.08), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

if st.session_state.playing:
    p = st.session_state.pos
    while st.session_state.playing and p < n:
        render(p)
        time.sleep(delay)
        p += 1
        st.session_state.pos = p
    st.session_state.playing = False
render(st.session_state.pos)

st.markdown("---")
st.caption("Model: LightGBM, 1-hour-ahead, features known at issue time only. "
           "Single site, single desert climate. Intervals are conformally calibrated on a validation set.")