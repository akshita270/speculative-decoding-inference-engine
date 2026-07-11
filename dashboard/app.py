"""Streamlit dashboard for the speculative decoding inference engine.

Reads observability/metrics_store.csv (written by every /generate request)
and renders latency/throughput trends, headline stats, and a table of recent
requests. Auto-refreshes so it's useful to watch while load_test.py runs.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "observability", "metrics_store.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "load_test", "results.json")

PURPLE = "#6A0DAD"
PURPLE_DARK = "#3C096C"
PURPLE_LIGHT = "#B185DB"
GREEN = "#2ecc71"
ORANGE = "#f39c12"
RED = "#e74c3c"
GREY = "#c9c2d6"
BLUE = "#2a78d6"
ORANGE_ACCENT = "#eb6834"
MAGENTA = "#e87ba4"
AQUA_LIGHT = "#b7ecd7"
AQUA_DARK = "#0f7a54"

# fixed color-by-identity, not by rank, so a category keeps its color
# regardless of which one happens to be most frequent right now
# (blue/orange pair validated colorblind-safe via the dataviz skill's
# validate_palette.js; grey is a deliberately muted "not really a draft
# model" bucket, not a peer category, so it's exempt from that check)
DRAFT_MODEL_COLORS = {"gpt2": BLUE, "gpt2-medium": ORANGE_ACCENT, "cache / fallback": GREY}

st.set_page_config(page_title="Speculative Decoding Dashboard", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #F4F2F9; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PURPLE} 0%, {PURPLE_DARK} 100%);
        border-top-right-radius: 40px;
        border-bottom-right-radius: 40px;
    }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] .stButton button {{
        background: rgba(255,255,255,0.12);
        border: none;
        border-radius: 14px;
        width: 100%;
        padding: 0.6rem;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background: rgba(255,255,255,0.28);
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 18px rgba(80, 20, 130, 0.08);
    }}
    /* the rule above targets bordered st.container()s app-wide, which also
    matches Streamlit's internal block wrappers inside the sidebar -- undo
    it there so the sidebar keeps its purple gradient instead of a leaked
    white box (which was also making the white nav-button text invisible). */
    [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        background: transparent !important;
        box-shadow: none !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    .stTextInput input {{
        border-radius: 12px;
    }}
    .sd-avatar {{
        width: 64px; height: 64px; border-radius: 50%;
        background: rgba(255,255,255,0.15);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; font-weight: 800; color: white;
        margin: 0.5rem auto 1.2rem auto;
    }}
    .sd-bignum {{ font-size: 2.1rem; font-weight: 800; color: {PURPLE_DARK}; }}
    .sd-caption {{ color: #8a7ba0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
    .sd-badge-ok {{ background: #eafaf1; color: #1e8449; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }}
    .sd-badge-warn {{ background: #fdecea; color: #c0392b; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "sd_page" not in st.session_state:
    st.session_state.sd_page = "Overview"

with st.sidebar:
    st.markdown('<div class="sd-avatar">SD</div>', unsafe_allow_html=True)
    if st.button("\U0001F3E0  Overview", key="nav_overview"):
        st.session_state.sd_page = "Overview"
    if st.button("\U0001F4CB  Request Log", key="nav_log"):
        st.session_state.sd_page = "Request Log"
    if st.button("\U0001F512  Limitations", key="nav_about"):
        st.session_state.sd_page = "Limitations"
    st.markdown("---")
    refresh_seconds = st.slider("Auto-refresh (s)", 2, 30, 5)
    st.caption(f"Source: {os.path.basename(CSV_PATH)}")

try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=refresh_seconds * 1000, key="refresh")
except ImportError:
    if st.sidebar.button("Refresh now"):
        st.rerun()

st.markdown(
    f"<h2 style='text-align:center; margin-bottom:0.3rem; color:{PURPLE_DARK};'>"
    "Speculative Decoding Inference Engine</h2>",
    unsafe_allow_html=True,
)
_, search_col, _ = st.columns([1, 2, 1])
with search_col:
    search = st.text_input(
        "Search prompts", placeholder="\U0001F50D  Search prompts...", label_visibility="collapsed"
    )

if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
    st.info("No requests logged yet. Start the API and send some /generate requests, or run load_test.py.")
    st.stop()

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

for col in ("cache_hit", "fallback_triggered"):
    if df[col].dtype == object:
        df[col] = df[col].map({"True": True, "False": False}).fillna(df[col])

if search:
    df_view = df[df["prompt"].str.contains(search, case=False, na=False)]
else:
    df_view = df

total = len(df_view)
cache_hit_rate = df_view["cache_hit"].mean() * 100 if total else 0.0
fallback_count = int(df_view["fallback_triggered"].sum()) if total else 0
non_cached = df_view[~df_view["cache_hit"]]
avg_acceptance = non_cached["acceptance_rate"].dropna().mean() if len(non_cached) else float("nan")
avg_tps = df_view["tokens_per_sec"].mean() if total else 0.0


def split_on_gaps(x, y, gap_minutes: float = 2.0):
    """Split a time series into separate contiguous segments wherever two
    points are more than `gap_minutes` apart. Plotly's `connectgaps=False`
    breaks the *line* at a None, but its filled area under the line isn't
    reliably broken the same way -- it can still draw a smooth fill spanning
    a stretch of time where the API simply wasn't used. Returning fully
    separate segments (one trace per segment) guarantees each fill only ever
    covers real, contiguous data -- a gap in activity looks like a gap, not
    a slope.
    """
    x = list(x)
    y = list(y)
    if not x:
        return []
    segments = [([x[0]], [y[0]])]
    for i in range(1, len(x)):
        if (x[i] - x[i - 1]).total_seconds() > gap_minutes * 60:
            segments.append(([], []))
        segments[-1][0].append(x[i])
        segments[-1][1].append(y[i])
    return segments


def gauge(value: float, color: str, title: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value, 1),
            number={"suffix": "%", "font": {"size": 26, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "visible": False},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#eee6f7",
                "borderwidth": 0,
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=10, r=10, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=PURPLE_DARK),
    )
    return fig


if st.session_state.sd_page == "Overview":
    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown('<div class="sd-caption">Statistics</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sd-bignum">{avg_tps:.2f} tok/s</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for seg_x, seg_y in split_on_gaps(df_view["timestamp"], df_view["tokens_per_sec"]):
                fig.add_trace(
                    go.Scatter(
                        x=seg_x,
                        y=seg_y,
                        mode="lines+markers",
                        line=dict(color=PURPLE, shape="linear", width=3),
                        marker=dict(size=5, color=PURPLE),
                        fill="tozeroy",
                        fillcolor="rgba(106,13,173,0.12)",
                        showlegend=False,
                    )
                )
            fig.update_layout(
                height=260,
                margin=dict(l=10, r=15, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=False, title=None, automargin=True),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#eee6f7",
                    title=dict(text="tokens/sec", standoff=10),
                    automargin=True,
                ),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with st.container(border=True):
            st.markdown('<div class="sd-caption">Latency over time</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            for seg_x, seg_y in split_on_gaps(df_view["timestamp"], df_view["latency_s"]):
                fig2.add_trace(
                    go.Scatter(
                        x=seg_x,
                        y=seg_y,
                        mode="lines+markers",
                        line=dict(color=PURPLE_DARK, shape="linear", width=3),
                        marker=dict(size=5, color=PURPLE_DARK),
                        fill="tozeroy",
                        fillcolor="rgba(60,9,108,0.10)",
                        showlegend=False,
                    )
                )
            fig2.update_layout(
                height=230,
                margin=dict(l=10, r=15, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=False, automargin=True),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#eee6f7",
                    title=dict(text="seconds", standoff=10),
                    automargin=True,
                ),
            )
            st.plotly_chart(fig2, use_container_width=True, theme=None)

    with right:
        with st.container(border=True):
            st.markdown('<div class="sd-caption">Reliability</div>', unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    gauge(cache_hit_rate, GREEN, "Cache hit"),
                    use_container_width=True,
                    key="gauge_cache",
                    theme=None,
                )
                st.markdown(
                    "<div style='text-align:center; color:#8a7ba0; font-weight:600;'>Cache hit rate</div>",
                    unsafe_allow_html=True,
                )
            with g2:
                acc_pct = avg_acceptance * 100 if pd.notna(avg_acceptance) else 0.0
                st.plotly_chart(
                    gauge(acc_pct, ORANGE, "Acceptance"),
                    use_container_width=True,
                    key="gauge_accept",
                    theme=None,
                )
                st.markdown(
                    "<div style='text-align:center; color:#8a7ba0; font-weight:600;'>Draft acceptance</div>",
                    unsafe_allow_html=True,
                )

        r1, r2 = st.columns(2)
        with r1:
            with st.container(border=True):
                st.markdown('<div class="sd-caption">Total requests</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sd-bignum">▲ {total}</div>', unsafe_allow_html=True)
                st.markdown('<span class="sd-badge-ok">Logged &amp; served</span>', unsafe_allow_html=True)
        with r2:
            with st.container(border=True):
                st.markdown('<div class="sd-caption">Fallback count</div>', unsafe_allow_html=True)
                arrow = "▲" if fallback_count > 0 else "▼"
                st.markdown(f'<div class="sd-bignum">{arrow} {fallback_count}</div>', unsafe_allow_html=True)
                badge_cls = "sd-badge-warn" if fallback_count > 0 else "sd-badge-ok"
                badge_text = "Draft model failed over" if fallback_count > 0 else "No failures"
                st.markdown(f'<span class="{badge_cls}">{badge_text}</span>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="sd-caption">Growth &mdash; throughput by concurrency</div>', unsafe_allow_html=True)
            if os.path.exists(RESULTS_PATH):
                with open(RESULTS_PATH) as f:
                    load_results = json.load(f)
                x = [str(r["concurrency"]) for r in load_results]
                y = [r["throughput_req_per_s"] for r in load_results]
            elif df_view["draft_model"].notna().any():
                counts = df_view["draft_model"].fillna("cache / fallback").value_counts()
                x, y = list(counts.index), list(counts.values)
            else:
                x, y = [], []

            if x:
                fig3 = go.Figure(
                    go.Bar(
                        x=x,
                        y=y,
                        marker=dict(
                            color=y,
                            colorscale=[[0, PURPLE_LIGHT], [1, PURPLE_DARK]],
                        ),
                    )
                )
                fig3.update_layout(
                    height=250,
                    margin=dict(l=10, r=15, t=10, b=10),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(
                        title=dict(
                            text="concurrency" if os.path.exists(RESULTS_PATH) else "draft model",
                            standoff=10,
                        ),
                        automargin=True,
                    ),
                    yaxis=dict(
                        title=dict(
                            text="req/s" if os.path.exists(RESULTS_PATH) else "count",
                            standoff=10,
                        ),
                        gridcolor="#eee6f7",
                        automargin=True,
                    ),
                )
                st.plotly_chart(fig3, use_container_width=True, theme=None)
            else:
                st.caption("Run load_test.py to populate this chart.")

    with st.container(border=True):
        st.markdown('<div class="sd-caption">More insights</div>', unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)

        with i1:
            st.markdown("**Draft model usage**")
            counts = df_view["draft_model"].fillna("cache / fallback").value_counts()
            fig4 = go.Figure(
                go.Bar(
                    x=list(counts.index),
                    y=list(counts.values),
                    marker=dict(color=[DRAFT_MODEL_COLORS.get(k, GREY) for k in counts.index]),
                )
            )
            fig4.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(automargin=True),
                yaxis=dict(
                    title=dict(text="requests", standoff=10), gridcolor="#eee6f7", automargin=True
                ),
            )
            st.plotly_chart(fig4, use_container_width=True, theme=None)
            st.caption("Which draft model the router picked per request -- fast (gpt2) under load, better (gpt2-medium) when quiet.")

        with i2:
            st.markdown("**Batch size distribution**")
            batch_counts = df_view["batch_size"].value_counts().sort_index()
            fig5 = go.Figure(
                go.Bar(
                    x=[str(v) for v in batch_counts.index],
                    y=list(batch_counts.values),
                    marker=dict(
                        color=list(batch_counts.index),
                        colorscale=[[0, AQUA_LIGHT], [1, AQUA_DARK]],
                    ),
                )
            )
            fig5.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(title=dict(text="batch size", standoff=10), automargin=True),
                yaxis=dict(
                    title=dict(text="requests", standoff=10), gridcolor="#eee6f7", automargin=True
                ),
            )
            st.plotly_chart(fig5, use_container_width=True, theme=None)
            st.caption("How many requests the batcher grouped together before running them through the engine.")

        with i3:
            st.markdown("**Latency distribution**")
            fig6 = go.Figure(
                go.Histogram(x=df_view["latency_s"], marker=dict(color=MAGENTA), nbinsx=20)
            )
            if len(df_view) > 0:
                p50 = df_view["latency_s"].median()
                p95 = df_view["latency_s"].quantile(0.95)
                fig6.add_vline(x=p50, line=dict(color=GREEN, dash="dash"), annotation_text="p50")
                fig6.add_vline(x=p95, line=dict(color=ORANGE, dash="dash"), annotation_text="p95")
            fig6.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(title=dict(text="seconds", standoff=10), automargin=True),
                yaxis=dict(title=dict(text="count", standoff=10), gridcolor="#eee6f7", automargin=True),
            )
            st.plotly_chart(fig6, use_container_width=True, theme=None)
            st.caption("Spread of request latencies -- p50 (typical) vs p95 (near-worst-case).")

    with st.container(border=True):
        st.markdown('<div class="sd-caption">Last 20 requests</div>', unsafe_allow_html=True)
        recent = df_view.tail(20).sort_values("timestamp", ascending=False)
        st.dataframe(
            recent[
                [
                    "timestamp",
                    "prompt",
                    "latency_s",
                    "tokens_per_sec",
                    "acceptance_rate",
                    "cache_hit",
                    "draft_model",
                    "fallback_triggered",
                    "fallback_reason",
                    "batch_size",
                ]
            ],
            use_container_width=True,
        )

elif st.session_state.sd_page == "Request Log":
    with st.container(border=True):
        st.markdown('<div class="sd-caption">Full request log</div>', unsafe_allow_html=True)
        st.dataframe(df_view.sort_values("timestamp", ascending=False), use_container_width=True, height=650)

elif st.session_state.sd_page == "Limitations":
    with st.container(border=True):
        st.markdown('<div class="sd-caption">Honest limitations</div>', unsafe_allow_html=True)
        st.markdown(
            """
- Small models (GPT-2 family) — modest, real speedups, not paper-scale ones.
- No KV-cache reuse across speculative rounds (biggest lever left on the table).
- Batching groups arrivals but still processes requests sequentially through the engine.
- Single in-process model worker — no multi-GPU/multi-replica parallelism.
- Load is simulated (`load_test.py`), not real multi-user traffic.
- Semantic cache is in-memory only, capped at 500 entries, nothing persists across restarts.

See [README.md](https://github.com/akshita270/speculative-decoding-inference-engine#readme) for full details and measured numbers.
            """
        )
