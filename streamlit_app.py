"""Streamlit research UI for TradingAgents. No live brokerage orders."""
import os
from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ZM AI Trading Research", page_icon="📈", layout="wide")
st.title("📈 ZM AI Trading Research")
st.caption("Research and paper analysis only — no real-money orders or guaranteed signals.")

with st.sidebar:
    st.header("Analysis settings")
    ticker = st.text_input("Yahoo Finance ticker", "AAPL").strip().upper()
    analysis_date = st.date_input("Analysis date", value=date.today(), max_value=date.today())
    provider = st.selectbox("AI provider", ["google", "groq", "openai"])
    quick_model = st.text_input("Quick model ID", "gemini-2.5-flash" if provider == "google" else "")
    deep_model = st.text_input("Deep model ID", "gemini-2.5-flash" if provider == "google" else "")
    analysts = st.multiselect("Analysts", ["market", "news", "fundamentals", "social"], default=["market"])
    st.caption("Set provider API key in Streamlit app Secrets, not in GitHub.")
    run = st.button("Run AI research", type="primary", disabled=not (ticker and analysts and quick_model and deep_model))

if not ticker or len(ticker) > 24 or not all(c.isalnum() or c in ".-^=" for c in ticker):
    st.error("Enter a valid Yahoo Finance ticker.")
    st.stop()

@st.cache_data(ttl=900, show_spinner=False)
def prices(symbol):
    return yf.download(symbol, period="3mo", interval="1d", auto_adjust=True, progress=False)

try:
    data = prices(ticker)
    if data.empty:
        st.warning("No price history returned for this ticker.")
    else:
        closes = data["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        st.metric("Latest available close", f"${float(closes.iloc[-1]):,.2f}")
        st.line_chart(closes)
        st.caption("Price data may be delayed. This is not a live trading feed.")
except Exception as exc:
    st.warning(f"Market data unavailable: {type(exc).__name__}")

if run:
    key_names = {"google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}
    key_name = key_names[provider]
    key = st.secrets.get(key_name, os.getenv(key_name, ""))
    if not key:
        st.error(f"Missing {key_name}. Add it under Streamlit → App settings → Secrets.")
        st.stop()
    os.environ[key_name] = key
    try:
        from tradingagents.default_config import build_default_config
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = build_default_config()
        config.update({
            "llm_provider": provider,
            "quick_think_llm": quick_model,
            "deep_think_llm": deep_model,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_tool_rounds": 5,
            "checkpoint_enabled": False,
        })
        with st.spinner("Agents analyzing market data; this may take several minutes..."):
            graph = TradingAgentsGraph(selected_analysts=tuple(analysts), config=config)
            asset_type = "crypto" if ticker.endswith("-USD") else "stock"
            state, decision = graph.propagate(ticker, analysis_date.isoformat(), asset_type=asset_type)
        st.success("Analysis completed")
        st.subheader("AI decision")
        st.write(str(decision))
        for key, label in [
            ("market_report", "Technical analysis"),
            ("news_report", "News"),
            ("fundamentals_report", "Fundamentals"),
            ("sentiment_report", "Sentiment"),
            ("final_trade_decision", "Trade reasoning"),
        ]:
            value = state.get(key) if isinstance(state, dict) else None
            if value:
                with st.expander(label):
                    st.write(str(value))
    except Exception as exc:
        st.error(f"AI analysis failed: {type(exc).__name__}: {exc}")
        st.info("Check API key, model IDs, provider quotas, and Streamlit logs.")
