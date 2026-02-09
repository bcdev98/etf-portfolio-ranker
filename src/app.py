import pandas as pd
import streamlit as st

from main import rank_etfs

st.set_page_config(page_title="ETF Return/Risk Ranker", layout="wide")
st.title("ETF Return/Risk Ranker")

etf_text = st.text_input("ETF list (space-separated)", value="SPY QQQ IWM XLE XLK VTI")
col1, col2, col3, col4 = st.columns(4)
period = col1.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
top_components = col2.slider("Top components", 5, 25, 10)
best_n = col3.slider("Top N ETFs", 3, 10, 5)
score_mode = col4.selectbox("Score mode", ["sharpe", "return_risk"], index=0)

col5, col6 = st.columns(2)
risk_free = col5.number_input("Risk-free rate (annual, decimal)", min_value=0.0, max_value=0.20, value=0.02, step=0.005)
sector_penalty = col6.number_input("Sector concentration penalty", min_value=0.0, max_value=2.0, value=0.20, step=0.05)

if st.button("Run ranking"):
    etfs = [x.strip().upper() for x in etf_text.split() if x.strip()]
    with st.spinner("Fetching holdings and prices..."):
        df = rank_etfs(
            etfs=etfs,
            period=period,
            top_n=top_components,
            best_n=best_n,
            score_mode=score_mode,
            risk_free=risk_free,
            sector_penalty=sector_penalty,
        )

    if df.empty:
        st.error("No ETF scores produced. Check symbols/data availability.")
    else:
        show = df.copy()
        for c in ["Annual Return", "Annual Volatility"]:
            show[c] = (show[c] * 100).round(2).astype(str) + "%"
        show["Sharpe-like"] = show["Sharpe-like"].round(3)
        show["Sector HHI"] = show["Sector HHI"].round(3)
        show["Score"] = show["Score"].round(3)
        st.dataframe(show, use_container_width=True)
        st.download_button("Download CSV", data=df.to_csv(index=False), file_name="etf_ranking.csv", mime="text/csv")
