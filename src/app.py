import streamlit as st

from main import optimize_portfolio, rank_etfs

st.set_page_config(page_title="ETF Return/Risk Ranker", layout="wide")
st.title("ETF Return/Risk Ranker + Optimizer")

etf_text = st.text_input("ETF list (space-separated)", value="SPY QQQ IWM XLE XLK VTI")
col1, col2, col3, col4 = st.columns(4)
period = col1.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
top_components = col2.slider("Top components", 5, 25, 10)
best_n = col3.slider("Top N ETFs", 3, 10, 5)
score_mode = col4.selectbox("Score mode", ["sharpe", "return_risk"], index=0)

col5, col6, col7 = st.columns(3)
risk_free = col5.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.02, step=0.005)
sector_penalty = col6.number_input("Sector penalty", min_value=0.0, max_value=2.0, value=0.20, step=0.05)
max_weight = col7.number_input("Optimizer max ETF weight", min_value=0.05, max_value=1.0, value=0.50, step=0.05)

run_opt = st.checkbox("Also run portfolio optimizer", value=True)

if st.button("Run"):
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
        st.subheader("ETF Ranking")
        show = df.copy()
        for c in ["Annual Return", "Annual Volatility"]:
            show[c] = (show[c] * 100).round(2).astype(str) + "%"
        show["Sharpe-like"] = show["Sharpe-like"].round(3)
        show["Sector HHI"] = show["Sector HHI"].round(3)
        show["Score"] = show["Score"].round(3)
        st.dataframe(show, use_container_width=True)
        st.download_button("Download ranking CSV", data=df.to_csv(index=False), file_name="etf_ranking.csv", mime="text/csv")

        if run_opt:
            with st.spinner("Optimizing ETF weights..."):
                out = optimize_portfolio(df, period, risk_free, sector_penalty, max_weight, 5000)
            if isinstance(out, tuple):
                wdf, sdf = out
                st.subheader("Optimized Weights")
                wshow = wdf.copy()
                wshow["Weight"] = (wshow["Weight"] * 100).round(2).astype(str) + "%"
                wshow["ETF Return"] = (wshow["ETF Return"] * 100).round(2).astype(str) + "%"
                wshow["ETF Sector HHI"] = wshow["ETF Sector HHI"].round(3)
                st.dataframe(wshow, use_container_width=True)

                st.subheader("Optimized Portfolio Summary")
                sshow = sdf.copy()
                sshow["Portfolio Annual Return"] = (sshow["Portfolio Annual Return"] * 100).round(2).astype(str) + "%"
                sshow["Portfolio Annual Volatility"] = (sshow["Portfolio Annual Volatility"] * 100).round(2).astype(str) + "%"
                sshow["Portfolio Sharpe-like"] = sshow["Portfolio Sharpe-like"].round(3)
                sshow["Objective (Sharpe-Penalty)"] = sshow["Objective (Sharpe-Penalty)"].round(3)
                st.dataframe(sshow, use_container_width=True)
