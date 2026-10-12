"""Streamlit front end for the regime P&L forecasting tool.

Run with:  streamlit run app.py
"""

import streamlit as st

import forecast as fc

st.set_page_config(page_title="Regime P&L Forecast", layout="wide")
st.title("Regime-conditioned P&L forecast and risk")
st.caption("Runs on synthetic data: no real trading data is included. Units are arbitrary.")


@st.cache_resource
def load():
    df = fc.load_data()
    return df, fc.fit_models(df)


df, models = load()
instruments = sorted(df["Instrument"].unique())
regimes = sorted(df["Regime"].unique())

tab_fc, tab_risk, tab_models = st.tabs(["Forecast", "Regime VaR / ES", "Model fit"])

with tab_fc:
    c1, c2, c3, c4, c5 = st.columns(5)
    inst = c1.selectbox("Instrument", instruments)
    seg = c2.selectbox("Client segment", ["Total", "A", "B"])
    start = c3.selectbox("Starting regime", regimes)
    days = c4.slider("Days ahead", 1, 30, 5)
    sims = c5.slider("Simulations", 500, 10000, 5000, step=500)
    if st.button("Run forecast", type="primary"):
        segments = fc.SEGMENTS if seg == "Total" else (seg,)
        paths = fc.simulate_paths(df, models, inst, days, sims, starting_regime=start, segments=segments)
        s = fc.summarise(paths[-1])
        m1, m2, m3 = st.columns(3)
        m1.metric("Median", f"{s['median']:,.0f}")
        m2.metric("Interquartile range", f"{s['q1']:,.0f} to {s['q3']:,.0f}")
        m3.metric("5th percentile", f"{s['p5']:,.0f}")
        st.pyplot(fc.plot_distribution(paths[-1], f"{inst} ({seg}): {days}-day P&L starting in {start}"))
        st.pyplot(fc.plot_paths(paths, f"Simulated cumulative P&L paths, {inst} ({seg})"))

with tab_risk:
    c1, c2, c3, c4, c5 = st.columns(5)
    r_inst = c1.selectbox("Instrument ", instruments, key="r_inst")
    r_seg = c2.selectbox("Client segment ", ["Total", "A", "B"], key="r_seg")
    r_regime = c3.selectbox("Regime at window start", regimes, key="r_regime")
    r_days = c4.slider("Horizon (days)", 1, 30, 5, key="r_days")
    r_conf = c5.slider("Confidence (%)", 90, 99, 95, key="r_conf")
    if st.button("Calculate risk", type="primary"):
        res = fc.regime_var(df, r_inst, r_regime, r_days, r_conf / 100, segment=r_seg)
        if res is None:
            st.error("Fewer than 20 historical windows start in this regime. Try a shorter horizon.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric(f"VaR ({r_conf}%)", f"{res['var']:,.0f}")
            m2.metric("Expected shortfall", f"{res['es']:,.0f}")
            m3.metric("Windows used", res["n_windows"])
            st.write(
                f"Across {res['n_windows']} historical {r_days}-day windows that began in **{r_regime}**, "
                f"the worst {100 - r_conf}% ended below **{res['var']:,.0f}**, "
                f"and averaged **{res['es']:,.0f}** when they did."
            )
            st.pyplot(fc.plot_var(res, f"{r_inst} ({r_seg}): {r_days}-day outcomes from {r_regime} starts"))

with tab_models:
    st.write("In-sample R² of each regime regression. Low values mean most of the simulated "
             "dispersion comes from the resampled residuals rather than the fitted drivers.")
    st.dataframe(fc.model_summary(models), hide_index=True, width="stretch")
