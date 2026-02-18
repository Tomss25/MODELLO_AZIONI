import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict
import io
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
import mstarpy
import seaborn as sns
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta

# =========================
# 1. CONFIGURAZIONE & STILE
# =========================
st.set_page_config(
    page_title="Wealth Model Institutional Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    h1, h2, h3 { color: #0F172A; font-weight: 700; letter-spacing: -0.5px; }
    
    /* KPI Cards */
    div[data-testid="stMetric"] { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    div[data-testid="stMetricLabel"] { font-size: 12px; color: #64748B; text-transform: uppercase; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; font-size: 24px; color: #0F172A; }
    
    /* Tabelle */
    div[data-testid="stDataFrame"] { font-family: 'Roboto Mono', monospace; font-size: 12px; }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #0F172A; }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    div[data-testid="stRadio"] label p { color: #F8FAFC !important; }

    /* Bottoni */
    div.stButton > button { background-color: #FF6600; color: white; border: none; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    div.stButton > button:hover { background-color: #CC5200; border: none; color: white; }
    
    /* Input Area Darker */
    .stTextArea textarea { background-color: #F1F5F9; color: #0F172A; border: 1px solid #CBD5E1; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. CONFIGURAZIONE PARAMETRI
# =========================
CONFIG: Dict = {
    "MC_SIMULATIONS": 5000,
    "SEED": 42,
    "MIN_WACC": 0.045,
    "MAX_WACC": 0.25,
    "MIN_TV_SPREAD": 0.015,
    "CREDIT_SPREAD": 0.02,
    "MAX_DEBT_WEIGHT": 0.60,
    "GDP_MIN_OFFSET": -0.05, 
    "GDP_MAX_OFFSET": 0.04,
    "WACC_VOL": 0.15,
    "GROWTH_VOL": 0.25,
    "MARGIN_VOL": 0.10,
    "CORRELATION_MATRIX": np.array([
        [ 1.0, -0.4, -0.3],
        [-0.4,  1.0,  0.6],
        [-0.3,  0.6,  1.0]
    ]),
}

# =========================
# 3. LOGICA DEL MODELLO (VALUTAZIONE)
# =========================
def _to_float(x, default=0.0) -> float:
    if pd.isna(x) or x is None: return float(default)
    s = str(x).replace(",", ".")
    try: return float(s)
    except ValueError: return float(default)

def _safe_str(x) -> str:
    if x is None: return ""
    return str(x).strip()

class WealthModelInstitutional:
    def __init__(self):
        self.rf = 0.04
        self.erp = 0.05
        self.gdp = 0.02
        self.fund_data = pd.DataFrame() 
        self.price_data = pd.DataFrame() 
        self.returns = pd.DataFrame()
        self.betas = {}
        self.sim_storage = {} 
        np.random.seed(CONFIG["SEED"])

    def fetch_from_yahoo(self, tickers_input: str) -> str:
        """Scarica dati fondamentali per la valutazione (Yahoo Only)"""
        status_msg = ""
        tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
        
        if not tickers_list: raise ValueError("Inserisci almeno un Ticker valido.")

        try:
            raw_df = yf.download(tickers_list, period="2y", interval="1wk", progress=False, auto_adjust=False)
            if raw_df.empty: raise ValueError("Yahoo non ha restituito dati.")

            data = pd.DataFrame()
            if isinstance(raw_df.columns, pd.MultiIndex):
                try: data = raw_df.xs('Adj Close', axis=1, level=0, drop_level=True)
                except KeyError:
                    try: data = raw_df.xs('Close', axis=1, level=0, drop_level=True)
                    except KeyError: data = raw_df.iloc[:, :len(tickers_list)]
            else:
                tgt_col = None
                for candidate in ['Adj Close', 'Close', 'AdjClose']:
                    if candidate in raw_df.columns:
                        tgt_col = candidate
                        break
                if tgt_col: data = raw_df[[tgt_col]].copy()
                else: data = raw_df.iloc[:, -1:].copy()
                data.columns = [tickers_list[0]]

            data.index = pd.to_datetime(data.index).tz_localize(None)
            self.price_data = data.ffill().dropna(how="all")
            self.returns = np.log(self.price_data / self.price_data.shift(1)).dropna(how="all")
            status_msg += f"✅ Market Data: {len(tickers_list)} tickers recuperati.\n"
        except Exception as e: raise ValueError(f"Errore download prezzi: {e}")

        fund_list = []
        progress_bar = st.progress(0)
        for i, t in enumerate(tickers_list):
            ebit = 0.0; sales = 0.0; tax_rate = 0.25; net_debt = 0.0; shares = 0.0; curr_price = 0.0; roic = 0.12; payout = 0.0
            try:
                stock = yf.Ticker(t)
                if t in self.price_data.columns: curr_price = self.price_data[t].iloc[-1]
                try:
                    info = stock.info
                    if info and len(info) > 5:
                        ebit = info.get('ebit') or info.get('ebitda') or (info.get('totalRevenue', 0) * 0.15) or 0
                        sales = info.get('totalRevenue', 0)
                        tr = info.get('taxRate'); 
                        if tr: tax_rate = tr
                        d = info.get('totalDebt') or 0; c = info.get('totalCash') or 0; net_debt = d - c
                        lp = info.get('currentPrice') or info.get('regularMarketPreviousClose')
                        if lp: curr_price = lp
                        shares = info.get('sharesOutstanding') or 0
                        roic = info.get('returnOnEquity', 0.12) or 0.12
                        payout = info.get('payoutRatio', 0.0) or 0.0
                except: pass 
                if (shares == 0) and (curr_price > 0):
                    try:
                        mkt_cap = stock.info.get('marketCap')
                        if mkt_cap: shares = mkt_cap / curr_price
                        else: shares = 1.0
                    except: shares = 1.0
                fund_list.append({"Ticker": t, "EBIT_TTM": ebit, "Sales": sales, "TaxRate": tax_rate, "ROIC": roic, "PayoutRatio": payout, "NetDebt": net_debt, "SharesOut": shares, "CurrentPrice": curr_price})
            except Exception: pass
            progress_bar.progress((i + 1) / len(tickers_list))
        progress_bar.empty()
        self.fund_data = pd.DataFrame(fund_list)
        if self.fund_data.empty: raise ValueError("Tutti i ticker sono invalidi.")
        status_msg += f"✅ Fundamentals: {len(self.fund_data)} aziende."
        return status_msg

    def calculate_betas(self) -> None:
        if self.returns.empty: return
        bench_series = self.returns.mean(axis=1).dropna()
        bench_var = bench_series.var()
        if bench_var == 0: bench_var = 1.0
        for t in self.returns.columns:
            try:
                common = pd.concat([self.returns[t], bench_series], axis=1).dropna()
                if len(common) < 10: self.betas[t] = 1.0; continue
                cov = common.iloc[:, 0].cov(common.iloc[:, 1])
                beta = cov / bench_var
                self.betas[t] = max(0.4, min(2.5, beta))
            except: self.betas[t] = 1.0

    def run_valuation(self) -> pd.DataFrame:
        results = []
        self.sim_storage = {}
        try: L = np.linalg.cholesky(CONFIG["CORRELATION_MATRIX"])
        except: L = np.eye(3)

        for _, r in self.fund_data.iterrows():
            ticker = _safe_str(r.get("Ticker"))
            try:
                shares = _to_float(r.get("SharesOut")); ebit = _to_float(r.get("EBIT_TTM"))
                nd = _to_float(r.get("NetDebt")); tax = _to_float(r.get("TaxRate"))
                curr_price = _to_float(r.get("CurrentPrice"))
                if shares <= 0 or ebit <= 0: continue
                
                beta = self.betas.get(ticker, 1.0)
                ke = self.rf + beta * self.erp; kd = self.rf + CONFIG["CREDIT_SPREAD"]
                equity_val = shares * curr_price; ev = equity_val + max(0, nd)
                wd = min(max(nd, 0) / ev, CONFIG["MAX_DEBT_WEIGHT"]) if ev > 0 else 0.0
                wacc_base = max(((1 - wd) * ke + wd * kd * (1 - tax)), CONFIG["MIN_WACC"])
                nopat_base = ebit * (1 - tax)

                Z = np.random.normal(0, 1, size=(3, CONFIG["MC_SIMULATIONS"]))
                shocks = L @ Z
                wacc_sim = np.clip(wacc_base * (1 + shocks[0] * CONFIG["WACC_VOL"]), CONFIG["MIN_WACC"], CONFIG["MAX_WACC"])
                gdp_sim = np.clip(self.gdp * (1 + shocks[1] * CONFIG["GROWTH_VOL"]), self.gdp + CONFIG["GDP_MIN_OFFSET"], self.gdp + CONFIG["GDP_MAX_OFFSET"])
                nopat_sim = nopat_base * (1 + shocks[2] * CONFIG["MARGIN_VOL"])

                fcff_sum = np.zeros(CONFIG["MC_SIMULATIONS"]); curr_nopat = nopat_sim
                for t in range(1, 11):
                    g_t = np.maximum(gdp_sim, (wacc_sim + 0.015) * np.exp(-0.2 * t)) 
                    f = curr_nopat * (1 + g_t) * 0.70 
                    fcff_sum += f / ((1 + wacc_sim) ** t)
                    curr_nopat *= (1 + g_t)

                tv = curr_nopat * (1 + gdp_sim) / np.maximum(wacc_sim - gdp_sim, CONFIG["MIN_TV_SPREAD"])
                ev_sim = fcff_sum + (tv / ((1 + wacc_sim) ** 10))
                equity_sim = ev_sim - nd; fv_sim = equity_sim / shares
                self.sim_storage[ticker] = fv_sim
                
                fv_med = np.nanmedian(fv_sim); fv_std = np.nanstd(fv_sim)
                results.append({"Ticker": ticker, "CurrentPrice": curr_price, "FairValue": fv_med, "Upside_Pct": (fv_med / curr_price - 1) if curr_price > 0 else 0, "Volatility_Risk": fv_std / fv_med if fv_med != 0 else 0, "Beta": beta, "WACC": wacc_base})
            except Exception: continue
        return pd.DataFrame(results).sort_values("Upside_Pct", ascending=False)

# =========================
# 4. DATA ENGINE HELPERS (NUOVI)
# =========================
def get_data_yahoo_series(ticker, start_dt):
    try:
        df = yf.download(ticker, start=start_dt, progress=False, auto_adjust=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                try: col = df.xs('Adj Close', axis=1, level=0, drop_level=True)
                except: col = df.iloc[:, 0]
            else:
                col = df['Adj Close'] if 'Adj Close' in df.columns else df.iloc[:, 0]
            
            series = col.squeeze()
            if isinstance(series, pd.DataFrame): series = series.iloc[:, 0] # Fix per multi-column return
            return series.ffill()
    except: return None
    return None

def get_data_morningstar_series(isin, start_dt, end_dt):
    try:
        fund = mstarpy.Funds(term=isin, country="it")
        history = fund.nav(start_date=start_dt, end_date=end_dt, frequency="daily")
        if history:
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            series = df['nav']
            series.index = series.index.normalize().tz_localize(None)
            return series
    except: return None
    return None

# =========================
# 5. PAGES
# =========================

def init_session_state():
    if 'model' not in st.session_state: st.session_state.model = WealthModelInstitutional()
    if 'results' not in st.session_state: st.session_state.results = None
    if 'data_fetched' not in st.session_state: st.session_state.data_fetched = False

def page_overview():
    st.header("⚡ WEALTH MODEL - TERMINAL")
    k1, k2, k3 = st.columns(3)
    k1.metric("Risk Free (Rf)", f"{st.session_state.model.rf:.2%}")
    k2.metric("Equity Risk Prem. (ERP)", f"{st.session_state.model.erp:.2%}")
    k3.metric("L-T Growth (GDP)", f"{st.session_state.model.gdp:.2%}")
    st.divider()
    col_input, col_action = st.columns([3, 1])
    with col_input:
        tickers_input = st.text_area("Inserisci Ticker (es: AAPL MSFT)", placeholder="Lista ticker...", height=68)
    with col_action:
        st.write(""); st.write("") 
        if st.button("🔎 SCARICA DATI", type="primary", use_container_width=True):
            with st.spinner("Connessione ai Data Feed..."):
                try:
                    msg = st.session_state.model.fetch_from_yahoo(tickers_input)
                    st.session_state.data_fetched = True
                    st.success(msg)
                except Exception as e: st.error(f"Errore Fetch: {e}")
    if st.session_state.data_fetched:
        st.subheader("🛠️ REVISIONE DATI FONDAMENTALI")
        st.info("Se vedi zeri, inserisci i dati manualmente.")
        edited_df = st.data_editor(st.session_state.model.fund_data, num_rows="dynamic", use_container_width=True)
        st.session_state.model.fund_data = edited_df
        st.divider()
        if st.button("🚀 LANCIA SIMULAZIONE", type="primary", use_container_width=True):
             with st.spinner(f"Esecuzione..."):
                st.session_state.model.calculate_betas()
                st.session_state.results = st.session_state.model.run_valuation()
                st.rerun()
    if st.session_state.results is not None:
        st.subheader("📋 VALUTAZIONE FINALE")
        df = st.session_state.results
        try: st.dataframe(df.style.format({"CurrentPrice": "{:.2f}", "FairValue": "{:.2f}", "Upside_Pct": "{:.2%}", "Volatility_Risk": "{:.2%}", "Beta": "{:.2f}", "WACC": "{:.1%}"}).background_gradient(subset=["Upside_Pct"], cmap="RdYlGn", vmin=-0.2, vmax=0.5), use_container_width=True, height=500)
        except ImportError: st.dataframe(df, use_container_width=True)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer: df.to_excel(writer, sheet_name="VALUATION", index=False)
        st.download_button("📥 DOWNLOAD REPORT", buffer.getvalue(), "Wealth_Model_Output.xlsx")

def page_deep_dive():
    st.header("🔍 ANALISI SINGOLO TITOLO")
    if st.session_state.results is None: st.warning("⚠️ Esegui prima la simulazione."); return
    df = st.session_state.results
    tickers = df["Ticker"].tolist()
    col_sel, _ = st.columns([1, 3])
    with col_sel: selected_ticker = st.selectbox("Seleziona Titolo", tickers)
    row = df[df["Ticker"] == selected_ticker].iloc[0]
    sim_data = st.session_state.model.sim_storage.get(selected_ticker, [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prezzo Attuale", f"{row['CurrentPrice']:.2f}")
    c2.metric("Fair Value", f"{row['FairValue']:.2f}", delta=f"{row['Upside_Pct']:.1%}")
    c3.metric("Rischio (CV)", f"{row['Volatility_Risk']:.2%}")
    c4.metric("WACC", f"{row['WACC']:.2%}")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim_data, nbinsx=60, name="Distribuzione FV", marker_color='#334155', opacity=0.75))
    fig.add_vline(x=row['CurrentPrice'], line_width=3, line_dash="dash", line_color="#FF6600", annotation_text="Prezzo Oggi")
    fig.add_vline(x=row['FairValue'], line_width=3, line_color="#10B981", annotation_text="Fair Value")
    fig.update_layout(title="Densità di Probabilità Fair Value", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

def page_market_view():
    st.header("🌍 MARKET MAP")
    if st.session_state.results is None: st.warning("⚠️ Esegui prima la simulazione."); return
    df = st.session_state.results
    fig = px.scatter(df, x="Volatility_Risk", y="Upside_Pct", text="Ticker", size="Beta", color="Upside_Pct", color_continuous_scale="RdYlGn", title="Frontiera Efficiente: Rischio vs Upside")
    fig.update_traces(textposition='top center')
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def page_data_engine():
    st.header("📥 DATA ENGINE & STORICI (Yahoo + Morningstar)")
    st.info("Scarica serie storiche per Analisi Tecnica o Esportazione. Supporta ETF/Fondi Morningstar (ISIN).")
    
    # Cheat Sheet Tab
    with st.expander("📋 CODICI UTILI & ALIAS (Clicca per espandere)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🇺🇸 INDICI**\n- S&P500: `^GSPC`\n- NASDAQ: `^NDX`\n- VIX: `^VIX`")
            st.markdown("**🇪🇺 INDICI**\n- MIB: `FTSEMIB.MI`\n- DAX: `^GDAXI`")
        with c2:
            st.markdown("**🛢️ COMMODITIES**\n- Oro: `GC=F`\n- Bitcoin: `BTC-USD`")
            st.markdown("**🇮🇹 MILANO**\n- Usa suffisso `.MI` (es. `SWDA.MI`, `ISP.MI`)")

    # Configuration Area
    col_input, col_conf = st.columns([2, 1])
    
    with col_input:
        raw_input = st.text_area("Lista Tickers / ISIN (uno per riga o separati da virgola)", 
                                 value="SP500\nSWDA.MI\nLU1287022708\nGOLD", height=150)
    
    with col_conf:
        years = st.selectbox("Anni Storico", [1, 3, 5, 10, 20], index=1)
        freq_options = {"Giornaliero": "D", "Settimanale": "W", "Mensile": "ME"}
        selected_freq_label = st.selectbox("Frequenza", list(freq_options.keys()))
        selected_freq_code = freq_options[selected_freq_label]
        
    start_date = datetime.now() - timedelta(days=years*365)
    end_date = datetime.now()

    # Alias Logic
    ALIAS_MAP = {
        "SP500": "^GSPC", "S&P500": "^GSPC", "NASDAQ": "^NDX", "NASDAQ100": "^NDX",
        "DOWJONES": "^DJI", "DAX": "^GDAXI", "CAC40": "^FCHI", "ESTX50": "^STOXX50E",
        "EUROSTOXX": "^STOXX50E", "VIX": "^VIX", "GOLD": "GC=F", "OIL": "CL=F",
        "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "EURUSD": "EURUSD=X"
    }
    
    # Processing Input
    if st.button("🔥 ESEGUI ESTRAZIONE DATI", type="primary", use_container_width=True):
        raw_tokens = re.findall(r"[\w\.\-\^\=]+", raw_input.upper())
        tickers_input = []
        for t in raw_tokens:
            if t in ALIAS_MAP: tickers_input.append(ALIAS_MAP[t])
            else: tickers_input.append(t)
            
        all_series = {}
        with st.spinner('Scaricamento dati da Yahoo e Morningstar...'):
            for t in tickers_input:
                series = None
                # 1. Try Yahoo
                series = get_data_yahoo_series(t, start_date)
                # 2. Try Morningstar if Yahoo fails
                if series is None:
                    series = get_data_morningstar_series(t, start_date, end_date)
                
                if series is not None:
                    series.name = t
                    all_series[t] = series
                else:
                    st.warning(f"⚠️ Dato non trovato: {t}")

        if all_series:
            df_daily = pd.DataFrame(all_series).ffill().dropna()
            
            # Resampling
            if selected_freq_code == "D":
                df_final = df_daily
                ann_factor = 252
            else:
                df_final = df_daily.resample(selected_freq_code).last()
                ann_factor = 52 if selected_freq_code == "W" else 12

            # Metrics Calculation
            metrics = []
            for col in df_final.columns:
                s = df_final[col]
                if len(s) > 1:
                    returns = s.pct_change().dropna()
                    tot_ret = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
                    vol = returns.std() * np.sqrt(ann_factor) * 100
                    roll_max = s.cummax()
                    drawdown = (s - roll_max) / roll_max
                    max_dd = drawdown.min() * 100
                    
                    metrics.append({
                        "Ticker": col,
                        "Prezzo Ultimo": round(s.iloc[-1], 2),
                        "Rend. Tot %": round(tot_ret, 2),
                        "Volatilità %": round(vol, 2),
                        "Max DD %": round(max_dd, 2)
                    })

            # 1. Table
            st.subheader(f"📅 Serie Storiche ({selected_freq_label})")
            st.dataframe(df_final.sort_index(ascending=False).round(2), use_container_width=True, height=400)
            
            # 2. Charts & Metrics
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("📈 Performance (Base 100)")
                if not df_final.empty:
                    df_b100 = (df_final / df_final.iloc[0]) * 100
                    st.line_chart(df_b100)
            with c2:
                st.subheader("🏆 KPI Analisi")
                st.dataframe(pd.DataFrame(metrics).set_index("Ticker"), use_container_width=True)

            st.markdown("---")
            
            # 3. Correlation (Matplotlib/Seaborn Integration)
            st.subheader("🔗 Matrice di Correlazione")
            if len(df_final.columns) > 1:
                corr = df_final.pct_change().corr()
                # Use context to avoid affecting global pyplot style
                with plt.style.context("dark_background"):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.heatmap(corr, annot=True, cmap="RdYlGn", fmt=".2f", vmin=-1, vmax=1, ax=ax)
                    st.pyplot(fig)

            # 4. Download
            df_final.index.name = "Data"
            csv = df_final.to_csv(sep=";", decimal=",", encoding="utf-8-sig")
            st.download_button(
                label=f"📥 SCARICA CSV ({selected_freq_label.upper()})", 
                data=csv, 
                file_name=f"Analisi_{selected_freq_label}.csv", 
                mime="text/csv",
                type="primary"
            )
        else:
            st.error("Nessun dato valido estratto. Verifica i ticker.")

# =========================
# MAIN
# =========================
def main():
    init_session_state()
    
    st.sidebar.title("TERMINAL")
    st.sidebar.caption("Institutional v4.0 (Data Engine Added)")
    
    # Navigation
    page = st.sidebar.radio("Navigazione", 
        ["📊 Valuation Dashboard", "🔍 Deep Dive", "🌍 Market Map", "📥 Data Engine & Storici"]
    )
    
    if page != "📥 Data Engine & Storici":
        st.sidebar.subheader("¶ Macro Assumptions")
        st.session_state.model.rf = st.sidebar.number_input("Risk Free Rate", 0.0, 0.20, 0.04, 0.001, format="%.3f")
        st.session_state.model.erp = st.sidebar.number_input("Equity Risk Premium", 0.0, 0.20, 0.05, 0.001, format="%.3f")
        st.session_state.model.gdp = st.sidebar.number_input("GDP Growth (L-T)", -0.05, 0.10, 0.02, 0.001, format="%.3f")
        st.sidebar.divider()

    if page == "📊 Valuation Dashboard":
        page_overview()
    elif page == "🔍 Deep Dive":
        page_deep_dive()
    elif page == "🌍 Market Map":
        page_market_view()
    elif page == "📥 Data Engine & Storici":
        page_data_engine()

if __name__ == "__main__":
    main()
