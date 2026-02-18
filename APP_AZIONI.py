import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict
import io
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 1. CONFIGURAZIONE & STILE BLOOMBERG
# =========================
st.set_page_config(
    page_title="Wealth Model Institutional Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS per look "High Finance" corretto
st.markdown("""
<style>
    /* Font principale professionale */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Titoli */
    h1, h2, h3 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* KPI Cards - Stile Bloomberg */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 24px;
        color: #0F172A;
    }
    
    /* Tabelle */
    div[data-testid="stDataFrame"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 12px;
    }

    /* SIDEBAR FIX: Sfondo scuro e Testo Bianco Forzato */
    section[data-testid="stSidebar"] {
        background-color: #0F172A; /* Midnight Blue molto scuro */
    }
    
    /* Forza il colore bianco su tutti gli elementi della sidebar */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    div[data-testid="stRadio"] label p {
        color: #F8FAFC !important;
    }

    /* Stile Radio Button nella Sidebar */
    div[data-testid="stRadio"] > div {
        background-color: transparent;
    }
    
    /* Bottoni */
    div.stButton > button {
        background-color: #FF6600; /* Bloomberg Orange accent */
        color: white;
        border: none;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        background-color: #CC5200;
        border: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. CONFIGURAZIONE PARAMETRI
# =========================
CONFIG: Dict = {
    "TRADING_WEEKS": 52,
    "MC_SIMULATIONS": 5000,
    "SEED": 42,
    "MIN_WACC": 0.045,
    "MAX_WACC": 0.25,
    "MIN_TV_SPREAD": 0.015,
    "CREDIT_SPREAD": 0.02,
    "MAX_DEBT_WEIGHT": 0.60,
    "GDP_MIN": -0.03,
    "GDP_MAX": 0.06,
    "WACC_VOL": 0.15,
    "GROWTH_VOL": 0.25,
    "MARGIN_VOL": 0.10,
    "CORRELATION_MATRIX": np.array([
        [ 1.0, -0.4, -0.3],
        [-0.4,  1.0,  0.6],
        [-0.3,  0.6,  1.0]
    ]),
    "MARKET_COLUMN_CANDIDATES": ["MARKET", "INDEX", "MKT", "SPX", "^GSPC", "SXP"],
}

# =========================
# 3. LOGICA DEL MODELLO
# =========================
def _to_float(x, default=np.nan) -> float:
    if pd.isna(x): return float(default)
    s = str(x).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float(default)

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

    def load_data(self, uploaded_file) -> str:
        status_msg = ""
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                raw = pd.read_csv(uploaded_file, header=None, engine="python")
            else:
                try:
                    raw = pd.read_excel(uploaded_file, sheet_name="INPUT_MASTER", header=None)
                except:
                    raw = pd.read_excel(uploaded_file, header=None)
        except Exception as e:
            raise ValueError(f"Errore lettura file: {e}")

        try:
            # Macro
            try:
                m_idx = raw[raw.iloc[:, 0].astype(str).str.contains("Risk", case=False, na=False)].index[0]
                self.rf = _to_float(raw.iloc[m_idx, 1], default=0.04)
                self.erp = _to_float(raw.iloc[m_idx + 1, 1], default=0.05)
                self.gdp = _to_float(raw.iloc[m_idx + 2, 1], default=0.02)
                status_msg += f"✅ Macro: Rf={self.rf:.1%}, ERP={self.erp:.1%}, GDP={self.gdp:.1%}\n"
            except:
                status_msg += "⚠️ Macro params default.\n"

            # Parsing
            idx_ticker = raw[raw.iloc[:, 0].astype(str).str.contains("Ticker", case=False, na=False)].index[0]
            idx_date = raw[raw.iloc[:, 0].astype(str).str.contains("Date", case=False, na=False)].index[0]

            self.fund_data = raw.iloc[idx_ticker + 1:idx_date - 1].copy()
            self.fund_data.columns = raw.iloc[idx_ticker].astype(str).tolist()
            self.fund_data = self.fund_data.dropna(subset=["Ticker"])
            
            cols_num = ["EBIT_TTM", "TaxRate", "ROIC", "PayoutRatio", "NetDebt", "SharesOut"]
            for c in cols_num:
                if c in self.fund_data.columns:
                    self.fund_data[c] = self.fund_data[c].apply(lambda x: _to_float(x))

            self.price_data = raw.iloc[idx_date + 1:].copy()
            self.price_data.columns = raw.iloc[idx_date].astype(str).tolist()
            self.price_data = self.price_data.dropna(subset=["Date"])
            self.price_data["Date"] = pd.to_datetime(self.price_data["Date"], errors="coerce")
            self.price_data = self.price_data.dropna(subset=["Date"]).set_index("Date").sort_index()

            for col in self.price_data.columns:
                self.price_data[col] = self.price_data[col].apply(lambda x: _to_float(x))
            
            self.price_data.dropna(axis=1, how="all", inplace=True)
            self.returns = np.log(self.price_data / self.price_data.shift(1)).dropna(how="all")
            
            status_msg += f"✅ Database: {len(self.fund_data)} Tickers, {len(self.returns)} Periods."
            return status_msg
        except Exception as e:
            raise ValueError(f"Struttura file non valida: {e}")

    def calculate_betas(self) -> None:
        if self.returns.empty: return
        bench_col = None
        for c in CONFIG["MARKET_COLUMN_CANDIDATES"]:
            if c in self.returns.columns:
                bench_col = c
                break
        
        bench_series = self.returns[bench_col].dropna() if bench_col else self.returns.mean(axis=1).dropna()
        bench_var = bench_series.var()
        if bench_var == 0 or np.isnan(bench_var): bench_var = 1.0

        for t in self.returns.columns:
            if t == bench_col: continue
            common = pd.concat([self.returns[t], bench_series], axis=1).dropna()
            if len(common) < 20: continue 
            cov = common.iloc[:, 0].cov(common.iloc[:, 1])
            beta = cov / bench_var
            self.betas[t] = max(0.2, min(3.0, beta))

    def _get_wacc(self, ticker: str, r: pd.Series) -> float:
        beta = self.betas.get(ticker, 1.0)
        ke = self.rf + beta * self.erp
        kd = self.rf + CONFIG["CREDIT_SPREAD"]
        nd = _to_float(r.get("NetDebt", 0))
        shares = _to_float(r.get("SharesOut", 0))
        tax = _to_float(r.get("TaxRate", 0.25))
        price = 0.0
        if ticker in self.price_data.columns:
            price = self.price_data[ticker].iloc[-1]
        equity = shares * price if (price > 0 and shares > 0) else 0
        ev = equity + max(0, nd)
        wd = min(max(nd, 0) / ev, CONFIG["MAX_DEBT_WEIGHT"]) if ev > 0 else 0.0
        wacc = (1 - wd) * ke + wd * kd * (1 - tax)
        return max(wacc, CONFIG["MIN_WACC"])

    def run_valuation(self) -> pd.DataFrame:
        results = []
        self.sim_storage = {}
        L = np.linalg.cholesky(CONFIG["CORRELATION_MATRIX"])

        for _, r in self.fund_data.iterrows():
            ticker = _safe_str(r.get("Ticker"))
            if not ticker or ticker not in self.price_data.columns: continue

            try:
                shares = _to_float(r.get("SharesOut"))
                ebit = _to_float(r.get("EBIT_TTM"))
                nd = _to_float(r.get("NetDebt"))
                tax = _to_float(r.get("TaxRate", 0.25))
                curr_price = self.price_data[ticker].iloc[-1]

                if shares <= 0 or ebit <= 0 or curr_price <= 0: continue

                wacc_base = self._get_wacc(ticker, r)
                nopat_base = ebit * (1 - tax)

                # Monte Carlo
                Z = np.random.normal(0, 1, size=(3, CONFIG["MC_SIMULATIONS"]))
                shocks = L @ Z

                wacc_sim = np.clip(wacc_base * (1 + shocks[0] * CONFIG["WACC_VOL"]), CONFIG["MIN_WACC"], CONFIG["MAX_WACC"])
                gdp_sim = np.clip(self.gdp * (1 + shocks[1] * CONFIG["GROWTH_VOL"]), CONFIG["GDP_MIN"], CONFIG["GDP_MAX"])
                nopat_sim = nopat_base * (1 + shocks[2] * CONFIG["MARGIN_VOL"])

                fcff_sum = np.zeros(CONFIG["MC_SIMULATIONS"])
                curr_nopat = nopat_sim
                
                for t in range(1, 11):
                    g_t = np.maximum(gdp_sim, (wacc_sim + 0.02) * np.exp(-0.2 * t))
                    reinv = 0.30
                    f = curr_nopat * (1 + g_t) * (1 - reinv)
                    fcff_sum += f / ((1 + wacc_sim) ** t)
                    curr_nopat *= (1 + g_t)

                den = np.maximum(wacc_sim - gdp_sim, CONFIG["MIN_TV_SPREAD"])
                tv = curr_nopat * (1 + gdp_sim) / den
                ev_sim = fcff_sum + (tv / ((1 + wacc_sim) ** 10))
                equity_sim = ev_sim - nd
                fv_sim = equity_sim / shares

                self.sim_storage[ticker] = fv_sim

                fv_med = np.nanmedian(fv_sim)
                fv_std = np.nanstd(fv_sim)
                
                results.append({
                    "Ticker": ticker,
                    "CurrentPrice": curr_price,
                    "FairValue": fv_med,
                    "Upside_Pct": (fv_med / curr_price - 1),
                    "Volatility_Risk": fv_std / fv_med,
                    "Beta": self.betas.get(ticker, 1.0),
                    "WACC": wacc_base
                })
            except Exception:
                continue

        return pd.DataFrame(results).sort_values("Upside_Pct", ascending=False)

# =========================
# 4. PAGINE DASHBOARD
# =========================

def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = WealthModelInstitutional()
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def page_overview():
    st.header("⚡ MARKET DASHBOARD")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.info("📂 **DATA FEED**")
        uploaded_file = st.file_uploader("Upload Institutional Data (CSV/XLSX)", type=["xlsx", "csv"])
        if uploaded_file:
            with st.spinner("Parsing Data Structure..."):
                try:
                    msg = st.session_state.model.load_data(uploaded_file)
                    st.session_state.data_loaded = True
                    st.success(msg)
                except Exception as e:
                    st.error(f"FATAL ERROR: {e}")

    with col2:
        if st.session_state.data_loaded:
            st.warning("⚙️ **EXECUTION ENGINE**")
            if st.button("RUN MONTE CARLO SIMULATION", type="primary", use_container_width=True):
                with st.spinner(f"Running {CONFIG['MC_SIMULATIONS']} vectorized simulations per ticker..."):
                    st.session_state.model.calculate_betas()
                    st.session_state.results = st.session_state.model.run_valuation()
                    st.rerun()

    with col3:
        st.metric("Risk Free Rate", f"{st.session_state.model.rf:.2%}")
        st.metric("Equity Risk Premium", f"{st.session_state.model.erp:.2%}")

    st.divider()

    # MAIN TAPE
    if st.session_state.results is not None:
        df = st.session_state.results
        
        # KPI ROW
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Top Upside", f"{df.iloc[0]['Ticker']} {df.iloc[0]['Upside_Pct']:.1%}")
        k2.metric("Median Market Upside", f"{df['Upside_Pct'].median():.1%}")
        k3.metric("Coverage", f"{len(df)} Tickers")
        k4.metric("Avg WACC", f"{df['WACC'].mean():.1%}")

        st.subheader("📋 VALUATION TAPE")
        
        display_df = df.copy()
        display_df = display_df[["Ticker", "CurrentPrice", "FairValue", "Upside_Pct", "Volatility_Risk", "Beta", "WACC"]]
        
        # Try-Catch per evitare crash se matplotlib manca, ma la soluzione è installarlo
        try:
            st.dataframe(
                display_df.style.format({
                    "CurrentPrice": "{:.2f}",
                    "FairValue": "{:.2f}",
                    "Upside_Pct": "{:.2%}",
                    "Volatility_Risk": "{:.2%}",
                    "Beta": "{:.2f}",
                    "WACC": "{:.1%}"
                }).background_gradient(subset=["Upside_Pct"], cmap="RdYlGn", vmin=-0.2, vmax=0.5),
                use_container_width=True,
                height=500
            )
        except ImportError:
            st.error("⚠️ MANCA 'matplotlib' IN REQUIREMENTS.TXT. Visualizzo tabella semplice.")
            st.dataframe(
                display_df.style.format({
                    "CurrentPrice": "{:.2f}",
                    "FairValue": "{:.2f}",
                    "Upside_Pct": "{:.2%}",
                    "Volatility_Risk": "{:.2%}",
                    "Beta": "{:.2f}",
                    "WACC": "{:.1%}"
                }),
                use_container_width=True,
                height=500
            )

        # Download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            display_df.to_excel(writer, sheet_name="VALUATION", index=False)
        st.download_button("📥 DOWNLOAD EXCEL REPORT", buffer.getvalue(), "Wealth_Model_Output.xlsx")

def page_deep_dive():
    st.header("🔍 SINGLE NAME ANALYSIS")
    
    if st.session_state.results is None:
        st.error("Please run the simulation in the Dashboard first.")
        return

    df = st.session_state.results
    tickers = df["Ticker"].tolist()
    
    col_sel, col_empty = st.columns([1, 3])
    with col_sel:
        selected_ticker = st.selectbox("Select Ticker", tickers)

    row = df[df["Ticker"] == selected_ticker].iloc[0]
    sim_data = st.session_state.model.sim_storage.get(selected_ticker, [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{row['CurrentPrice']:.2f}")
    c2.metric("Fair Value (Median)", f"{row['FairValue']:.2f}", delta=f"{row['Upside_Pct']:.1%}")
    c3.metric("Risk (CV)", f"{row['Volatility_Risk']:.2%}")
    c4.metric("Beta", f"{row['Beta']:.2f}")

    st.subheader(f"🎲 Monte Carlo Distribution: {selected_ticker}")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=sim_data,
        nbinsx=50,
        name="Simulated FV",
        marker_color='#334155',
        opacity=0.75
    ))
    
    fig.add_vline(x=row['CurrentPrice'], line_width=3, line_dash="dash", line_color="#FF6600", annotation_text="Price")
    fig.add_vline(x=row['FairValue'], line_width=3, line_color="#10B981", annotation_text="Fair Value")
    
    fig.update_layout(
        title="Fair Value Probability Density",
        xaxis_title="Price / Value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500,
        bargap=0.1
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("The distribution shows the range of possible Fair Values based on volatility of WACC, Growth, and Margins.")

def page_market_view():
    st.header("🌍 MARKET VIEW & ALPHA")
    
    if st.session_state.results is None:
        st.error("Please run the simulation in the Dashboard first.")
        return
        
    df = st.session_state.results
    
    st.subheader("Risk vs. Reward Frontier")
    
    fig = px.scatter(
        df,
        x="Volatility_Risk",
        y="Upside_Pct",
        text="Ticker",
        size="Beta",
        color="Upside_Pct",
        color_continuous_scale="RdYlGn",
        hover_data=["FairValue", "CurrentPrice"],
        title="Upside Potential vs. Valuation Uncertainty"
    )
    
    fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    
    fig.update_layout(
        xaxis_title="Risk (Coefficient of Variation)",
        yaxis_title="Upside Potential (%)",
        template="plotly_white",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Picks (Undervalued)")
    top_picks = df.head(10).sort_values("Upside_Pct", ascending=True)
    
    fig_bar = px.bar(
        top_picks,
        y="Ticker",
        x="Upside_Pct",
        orientation='h',
        title="Top 10 Highest Upside",
        text_auto='.1%',
        color="Upside_Pct",
        color_continuous_scale="Greens"
    )
    fig_bar.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# MAIN APP CONTROLLER
# =========================
def main():
    init_session_state()
    
    st.sidebar.title("TERMINAL")
    st.sidebar.caption("v2.1 Institutional")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Deep Dive Analysis", "🌍 Market View"],
        index=0
    )
    
    st.sidebar.divider()
    st.sidebar.info(f"Session: {CONFIG['MC_SIMULATIONS']} sims/ticker")
    
    if page == "📊 Dashboard":
        page_overview()
    elif page == "🔍 Deep Dive Analysis":
        page_deep_dive()
    elif page == "🌍 Market View":
        page_market_view()

if __name__ == "__main__":
    main()
