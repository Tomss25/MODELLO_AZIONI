import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict
import io
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# =========================
# 1. CONFIGURAZIONE & STILE BLOOMBERG
# =========================
st.set_page_config(
    page_title="Wealth Model Institutional Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS per look "High Finance"
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
# 2. CONFIGURAZIONE PARAMETRI FISSI
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
# 3. LOGICA DEL MODELLO
# =========================
def _to_float(x, default=0.0) -> float:
    if pd.isna(x) or x is None: return float(default)
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
        # Parametri Macro (verranno sovrascritti dalla sidebar)
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
        """Scarica dati fondamentali e prezzi da Yahoo Finance (Versione Robusta)"""
        status_msg = ""
        # Pulizia input
        tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
        
        if not tickers_list:
            raise ValueError("Inserisci almeno un Ticker valido.")

        # 1. SCARICO PREZZI CON CHECK ERRORI
        try:
            # Scarica dataset grezzo
            raw_df = yf.download(tickers_list, period="2y", interval="1wk", progress=False)
            
            if raw_df.empty:
                raise ValueError("Nessun dato trovato su Yahoo. Verifica i Ticker.")

            # Gestione Colonne Resiliente
            if 'Adj Close' in raw_df.columns:
                data = raw_df['Adj Close']
            elif 'Close' in raw_df.columns:
                data = raw_df['Close']
            else:
                data = raw_df # Fallback estremo
            
            # Normalizzazione se Serie o DataFrame singolo
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers_list[0])
            
            if len(tickers_list) == 1 and isinstance(data, pd.DataFrame) and tickers_list[0] not in data.columns:
                 data.columns = [tickers_list[0]]

            # Pulizia Prezzi
            data.index = pd.to_datetime(data.index).tz_localize(None)
            self.price_data = data.ffill().dropna(how="all")
            self.returns = np.log(self.price_data / self.price_data.shift(1)).dropna(how="all")
            status_msg += f"✅ Market Data: {len(tickers_list)} tickers recuperati.\n"
        except Exception as e:
            raise ValueError(f"Errore download prezzi: {e}")

        # 2. SCARICO FONDAMENTALI
        fund_list = []
        progress_bar = st.progress(0)
        
        for i, t in enumerate(tickers_list):
            try:
                stock = yf.Ticker(t)
                fast_info = stock.fast_info 
                info = stock.info
                
                # Logica recupero dati a cascata
                ebit = info.get('ebit')
                if ebit is None: ebit = info.get('ebitda')
                if ebit is None: ebit = info.get('totalRevenue', 0) * 0.15 
                
                sales = info.get('totalRevenue', 0)
                
                # Tax Rate Stima
                try:
                    pretax = info.get('pretaxIncome')
                    tax_prov = info.get('taxProvision')
                    if pretax and tax_prov and pretax != 0:
                        tax_rate = tax_prov / pretax
                    else:
                        tax_rate = info.get('taxRate', 0.25)
                except:
                    tax_rate = 0.25

                if tax_rate is None: tax_rate = 0.25

                # Net Debt
                total_debt = info.get('totalDebt', 0)
                cash = info.get('totalCash', 0)
                if total_debt is None: total_debt = 0
                if cash is None: cash = 0
                net_debt = total_debt - cash
                
                # Shares & Price
                shares = info.get('sharesOutstanding', 0)
                curr_price = 0.0
                
                try:
                    curr_price = fast_info.last_price
                except:
                    curr_price = info.get('currentPrice', info.get('regularMarketPreviousClose', 0))
                
                if curr_price == 0 and t in self.price_data.columns:
                    curr_price = self.price_data[t].iloc[-1]

                if shares is None or shares == 0:
                    mkt_cap = info.get('marketCap', 0)
                    if mkt_cap > 0 and curr_price > 0:
                        shares = mkt_cap / curr_price
                    else:
                        shares = 1.0

                # ROIC & Payout
                roic = info.get('returnOnEquity', 0.12)
                if roic is None: roic = 0.12
                
                payout = info.get('payoutRatio', 0.0)
                if payout is None: payout = 0.0

                fund_list.append({
                    "Ticker": t,
                    "EBIT_TTM": ebit,
                    "Sales": sales,
                    "TaxRate": tax_rate,
                    "ROIC": roic,
                    "PayoutRatio": payout,
                    "NetDebt": net_debt,
                    "SharesOut": shares,
                    "CurrentPrice": curr_price
                })
            except Exception as e:
                print(f"Warning su {t}: {e}")
            
            progress_bar.progress((i + 1) / len(tickers_list))

        progress_bar.empty()
        self.fund_data = pd.DataFrame(fund_list)
        
        if self.fund_data.empty:
            raise ValueError("Impossibile recuperare i dati fondamentali.")
            
        status_msg += f"✅ Fundamentals: {len(self.fund_data)} aziende analizzate."
        return status_msg

    def calculate_betas(self) -> None:
        if self.returns.empty: return
        bench_series = self.returns.mean(axis=1).dropna()
        bench_var = bench_series.var()
        if bench_var == 0: bench_var = 1.0

        for t in self.returns.columns:
            try:
                common = pd.concat([self.returns[t], bench_series], axis=1).dropna()
                if len(common) < 10: 
                    self.betas[t] = 1.0
                    continue
                cov = common.iloc[:, 0].cov(common.iloc[:, 1])
                beta = cov / bench_var
                self.betas[t] = max(0.4, min(2.5, beta))
            except:
                self.betas[t] = 1.0

    def run_valuation(self) -> pd.DataFrame:
        results = []
        self.sim_storage = {}
        L = np.linalg.cholesky(CONFIG["CORRELATION_MATRIX"])

        for _, r in self.fund_data.iterrows():
            ticker = _safe_str(r.get("Ticker"))
            try:
                shares = _to_float(r.get("SharesOut"))
                ebit = _to_float(r.get("EBIT_TTM"))
                nd = _to_float(r.get("NetDebt"))
                tax = _to_float(r.get("TaxRate"))
                curr_price = _to_float(r.get("CurrentPrice"))

                if shares <= 0 or ebit <= 0: continue

                # WACC
                beta = self.betas.get(ticker, 1.0)
                ke = self.rf + beta * self.erp
                kd = self.rf + CONFIG["CREDIT_SPREAD"]
                
                equity_val = shares * curr_price
                ev = equity_val + max(0, nd)
                wd = min(max(nd, 0) / ev, CONFIG["MAX_DEBT_WEIGHT"]) if ev > 0 else 0.0
                
                wacc_base = max(((1 - wd) * ke + wd * kd * (1 - tax)), CONFIG["MIN_WACC"])
                nopat_base = ebit * (1 - tax)

                # Monte Carlo
                Z = np.random.normal(0, 1, size=(3, CONFIG["MC_SIMULATIONS"]))
                shocks = L @ Z

                wacc_sim = np.clip(wacc_base * (1 + shocks[0] * CONFIG["WACC_VOL"]), CONFIG["MIN_WACC"], CONFIG["MAX_WACC"])
                gdp_sim = np.clip(self.gdp * (1 + shocks[1] * CONFIG["GROWTH_VOL"]), self.gdp + CONFIG["GDP_MIN_OFFSET"], self.gdp + CONFIG["GDP_MAX_OFFSET"])
                nopat_sim = nopat_base * (1 + shocks[2] * CONFIG["MARGIN_VOL"])

                fcff_sum = np.zeros(CONFIG["MC_SIMULATIONS"])
                curr_nopat = nopat_sim
                
                for t in range(1, 11):
                    g_t = np.maximum(gdp_sim, (wacc_sim + 0.015) * np.exp(-0.2 * t)) 
                    f = curr_nopat * (1 + g_t) * 0.70 
                    fcff_sum += f / ((1 + wacc_sim) ** t)
                    curr_nopat *= (1 + g_t)

                tv = curr_nopat * (1 + gdp_sim) / np.maximum(wacc_sim - gdp_sim, CONFIG["MIN_TV_SPREAD"])
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
                    "Upside_Pct": (fv_med / curr_price - 1) if curr_price > 0 else 0,
                    "Volatility_Risk": fv_std / fv_med if fv_med != 0 else 0,
                    "Beta": beta,
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
    if 'data_fetched' not in st.session_state:
        st.session_state.data_fetched = False

def page_overview():
    st.header("⚡ WEALTH MODEL - TERMINAL")
    
    # KPI Macro
    k1, k2, k3 = st.columns(3)
    k1.metric("Risk Free (Rf)", f"{st.session_state.model.rf:.2%}")
    k2.metric("Equity Risk Prem. (ERP)", f"{st.session_state.model.erp:.2%}")
    k3.metric("L-T Growth (GDP)", f"{st.session_state.model.gdp:.2%}")

    st.divider()

    # Input & Fetch
    col_input, col_action = st.columns([3, 1])
    with col_input:
        tickers_input = st.text_area("Inserisci Ticker (es: AAPL MSFT ENI.MI LVMH.PA)", 
                                     placeholder="Lista ticker separati da spazio...", height=68)
    with col_action:
        st.write("") 
        st.write("") 
        if st.button("🔎 SCARICA DATI", type="primary", use_container_width=True):
            with st.spinner("Connessione ai Data Feed..."):
                try:
                    msg = st.session_state.model.fetch_from_yahoo(tickers_input)
                    st.session_state.data_fetched = True
                    st.success(msg)
                except Exception as e:
                    st.error(f"Errore Fetch: {e}")

    # Override Manuale
    if st.session_state.data_fetched:
        st.subheader("🛠️ REVISIONE DATI FONDAMENTALI")
        st.info("Modifica i valori nella tabella se necessario (doppio click sulla cella).")
        
        edited_df = st.data_editor(
            st.session_state.model.fund_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "EBIT_TTM": st.column_config.NumberColumn(format="%.0f"),
                "NetDebt": st.column_config.NumberColumn(format="%.0f"),
                "SharesOut": st.column_config.NumberColumn(format="%.2f"),
                "TaxRate": st.column_config.NumberColumn(format="%.2%", min_value=0, max_value=1),
                "ROIC": st.column_config.NumberColumn(format="%.2%", min_value=0, max_value=1),
                "CurrentPrice": st.column_config.NumberColumn(format="%.2f"),
            }
        )
        st.session_state.model.fund_data = edited_df

        st.divider()
        
        if st.button("🚀 LANCIA SIMULAZIONE", type="primary", use_container_width=True):
             with st.spinner(f"Esecuzione {CONFIG['MC_SIMULATIONS']} scenari..."):
                st.session_state.model.calculate_betas()
                st.session_state.results = st.session_state.model.run_valuation()
                st.rerun()

    # Risultati
    if st.session_state.results is not None:
        st.subheader("📋 VALUTAZIONE FINALE")
        df = st.session_state.results
        
        # Gestione Sicura Colori (Richiede matplotlib, fallback se manca)
        try:
            st.dataframe(
                df.style.format({
                    "CurrentPrice": "{:.2f}",
                    "FairValue": "{:.2f}",
                    "Upside_Pct": "{:.2%}",
                    "Volatility_Risk": "{:.2%}",
                    "Beta": "{:.2f}",
                    "WACC": "{:.1%}"
                }).background_gradient(subset=["Upside_Pct"], cmap="RdYlGn", vmin=-0.2, vmax=0.5),
                use_container_width=True, height=500
            )
        except ImportError:
            st.warning("Installare 'matplotlib' per vedere i colori. Visualizzazione standard attiva.")
            st.dataframe(df, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="VALUATION", index=False)
        st.download_button("📥 DOWNLOAD REPORT", buffer.getvalue(), "Wealth_Model_Output.xlsx")

def page_deep_dive():
    st.header("🔍 ANALISI SINGOLO TITOLO")
    
    if st.session_state.results is None:
        st.warning("Esegui prima la simulazione nella Dashboard.")
        return

    df = st.session_state.results
    tickers = df["Ticker"].tolist()
    
    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_ticker = st.selectbox("Seleziona Titolo", tickers)

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
    if st.session_state.results is None:
        st.warning("Esegui prima la simulazione.")
        return
    df = st.session_state.results
    
    fig = px.scatter(
        df, x="Volatility_Risk", y="Upside_Pct", text="Ticker", size="Beta",
        color="Upside_Pct", color_continuous_scale="RdYlGn",
        title="Frontiera Efficiente: Rischio vs Upside"
    )
    fig.update_traces(textposition='top center')
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# MAIN APP CONTROLLER
# =========================
def main():
    init_session_state()
    
    st.sidebar.title("TERMINAL")
    st.sidebar.caption("Institutional v2.3")
    st.sidebar.subheader("¶ Macro Assumptions")
    
    # Input Macro Sidebar
    st.session_state.model.rf = st.sidebar.number_input("Risk Free Rate", 0.0, 0.20, 0.04, 0.001, format="%.3f")
    st.session_state.model.erp = st.sidebar.number_input("Equity Risk Premium", 0.0, 0.20, 0.05, 0.001, format="%.3f")
    st.session_state.model.gdp = st.sidebar.number_input("GDP Growth (L-T)", -0.05, 0.10, 0.02, 0.001, format="%.3f")
    
    st.sidebar.divider()
    page = st.sidebar.radio("Navigazione", ["📊 Dashboard & Dati", "🔍 Deep Dive", "🌍 Market Map"])
    
    if page == "📊 Dashboard & Dati":
        page_overview()
    elif page == "🔍 Deep Dive":
        page_deep_dive()
    elif page == "🌍 Market Map":
        page_market_view()

if __name__ == "__main__":
    main()
