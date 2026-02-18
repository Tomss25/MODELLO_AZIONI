import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict
import io
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# =========================
# 1. CONFIGURAZIONE & STILE "BLOOMBERG TERMINAL"
# =========================
st.set_page_config(
    page_title="Wealth Model Institutional Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Professionale (Dark Sidebar, Light Content, Font Roboto)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Intestazioni */
    h1, h2, h3 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* KPI Cards */
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
    
    /* Tabelle Dati */
    div[data-testid="stDataFrame"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 12px;
    }

    /* SIDEBAR: Dark Mode Forzata */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    
    /* Testi bianchi nella sidebar */
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

    /* Bottoni Action */
    div.stButton > button {
        background-color: #FF6600; /* Bloomberg Orange */
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
        # I parametri macro vengono gestiti dalla sidebar
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
        """
        Scarica dati da Yahoo Finance con logica difensiva per evitare crash
        su colonne mancanti o formati imprevisti.
        """
        status_msg = ""
        tickers_list = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
        
        if not tickers_list:
            raise ValueError("Inserisci almeno un Ticker valido.")

        # --- FASE 1: SCARICO PREZZI (Resiliente) ---
        try:
            # Scarica dati grezzi senza aggiustamenti automatici per controllare le colonne
            raw_df = yf.download(tickers_list, period="2y", interval="1wk", progress=False, auto_adjust=False)
            
            if raw_df.empty:
                raise ValueError("Yahoo non ha restituito dati. Controlla i Ticker.")

            data = pd.DataFrame()

            # Gestione MultiIndex (quando ci sono più ticker)
            if isinstance(raw_df.columns, pd.MultiIndex):
                # Prova a prendere 'Adj Close', se fallisce prendi 'Close'
                try:
                    data = raw_df.xs('Adj Close', axis=1, level=0, drop_level=True)
                except KeyError:
                    try:
                        data = raw_df.xs('Close', axis=1, level=0, drop_level=True)
                    except KeyError:
                        # Se tutto fallisce, prendi le prime N colonne
                        data = raw_df.iloc[:, :len(tickers_list)]
            
            # Gestione Index Singolo (un solo ticker)
            else:
                # Cerca la colonna corretta
                tgt_col = None
                for candidate in ['Adj Close', 'Close', 'AdjClose']:
                    if candidate in raw_df.columns:
                        tgt_col = candidate
                        break
                
                if tgt_col:
                    data = raw_df[[tgt_col]].copy()
                else:
                    # Prendi l'ultima colonna disponibile
                    data = raw_df.iloc[:, -1:].copy()
                
                # Rinomina la colonna con il nome del ticker
                data.columns = [tickers_list[0]]

            # Pulizia finale prezzi
            data.index = pd.to_datetime(data.index).tz_localize(None)
            self.price_data = data.ffill().dropna(how="all")
            
            if self.price_data.empty:
                 raise ValueError("Dati prezzi vuoti dopo il download.")

            self.returns = np.log(self.price_data / self.price_data.shift(1)).dropna(how="all")
            status_msg += f"✅ Market Data: {len(tickers_list)} tickers recuperati.\n"

        except Exception as e:
            raise ValueError(f"Errore critico download prezzi: {e}")

        # --- FASE 2: SCARICO FONDAMENTALI ---
        fund_list = []
        progress_bar = st.progress(0)
        
        for i, t in enumerate(tickers_list):
            try:
                stock = yf.Ticker(t)
                info = stock.info
                
                # Recupero dati con fallback (se manca X, usa Y)
                # EBIT
                ebit = info.get('ebit')
                if ebit is None: ebit = info.get('ebitda')
                if ebit is None: ebit = (info.get('totalRevenue', 0) or 0) * 0.15
                
                sales = info.get('totalRevenue', 0)
                
                # Tax Rate
                tax_rate = info.get('taxRate', 0.25)
                if tax_rate is None: tax_rate = 0.25

                # Net Debt
                d = info.get('totalDebt') or 0
                c = info.get('totalCash') or 0
                net_debt = d - c
                
                # Prezzo Corrente
                curr_price = info.get('currentPrice') or info.get('regularMarketPreviousClose') or 0.0
                if curr_price == 0 and t in self.price_data.columns:
                    curr_price = self.price_data[t].iloc[-1]

                # Shares Out
                shares = info.get('sharesOutstanding') or 0
                if (shares == 0) and (curr_price > 0):
                    shares = (info.get('marketCap') or 0) / curr_price

                # ROIC & Payout
                roic = info.get('returnOnEquity', 0.12) or 0.12
                payout = info.get('payoutRatio', 0.0) or 0.0

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
            except Exception:
                # Se un ticker è corrotto, lo saltiamo senza bloccare gli altri
                pass
            
            progress_bar.progress((i + 1) / len(tickers_list))

        progress_bar.empty()
        self.fund_data = pd.DataFrame(fund_list)
        
        if self.fund_data.empty:
            raise ValueError("Impossibile scaricare i fondamentali. Verifica la connessione o i ticker.")
            
        status_msg += f"✅ Fundamentals: {len(self.fund_data)} aziende analizzate."
        return status_msg

    def calculate_betas(self) -> None:
        if self.returns.empty: return
        
        # Benchmark interno (Media del portafoglio)
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
                self.betas[t] = max(0.4, min(2.5, beta)) # Capping prudenziale
            except:
                self.betas[t] = 1.0

    def run_valuation(self) -> pd.DataFrame:
        results = []
        self.sim_storage = {}
        # Scomposizione Cholesky per correlazione scenari
        try:
            L = np.linalg.cholesky(CONFIG["CORRELATION_MATRIX"])
        except:
            L = np.eye(3) # Fallback se matrice non positiva definita

        for _, r in self.fund_data.iterrows():
            ticker = _safe_str(r.get("Ticker"))
            try:
                shares = _to_float(r.get("SharesOut"))
                ebit = _to_float(r.get("EBIT_TTM"))
                nd = _to_float(r.get("NetDebt"))
                tax = _to_float(r.get("TaxRate"))
                curr_price = _to_float(r.get("CurrentPrice"))

                if shares <= 0 or ebit <= 0: continue

                # Calcolo WACC Dinamico
                beta = self.betas.get(ticker, 1.0)
                ke = self.rf + beta * self.erp
                kd = self.rf + CONFIG["CREDIT_SPREAD"]
                
                equity_val = shares * curr_price
                ev = equity_val + max(0, nd)
                
                wd = 0.0
                if ev > 0:
                    wd = min(max(nd, 0) / ev, CONFIG["MAX_DEBT_WEIGHT"])
                
                wacc_base = max(((1 - wd) * ke + wd * kd * (1 - tax)), CONFIG["MIN_WACC"])
                nopat_base = ebit * (1 - tax)

                # Monte Carlo Vectorized
                Z = np.random.normal(0, 1, size=(3, CONFIG["MC_SIMULATIONS"]))
                shocks = L @ Z

                wacc_sim = np.clip(wacc_base * (1 + shocks[0] * CONFIG["WACC_VOL"]), CONFIG["MIN_WACC"], CONFIG["MAX_WACC"])
                gdp_sim = np.clip(self.gdp * (1 + shocks[1] * CONFIG["GROWTH_VOL"]), self.gdp + CONFIG["GDP_MIN_OFFSET"], self.gdp + CONFIG["GDP_MAX_OFFSET"])
                nopat_sim = nopat_base * (1 + shocks[2] * CONFIG["MARGIN_VOL"])

                fcff_sum = np.zeros(CONFIG["MC_SIMULATIONS"])
                curr_nopat = nopat_sim
                
                # DCF 10 Anni
                for t in range(1, 11):
                    # Crescita che decade verso GDP
                    g_t = np.maximum(gdp_sim, (wacc_sim + 0.015) * np.exp(-0.2 * t)) 
                    # Reinvestimento implicito 30%
                    f = curr_nopat * (1 + g_t) * 0.70 
                    fcff_sum += f / ((1 + wacc_sim) ** t)
                    curr_nopat *= (1 + g_t)

                # Terminal Value
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
# 4. UI: DASHBOARD & NAVIGAZIONE
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
    
    # Visualizzazione Macro (Solo lettura qui, modifica in Sidebar)
    k1, k2, k3 = st.columns(3)
    k1.metric("Risk Free (Rf)", f"{st.session_state.model.rf:.2%}")
    k2.metric("Equity Risk Prem. (ERP)", f"{st.session_state.model.erp:.2%}")
    k3.metric("L-T Growth (GDP)", f"{st.session_state.model.gdp:.2%}")

    st.divider()

    # Input Area
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

    # Override Manuale (Cruciale)
    if st.session_state.data_fetched:
        st.subheader("🛠️ REVISIONE DATI FONDAMENTALI")
        st.info("Modifica i valori nella tabella se necessario (doppio click sulla cella).")
        
        edited_df = st.data_editor(
            st.session_state.model.fund_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn(disabled=True),
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
        
        if st.button("🚀 LANCIA SIMULAZIONE MONTE CARLO", type="primary", use_container_width=True):
             with st.spinner(f"Esecuzione {CONFIG['MC_SIMULATIONS']} scenari vettoriali..."):
                st.session_state.model.calculate_betas()
                st.session_state.results = st.session_state.model.run_valuation()
                st.rerun()

    # Output Risultati
    if st.session_state.results is not None:
        st.subheader("📋 VALUTAZIONE FINALE")
        df = st.session_state.results
        
        # Tentativo di formattazione colori (Safe Mode)
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
            st.warning("Matplotlib non trovato. Visualizzazione standard.")
            st.dataframe(df, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="VALUATION", index=False)
        st.download_button("📥 DOWNLOAD REPORT EXCEL", buffer.getvalue(), "Wealth_Model_Output.xlsx")

def page_deep_dive():
    st.header("🔍 ANALISI SINGOLO TITOLO")
    
    if st.session_state.results is None:
        st.warning("⚠️ Esegui prima la simulazione nella Dashboard.")
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

    # Plotly Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim_data, nbinsx=60, name="Distribuzione FV", marker_color='#334155', opacity=0.75))
    fig.add_vline(x=row['CurrentPrice'], line_width=3, line_dash="dash", line_color="#FF6600", annotation_text="Prezzo Oggi")
    fig.add_vline(x=row['FairValue'], line_width=3, line_color="#10B981", annotation_text="Fair Value")
    
    fig.update_layout(title="Densità di Probabilità Fair Value (Monte Carlo)", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

def page_market_view():
    st.header("🌍 MARKET MAP")
    if st.session_state.results is None:
        st.warning("⚠️ Esegui prima la simulazione.")
        return
    df = st.session_state.results
    
    # Scatter Plot Risk/Reward
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
    st.sidebar.caption("Institutional v3.0 Final")
    st.sidebar.subheader("¶ Macro Assumptions")
    
    # Input Macro Sidebar - Controllo Globale
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
