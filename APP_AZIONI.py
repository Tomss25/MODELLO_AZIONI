import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.covariance import LedoitWolf
import io

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Wealth Model Institutional", layout="wide")

# =========================
# LOGGING & CONFIG
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
# HELPERS
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

# =========================
# MODEL LOGIC
# =========================
class WealthModelInstitutional:
    def __init__(self, uploaded_file):
        self.rf = 0.04
        self.erp = 0.05
        self.gdp = 0.02
        self.uploaded_file = uploaded_file # File object da Streamlit
        self.fund_data = pd.DataFrame()
        self.price_data = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.betas = {}
        np.random.seed(CONFIG["SEED"])

    def load_data(self) -> str:
        status_msg = ""
        try:
            # Lettura diretta dal buffer di memoria di Streamlit
            if self.uploaded_file.name.lower().endswith(".csv"):
                raw = pd.read_csv(self.uploaded_file, header=None, engine="python")
            else:
                try:
                    raw = pd.read_excel(self.uploaded_file, sheet_name="INPUT_MASTER", header=None)
                except:
                    raw = pd.read_excel(self.uploaded_file, header=None)
        except Exception as e:
            raise ValueError(f"Errore lettura file: {e}")

        # Individuazione Blocchi Dati
        try:
            # 1. Macro
            try:
                m_idx = raw[raw.iloc[:, 0].astype(str).str.contains("Risk", case=False, na=False)].index[0]
                self.rf = _to_float(raw.iloc[m_idx, 1], default=0.04)
                self.erp = _to_float(raw.iloc[m_idx + 1, 1], default=0.05)
                self.gdp = _to_float(raw.iloc[m_idx + 2, 1], default=0.02)
                status_msg += f"✅ Macro Params: Rf={self.rf:.1%}, ERP={self.erp:.1%}, GDP={self.gdp:.1%}\n"
            except:
                status_msg += "⚠️ Macro params non trovati, uso default.\n"

            # 2. Fondamentali e Prezzi
            idx_ticker = raw[raw.iloc[:, 0].astype(str).str.contains("Ticker", case=False, na=False)].index[0]
            idx_date = raw[raw.iloc[:, 0].astype(str).str.contains("Date", case=False, na=False)].index[0]

        except IndexError:
            raise ValueError("Struttura file non valida: Impossibile trovare le righe 'Ticker' o 'Date'.")

        # Parsing Fondamentali
        self.fund_data = raw.iloc[idx_ticker + 1:idx_date - 1].copy()
        self.fund_data.columns = raw.iloc[idx_ticker].astype(str).tolist()
        self.fund_data = self.fund_data.dropna(subset=["Ticker"])
        
        cols_num = ["EBIT_TTM", "TaxRate", "ROIC", "PayoutRatio", "NetDebt", "SharesOut"]
        for c in cols_num:
            if c in self.fund_data.columns:
                self.fund_data[c] = self.fund_data[c].apply(lambda x: _to_float(x))

        # Parsing Prezzi
        self.price_data = raw.iloc[idx_date + 1:].copy()
        self.price_data.columns = raw.iloc[idx_date].astype(str).tolist()
        self.price_data = self.price_data.dropna(subset=["Date"])
        self.price_data["Date"] = pd.to_datetime(self.price_data["Date"], errors="coerce")
        self.price_data = self.price_data.dropna(subset=["Date"]).set_index("Date").sort_index()

        for col in self.price_data.columns:
            self.price_data[col] = self.price_data[col].apply(lambda x: _to_float(x))
        
        self.price_data.dropna(axis=1, how="all", inplace=True)
        self.returns = np.log(self.price_data / self.price_data.shift(1)).dropna(how="all")
        
        status_msg += f"✅ Dati caricati: {len(self.fund_data)} aziende, {len(self.returns)} settimane."
        return status_msg

    def calculate_betas(self) -> None:
        if self.returns.empty: return
        bench_col = None
        for c in CONFIG["MARKET_COLUMN_CANDIDATES"]:
            if c in self.returns.columns:
                bench_col = c
                break
        
        if bench_col:
            bench_series = self.returns[bench_col].dropna()
        else:
            bench_series = self.returns.mean(axis=1).dropna()

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
        wd = 0.0
        if ev > 0:
            wd = min(max(nd, 0) / ev, CONFIG["MAX_DEBT_WEIGHT"])
        wacc = (1 - wd) * ke + wd * kd * (1 - tax)
        return max(wacc, CONFIG["MIN_WACC"])

    def run_valuation(self) -> pd.DataFrame:
        results = []
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

                fv_med = np.nanmedian(fv_sim)
                results.append({
                    "Ticker": ticker,
                    "CurrentPrice": round(curr_price, 2),
                    "FairValue": round(fv_med, 2),
                    "Upside_Pct": round((fv_med / curr_price - 1), 4),
                    "Volatility_Risk": round(np.nanstd(fv_sim) / fv_med, 4)
                })
            except Exception as e:
                continue

        return pd.DataFrame(results).sort_values("Upside_Pct", ascending=False)

# =========================
# STREAMLIT UI
# =========================
def main():
    st.title("🏦 Wealth Model Institutional")
    st.markdown("### Monte Carlo Valuation Engine")

    # 1. FILE UPLOAD
    uploaded_file = st.file_uploader("Carica file Excel/CSV (con Ticker, Dati e Prezzi)", type=["xlsx", "csv"])

    if uploaded_file is not None:
        model = WealthModelInstitutional(uploaded_file)
        
        with st.spinner("Lettura e analisi dati in corso..."):
            try:
                msg = model.load_data()
                st.success(msg)
            except Exception as e:
                st.error(f"Errore critico: {e}")
                st.stop()

        # 2. RUN BUTTON
        if st.button("🚀 Esegui Valutazione", type="primary"):
            with st.spinner(f"Simulazione Monte Carlo ({CONFIG['MC_SIMULATIONS']} scenari) in corso..."):
                model.calculate_betas()
                df_results = model.run_valuation()
            
            # 3. OUTPUT
            st.divider()
            st.subheader("Risultati Valutazione")
            
            # Formattazione per visualizzazione
            st.dataframe(
                df_results.style.format({
                    "CurrentPrice": "{:.2f} €",
                    "FairValue": "{:.2f} €",
                    "Upside_Pct": "{:.2%}",
                    "Volatility_Risk": "{:.2%}"
                }),
                use_container_width=True,
                height=600
            )

            # 4. DOWNLOAD
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_results.to_excel(writer, sheet_name="VALUATION", index=False)
            
            st.download_button(
                label="📥 Scarica Report Excel",
                data=buffer.getvalue(),
                file_name="WEALTH_MODEL_OUTPUT.xlsx",
                mime="application/vnd.ms-excel"
            )

if __name__ == "__main__":
    main()