"""
Dashboard Perbandingan ARIMA vs FTS-TLUS — Saham BRIS.JK
Tampilan bergaya Stockbit (dark theme, kartu harga, grafik interaktif).

Cara menjalankan (lokal):
    pip install streamlit pandas numpy plotly
    streamlit run app.py

Struktur folder yang dibutuhkan:
    app.py
    data/
        forecast_comparison.csv
        model_comparison.csv
        BRIS_clean_data.csv
        BRIS_train.csv
        BRIS_test.csv
        fts_fuzzy_sets.csv
        fts_tlus_table.csv
        FTS_TLUS_forecast.csv
        ARIMA_forecast.csv
        validation_checklist.csv
        arima_model_info.json
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN & TEMA STOCKBIT
# ============================================================

st.set_page_config(
    page_title="BRIS.JK | ARIMA vs FTS-TLUS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TICKER = "BRIS.JK"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GREEN = "#00C896"     # warna kenaikan ala Stockbit
RED = "#FF4B4B"       # warna penurunan
BLUE = "#3D7BFF"       # aksen ARIMA
YELLOW = "#FFC94D"     # aksen FTS-TLUS
BG_DARK = "#0E1117"
CARD_BG = "#161B22"
BORDER = "#262B33"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG_DARK};
    }}
    /* Kartu metrik ala Stockbit */
    .stockbit-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }}
    .stockbit-label {{
        color: #8B949E;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .stockbit-value {{
        color: #F0F3F6;
        font-size: 26px;
        font-weight: 700;
        line-height: 1.1;
    }}
    .stockbit-sub {{
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }}
    .ticker-header {{
        display: flex;
        align-items: baseline;
        gap: 14px;
        flex-wrap: wrap;
    }}
    .ticker-name {{
        font-size: 34px;
        font-weight: 800;
        color: #F0F3F6;
    }}
    .ticker-price {{
        font-size: 34px;
        font-weight: 800;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
    }}
    .badge-green {{ background-color: rgba(0,200,150,0.15); color: {GREEN}; }}
    .badge-red {{ background-color: rgba(255,75,75,0.15); color: {RED}; }}
    .badge-blue {{ background-color: rgba(61,123,255,0.15); color: {BLUE}; }}
    .badge-yellow {{ background-color: rgba(255,201,77,0.15); color: {YELLOW}; }}
    .section-title {{
        color: #F0F3F6;
        font-size: 18px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 10px;
        border-left: 4px solid {BLUE};
        padding-left: 10px;
    }}
    div[data-testid="stMetricValue"] {{
        color: #F0F3F6;
    }}
    thead tr th {{
        background-color: {CARD_BG} !important;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# LOAD DATA (cached agar tidak dibaca ulang setiap interaksi)
# ============================================================

@st.cache_data
def load_all_data():
    comparison = pd.read_csv(os.path.join(DATA_DIR, "forecast_comparison.csv"), parse_dates=["Date"])
    metrics = pd.read_csv(os.path.join(DATA_DIR, "model_comparison.csv"))
    clean = pd.read_csv(os.path.join(DATA_DIR, "BRIS_clean_data.csv"), parse_dates=["Date"])
    train = pd.read_csv(os.path.join(DATA_DIR, "BRIS_train.csv"), parse_dates=["Date"])
    test = pd.read_csv(os.path.join(DATA_DIR, "BRIS_test.csv"), parse_dates=["Date"])
    fuzzy_sets = pd.read_csv(os.path.join(DATA_DIR, "fts_fuzzy_sets.csv"))
    tlus_table = pd.read_csv(os.path.join(DATA_DIR, "fts_tlus_table.csv"))
    fts_forecast = pd.read_csv(os.path.join(DATA_DIR, "FTS_TLUS_forecast.csv"), parse_dates=["Date"])
    arima_forecast = pd.read_csv(os.path.join(DATA_DIR, "ARIMA_forecast.csv"), parse_dates=["Date"])
    checklist = pd.read_csv(os.path.join(DATA_DIR, "validation_checklist.csv"))
    with open(os.path.join(DATA_DIR, "arima_model_info.json")) as f:
        arima_info = json.load(f)
    return {
        "comparison": comparison, "metrics": metrics, "clean": clean,
        "train": train, "test": test, "fuzzy_sets": fuzzy_sets,
        "tlus_table": tlus_table, "fts_forecast": fts_forecast,
        "arima_forecast": arima_forecast, "checklist": checklist,
        "arima_info": arima_info,
    }


try:
    data = load_all_data()
except FileNotFoundError as e:
    st.error(
        f"File data tidak ditemukan: {e}\n\n"
        f"Pastikan folder `data/` berisi seluruh file CSV/JSON hasil notebook, "
        f"berada di direktori yang sama dengan app.py."
    )
    st.stop()

comparison = data["comparison"]
metrics = data["metrics"]
clean = data["clean"]
train = data["train"]
test = data["test"]
fuzzy_sets = data["fuzzy_sets"]
tlus_table = data["tlus_table"]
checklist = data["checklist"]
arima_info = data["arima_info"]

metrics_idx = metrics.set_index("Model")
arima_row = metrics_idx.loc["ARIMA"]
fts_row = metrics_idx.loc["FTS-TLUS"]

last_actual = comparison["Actual"].iloc[-1]
prev_actual = comparison["Actual"].iloc[-2]
last_date = comparison["Date"].iloc[-1]
change_val = last_actual - prev_actual
change_pct = 100 * change_val / prev_actual

last_arima = comparison["ARIMA_Forecast"].iloc[-1]
last_fts = comparison["FTS_TLUS_Forecast"].iloc[-1]

better_mae = "ARIMA" if arima_row["MAE"] < fts_row["MAE"] else "FTS-TLUS"
better_rmse = "ARIMA" if arima_row["RMSE"] < fts_row["RMSE"] else "FTS-TLUS"
better_mape = "ARIMA" if arima_row["MAPE"] < fts_row["MAPE"] else "FTS-TLUS"


# ============================================================
# HELPER TAMPILAN
# ============================================================

def fmt_rp(x):
    return f"Rp{x:,.0f}".replace(",", ".")


def metric_card(label, value, sub=None, sub_color=None):
    sub_html = f'<div class="stockbit-sub" style="color:{sub_color}">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="stockbit-card">
                <div class="stockbit-label">{label}</div>
                <div class="stockbit-value">{value}</div>
                {sub_html}
            </div>""",
        unsafe_allow_html=True,
    )


def badge(text, kind):
    return f'<span class="badge badge-{kind}">{text}</span>'


def style_apply_elementwise(styler, func, subset=None):
    """
    Kompatibel dengan pandas lama (Styler.applymap) dan pandas baru
    (Styler.applymap dihapus, diganti Styler.map) tanpa perlu deteksi versi manual.
    """
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)


# ============================================================
# HEADER ALA STOCKBIT: NAMA SAHAM + HARGA + PERUBAHAN
# ============================================================

price_color = GREEN if change_val >= 0 else RED
arrow = "▲" if change_val >= 0 else "▼"

header_col1, header_col2 = st.columns([3, 2])
with header_col1:
    st.markdown(
        f"""
        <div class="ticker-header">
            <span class="ticker-name">{TICKER}</span>
            <span class="ticker-price" style="color:{price_color}">{fmt_rp(last_actual)}</span>
            <span class="badge badge-{'green' if change_val >= 0 else 'red'}">
                {arrow} {fmt_rp(abs(change_val))} ({change_pct:+.2f}%)
            </span>
        </div>
        <div style="color:#8B949E; font-size:13px; margin-top:2px;">
            Harga penutupan terakhir pada data testing &middot; {last_date.strftime('%d %b %Y')}
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown(
        f"""
        <div style="text-align:right; margin-top:6px;">
            {badge(f"ARIMA{tuple(arima_info[k] for k in ['p','d','q'])}", 'blue')}
            {badge("FTS-TLUS (Table Look-Up Scheme)", 'yellow')}
        </div>
        <div style="text-align:right; color:#8B949E; font-size:12px; margin-top:6px;">
            Training: {train['Date'].min().strftime('%d %b %Y')} – {train['Date'].max().strftime('%d %b %Y')}
            &nbsp;|&nbsp; Testing: {test['Date'].min().strftime('%d %b %Y')} – {test['Date'].max().strftime('%d %b %Y')}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border-color:#262B33; margin-top:6px;'>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — KONTROL DASHBOARD
# ============================================================

st.sidebar.markdown("### ⚙️ Kontrol Tampilan")
show_arima = st.sidebar.checkbox("Tampilkan ARIMA", value=True)
show_fts = st.sidebar.checkbox("Tampilkan FTS-TLUS", value=True)
show_full_history = st.sidebar.checkbox("Tampilkan riwayat training juga", value=False)

st.sidebar.markdown("### 🗓️ Rentang Tanggal Testing")
min_date, max_date = comparison["Date"].min(), comparison["Date"].max()
date_range = st.sidebar.slider(
    "Pilih rentang tanggal",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="DD MMM YYYY",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Tentang Model")
st.sidebar.markdown(
    f"""
    **ARIMA{tuple(arima_info[k] for k in ['p','d','q'])}**
    AIC = {arima_info['AIC']:.2f} · BIC = {arima_info['BIC']:.2f}
    Ljung-Box: {'✅ Lolos (white noise)' if arima_info['ljung_box_pass'] else '⚠️ Tidak lolos sepenuhnya'}

    **FTS-TLUS**
    Jumlah fuzzy set: {len(fuzzy_sets)}
    Jumlah aturan look-up: {len(tlus_table)}
    """
)

checklist_pass = (checklist["Status"] == "PASS").sum()
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"### ✅ Validasi Notebook\n{checklist_pass}/{len(checklist)} item **PASS** (bebas data leakage)"
)
with st.sidebar.expander("Lihat detail checklist"):
    for _, r in checklist.iterrows():
        icon = "✅" if r["Status"] == "PASS" else "⚠️"
        st.markdown(f"{icon} {r['Item']}")

mask = (comparison["Date"] >= date_range[0]) & (comparison["Date"] <= date_range[1])
filtered = comparison.loc[mask].copy()


# ============================================================
# BARIS KARTU METRIK (mirip Open/High/Low/Volume di Stockbit)
# ============================================================

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Prediksi ARIMA (terakhir)", fmt_rp(last_arima),
               f"Selisih {fmt_rp(abs(last_actual - last_arima))} dari aktual",
               GREEN if abs(last_actual - last_arima) < abs(last_actual - last_fts) else RED)
with c2:
    metric_card("Prediksi FTS-TLUS (terakhir)", fmt_rp(last_fts),
               f"Selisih {fmt_rp(abs(last_actual - last_fts))} dari aktual",
               GREEN if abs(last_actual - last_fts) < abs(last_actual - last_arima) else RED)
with c3:
    metric_card("Model dengan MAE Terendah", better_mae,
               f"MAE {min(arima_row['MAE'], fts_row['MAE']):,.2f}", GREEN)
with c4:
    metric_card("Jumlah Observasi Testing", f"{len(comparison)} hari",
               f"{test['Date'].min().strftime('%b %Y')} – {test['Date'].max().strftime('%b %Y')}")

st.markdown("<div class='section-title'>📊 Grafik Harga: Aktual vs Prediksi</div>", unsafe_allow_html=True)


# ============================================================
# GRAFIK UTAMA (Plotly, gaya trading chart gelap)
# ============================================================

fig = go.Figure()

if show_full_history:
    fig.add_trace(go.Scatter(
        x=train["Date"], y=train["Close"], name="Training (historis)",
        line=dict(color="#4B5563", width=1.3), opacity=0.7,
    ))

fig.add_trace(go.Scatter(
    x=filtered["Date"], y=filtered["Actual"], name="Aktual",
    line=dict(color="#F0F3F6", width=2.2),
))

if show_arima:
    fig.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["ARIMA_Forecast"], name=f"ARIMA{tuple(arima_info[k] for k in ['p','d','q'])}",
        line=dict(color=BLUE, width=1.8, dash="solid"),
    ))

if show_fts:
    fig.add_trace(go.Scatter(
        x=filtered["Date"], y=filtered["FTS_TLUS_Forecast"], name="FTS-TLUS",
        line=dict(color=YELLOW, width=1.8, dash="solid"),
    ))

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
    height=460,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(showgrid=False, rangeslider=dict(visible=True, thickness=0.06)),
    yaxis=dict(showgrid=True, gridcolor=BORDER, title="Harga (Rp)"),
    hovermode="x unified",
)
st.plotly_chart(fig, width='stretch')


# ============================================================
# TAB: METRIK AKURASI | ERROR HARIAN | TABEL DATA | FTS DETAIL
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Akurasi Model", "📉 Error Harian", "🧾 Tabel Forecast", "🔍 Detail FTS-TLUS"]
)

with tab1:
    st.markdown("#### Perbandingan MAE / RMSE / MAPE (data testing)")
    mcol1, mcol2, mcol3 = st.columns(3)
    for col, metric_name, unit in zip([mcol1, mcol2, mcol3], ["MAE", "RMSE", "MAPE"], ["Rp", "Rp", "%"]):
        with col:
            bar = go.Figure(go.Bar(
                x=["ARIMA", "FTS-TLUS"],
                y=[arima_row[metric_name], fts_row[metric_name]],
                marker_color=[BLUE, YELLOW],
                text=[f"{arima_row[metric_name]:,.2f}", f"{fts_row[metric_name]:,.2f}"],
                textposition="outside",
            ))
            bar.update_layout(
                template="plotly_dark", plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
                height=280, margin=dict(l=10, r=10, t=40, b=10),
                title=f"{metric_name} ({unit})", showlegend=False,
                yaxis=dict(showgrid=True, gridcolor=BORDER),
            )
            st.plotly_chart(bar, width='stretch')

    st.markdown(
        f"""
        <div class="stockbit-card">
            Pada data testing, model <b style="color:{BLUE if better_mae=='ARIMA' else YELLOW}">{better_mae}</b>
            menghasilkan MAE lebih rendah, model
            <b style="color:{BLUE if better_rmse=='ARIMA' else YELLOW}">{better_rmse}</b> menghasilkan RMSE lebih rendah,
            dan model <b style="color:{BLUE if better_mape=='ARIMA' else YELLOW}">{better_mape}</b> menghasilkan MAPE lebih rendah.
            Interpretasi ini murni deskriptif berdasarkan angka yang dihitung — tanpa uji signifikansi tambahan
            di dashboard ini, selisih kecil antar-model belum dapat disimpulkan berbeda secara statistik.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("#### Error Absolut Harian")
    err_fig = go.Figure()
    if show_arima:
        err_fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered["ARIMA_Absolute_Error"],
                                     name="Error ARIMA", line=dict(color=BLUE, width=1.5)))
    if show_fts:
        err_fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered["FTS_Absolute_Error"],
                                     name="Error FTS-TLUS", line=dict(color=YELLOW, width=1.5)))
    err_fig.update_layout(
        template="plotly_dark", plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="Error Absolut (Rp)", showgrid=True, gridcolor=BORDER),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(err_fig, width='stretch')

    top_err_col1, top_err_col2 = st.columns(2)
    with top_err_col1:
        st.markdown("**5 error terbesar — ARIMA**")
        st.dataframe(
            filtered.nlargest(5, "ARIMA_Absolute_Error")[["Date", "Actual", "ARIMA_Forecast", "ARIMA_Absolute_Error"]]
            .style.format({"Actual": "{:,.0f}", "ARIMA_Forecast": "{:,.0f}", "ARIMA_Absolute_Error": "{:,.0f}"}),
            width='stretch', hide_index=True,
        )
    with top_err_col2:
        st.markdown("**5 error terbesar — FTS-TLUS**")
        st.dataframe(
            filtered.nlargest(5, "FTS_Absolute_Error")[["Date", "Actual", "FTS_TLUS_Forecast", "FTS_Absolute_Error"]]
            .style.format({"Actual": "{:,.0f}", "FTS_TLUS_Forecast": "{:,.0f}", "FTS_Absolute_Error": "{:,.0f}"}),
            width='stretch', hide_index=True,
        )

with tab3:
    st.markdown("#### Tabel Lengkap Actual vs Forecast")
    display_cols = ["Date", "Actual", "ARIMA_Forecast", "FTS_TLUS_Forecast",
                    "ARIMA_Absolute_Error", "FTS_Absolute_Error"]
    styled = filtered[display_cols].copy()
    styled["Date"] = styled["Date"].dt.strftime("%Y-%m-%d")

    def highlight_error(val, threshold):
        color = RED if val > threshold else GREEN
        return f"color: {color}"

    arima_thr = filtered["ARIMA_Absolute_Error"].median()
    fts_thr = filtered["FTS_Absolute_Error"].median()

    styler = styled.style.format({
        "Actual": "{:,.0f}", "ARIMA_Forecast": "{:,.0f}", "FTS_TLUS_Forecast": "{:,.0f}",
        "ARIMA_Absolute_Error": "{:,.1f}", "FTS_Absolute_Error": "{:,.1f}",
    })
    styler = style_apply_elementwise(styler, lambda v: highlight_error(v, arima_thr), subset=["ARIMA_Absolute_Error"])
    styler = style_apply_elementwise(styler, lambda v: highlight_error(v, fts_thr), subset=["FTS_Absolute_Error"])
    st.dataframe(styler, width='stretch', height=420, hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh tabel (CSV)", data=csv_bytes,
                       file_name="forecast_comparison_filtered.csv", mime="text/csv")

with tab4:
    st.markdown("#### Universe of Discourse & Fuzzy Sets")
    st.dataframe(
        fuzzy_sets.style.format({"Lower_Bound": "{:,.2f}", "Upper_Bound": "{:,.2f}", "Midpoint": "{:,.2f}"}),
        width='stretch', hide_index=True,
    )

    st.markdown("#### Table Look-Up Scheme (aturan FLR terpilih)")
    st.dataframe(
        tlus_table.style.format({"Rule_Degree": "{:.4f}", "Midpoint_Konsekuen": "{:,.2f}"}),
        width='stretch', hide_index=True,
    )

    identity_rule = bool((tlus_table["Current_State"] == tlus_table["Selected_Next_State"]).all())
    if identity_rule:
        st.info(
            "Seluruh aturan look-up berbentuk **A_i → A_i**. Secara matematis, pada partisi segitiga, "
            "center-average defuzzifier dengan aturan identitas mengembalikan nilai input — sehingga "
            "forecast FTS-TLUS mendekati prediksi naif (random walk) pada kondisi ini."
        )

    st.markdown("#### Distribusi Fuzzy State pada Data Training")
    hist_fig = go.Figure(go.Bar(
        x=fuzzy_sets["Fuzzy_Set"],
        y=[len(train[(train["Close"] >= r["Lower_Bound"]) & (train["Close"] < r["Upper_Bound"])])
           for _, r in fuzzy_sets.iterrows()],
        marker_color=YELLOW,
    ))
    hist_fig.update_layout(
        template="plotly_dark", plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Jumlah observasi", showgrid=True, gridcolor=BORDER),
    )
    st.plotly_chart(hist_fig, width='stretch')


# ============================================================
# FOOTER
# ============================================================

st.markdown("<hr style='border-color:#262B33;'>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="color:#8B949E; font-size:12px; text-align:center; padding:8px 0;">
        Dashboard analisis akademik &middot; ARIMA vs FTS-TLUS &middot; Saham {TICKER}
        &middot; Data testing {test['Date'].min().strftime('%d %b %Y')} – {test['Date'].max().strftime('%d %b %Y')}
        <br>Seluruh angka dihitung langsung dari notebook penelitian, tidak ada nilai yang direkayasa.
    </div>
    """,
    unsafe_allow_html=True,
)
