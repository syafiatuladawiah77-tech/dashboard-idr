from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Analisis Nilai Tukar USD/IDR", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

VARS = ["gold_usd_per_ounce", "usd_index", "close", "idr"]
LABELS = {
    "gold_usd_per_ounce": "Harga Emas (USD/oz)",
    "usd_index": "USD Index",
    "close": "IHSG (close)",
    "idr": "USD/IDR",
    "open": "IHSG Open",
    "high": "IHSG High",
    "low": "IHSG Low",
    "volume": "IHSG Volume",
    "myr": "MYR", "sgd": "SGD", "thb": "THB", "php": "PHP",
    "xau_per_usd": "XAU per USD",
}


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_data() -> pd.DataFrame:
    cur = pd.read_csv(DATA_DIR / "data_currency_daily_rates.csv", parse_dates=["date"])
    gold = pd.read_csv(DATA_DIR / "data_commodity_gold_price.csv", parse_dates=["date"])
    usd = pd.read_csv(DATA_DIR / "data_market_usd_index.csv", parse_dates=["date"])
    ihsg = pd.read_csv(DATA_DIR / "data_market_ihsg.csv", parse_dates=["date"])

    for df in (cur, gold, usd, ihsg):
        df["date"] = pd.to_datetime(df["date"])
        df.drop_duplicates(inplace=True)

    master = (
        cur.merge(gold, on="date", how="inner")
        .merge(usd, on="date", how="inner")
        .merge(ihsg, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    master["idr_change"] = master["idr"].diff()
    master["idr_lag_1"] = master["idr"].shift(1)
    master["idr_lag_7"] = master["idr"].shift(7)
    master["idr_rolling_7"] = master["idr"].rolling(7).mean()
    return master


@st.cache_resource
def train_models(master: pd.DataFrame):
    from xgboost import XGBRegressor

    features = [
        "idr_current", "idr_lag_1", "idr_lag_7", "idr_rolling_7",
        "gold_usd_per_ounce", "usd_index", "close",
    ]
    df_ml = master.copy().sort_values("date").reset_index(drop=True)
    df_ml["idr_current"] = df_ml["idr"]
    df_ml["idr_target"] = df_ml["idr"].shift(-1)
    df_ml = df_ml[["date"] + features + ["idr_target"]].dropna().reset_index(drop=True)

    split = int(len(df_ml) * 0.8)
    train, test = df_ml.iloc[:split], df_ml.iloc[split:]
    Xtr, ytr, Xte, yte = train[features], train["idr_target"], test[features], test["idr_target"]

    def score(y_true, y_pred):
        return dict(
            MAE=mean_absolute_error(y_true, y_pred),
            RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            R2=r2_score(y_true, y_pred),
        )

    preds = {"Naive Baseline": yte.values * 0 + test["idr_current"].values}
    metrics = {"Naive Baseline": score(yte, test["idr_current"])}

    for name, model in [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=300, max_depth=5,
                                                min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                 random_state=42, objective="reg:squarederror", n_jobs=-1)),
    ]:
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        preds[name] = p
        metrics[name] = score(yte, p)
        if name == "XGBoost":
            xgb_model, xgb_features = model, features

    imp = pd.DataFrame({"Feature": xgb_features, "Importance": xgb_model.feature_importances_})
    imp["Label"] = imp["Feature"].map(LABELS).fillna(imp["Feature"])

    res = pd.DataFrame({"Model": metrics.keys(), **{k: [m[k] for m in metrics.values()] for k in ["MAE", "RMSE", "R2"]}})
    res = res.sort_values("RMSE").reset_index(drop=True)

    test_out = test[["date", "idr_current"]].copy()
    for name, p in preds.items():
        test_out[name] = p
    test_out["Aktual"] = yte.values
    defaults = df_ml.iloc[-1][features].to_dict()
    last_date = df_ml["date"].iloc[-1]
    return res, imp, test_out, xgb_model, features, defaults, last_date


def fmt_idr(v):
    return f"Rp{v:,.2f}"


# ---------------------------------------------------------------- page
st.title("📊 Dashboard Analisis Nilai Tukar Rupiah (USD/IDR)")
st.caption("Harga Emas · USD Index · IHSG · Machine Learning (XGBoost) — Syafiatul Adawiah (E1E124017)")

master = load_data()

avail = sorted(master["date"].dt.date.unique().tolist())

with st.sidebar:
    st.header("Filter & Info")
    st.caption("Hanya tanggal yang tersedia di data yang bisa dipilih (akhir pekan/libur tidak ada).")
    start = st.select_slider("Tanggal awal", options=avail, value=avail[0])
    end = st.select_slider("Tanggal akhir", options=avail, value=avail[-1])
    if start > end:
        start, end = end, start
    st.metric("Total hari gabungan", f"{len(master):,}")
    st.caption(f"Periode data: {avail[0]} s/d {avail[-1]}")

s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
dfv = master[(master["date"] >= s) & (master["date"] < e)].copy()

tab_ov, tab_eda, tab_ml, tab_ai = st.tabs(["🏦 Ringkasan", "🔍 EDA & Korelasi", "🤖 Prediksi ML", "💬 Tanya Jawab AI"])

# ---- Ringkasan
with tab_ov:
    if dfv.empty:
        st.warning("Tidak ada data pada rentang ini.")
    else:
        c = st.columns(4)
        c[0].metric("Rata-rata USD/IDR", fmt_idr(dfv["idr"].mean()))
        c[1].metric("Terendah", fmt_idr(dfv["idr"].min()))
        c[2].metric("Tertinggi", fmt_idr(dfv["idr"].max()))
        c[3].metric("Std Deviasi", f"Rp{dfv['idr'].std():,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfv["date"], y=dfv["idr"], name="USD/IDR", mode="lines"))
        fig.add_trace(go.Scatter(x=dfv["date"], y=dfv["idr_rolling_7"], name="Rolling Mean 7 Hari",
                                 mode="lines", line=dict(dash="dash")))
        fig.update_layout(title="Pergerakan Nilai Tukar USD/IDR", height=430,
                          yaxis_title="USD/IDR", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Perubahan harian terbesar")
        up = dfv.loc[dfv["idr_change"].idxmin()] if dfv["idr_change"].notna().any() else None
        dn = dfv.loc[dfv["idr_change"].idxmax()] if dfv["idr_change"].notna().any() else None
        if up is not None and pd.notna(up["idr_change"]):
            k = st.columns(2)
            k[0].metric("Penguatan Rupiah (IDR turun)", f"{up['idr_change']:+,.2f}", delta=str(up["date"].date()))
            k[1].metric("Pelemahan Rupiah (IDR naik)", f"{dn['idr_change']:+,.2f}", delta=str(dn["date"].date()))

        st.subheader("Kurs lain terhadap USD")
        st.plotly_chart(px.line(dfv, x="date", y=["myr", "sgd", "thb", "php"],
                                labels={"value": "Kurs", "variable": "Mata Uang", "date": "Tanggal"}),
                        use_container_width=True)

# ---- EDA
with tab_eda:
    if len(dfv) < 3:
        st.info("Rentang tanggal terlalu pendek (kurang dari 3 hari dengan data). "
                "Korelasi & scatter butuh lebih banyak titik — silakan perluas rentang tanggal di panel kiri.")
    else:
        corr = dfv[VARS].corr()
        st.subheader("Heatmap Korelasi")
        st.plotly_chart(px.imshow(corr.round(2), text_auto=".2f",
                                  x=[LABELS[v] for v in VARS], y=[LABELS[v] for v in VARS],
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                        use_container_width=True)

        corr_idr = corr["idr"].drop("idr").sort_values()
        st.subheader("Kekuatan hubungan terhadap USD/IDR")
        st.plotly_chart(px.bar(x=[LABELS[i] for i in corr_idr.index], y=corr_idr.values,
                               labels={"x": "Variabel", "y": "Korelasi"}, text_auto=".2f"),
                        use_container_width=True)
        strongest = corr_idr.abs().idxmax()
        st.info(f"Variabel dengan hubungan paling kuat: **{LABELS[strongest]}** "
                f"(r = {corr_idr[strongest]:.2f}).")

        st.subheader("Scatter plot")
        col = st.columns(3)
        for ax, var in zip(col, ["gold_usd_per_ounce", "usd_index", "close"]):
            with ax:
                st.plotly_chart(px.scatter(dfv, x=var, y="idr",
                                           labels={"x": LABELS[var], "y": "USD/IDR"},
                                           opacity=0.7), use_container_width=True)

# ---- ML
with tab_ml:
    res, imp, test_out, xgb_model, features, defaults, last_date = train_models(master)

    st.subheader("🎯 Prediksi Manual USD/IDR Hari Berikutnya")
    st.caption("Isi nilai fitur di bawah (sudah terisi otomatis dari data terbaru), lalu tekan "
               "**Hitung Prediksi**. Model XGBoost akan meramal kurs USD/IDR pada hari observasi berikutnya.")

    feature_ui = {
        "idr_current": ("USD/IDR hari ini (t)", 1.0),
        "idr_lag_1": ("USD/IDR kemarin (t-1)", 1.0),
        "idr_lag_7": ("USD/IDR 7 hari lalu (t-7)", 1.0),
        "idr_rolling_7": ("Rata-rata 7 hari", 1.0),
        "gold_usd_per_ounce": ("Harga Emas (USD/oz)", 1.0),
        "usd_index": ("USD Index", 0.01),
        "close": ("IHSG (close)", 1.0),
    }

    with st.form("prediksi_form"):
        cols = st.columns(2)
        inputs = {}
        for i, feat in enumerate(features):
            label, step = feature_ui[feat]
            with cols[i % 2]:
                inputs[feat] = st.number_input(label, value=float(defaults[feat]),
                                               step=step, format="%.4f")
        submitted = st.form_submit_button("Hitung Prediksi", type="primary")

    if submitted:
        row = pd.DataFrame([[inputs[f] for f in features]], columns=features)
        pred = float(xgb_model.predict(row)[0])
        today = inputs["idr_current"]
        delta = pred - today
        st.success(f"Prediksi USD/IDR hari berikutnya: **{fmt_idr(pred)}**")
        arah = "melemah (rupiah turun)" if delta > 0 else "menguat (rupiah naik)"
        st.write(f"Perubahan dari nilai hari ini: **{delta:+,.2f}** poin → {arah}.")
        with st.expander("Catatan akurasi"):
            xgb_row = res[res["Model"] == "XGBoost"].iloc[0]
            st.write(f"Model ini punya MAE ≈ Rp{xgb_row['MAE']:,.2f} dan RMSE ≈ "
                     f"Rp{xgb_row['RMSE']:,.2f} pada data uji. Jadi prediksi bisa meleset "
                     f"sekitar puluhan hingga ratusan poin, terutama saat pergerakan tajam.")

    st.divider()
    st.subheader("Aktual vs Prediksi")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_out["date"], y=test_out["Aktual"], name="Aktual", mode="lines+markers"))
    for m in ["Naive Baseline", "Linear Regression", "Random Forest", "XGBoost"]:
        if m in test_out:
            fig.add_trace(go.Scatter(x=test_out["date"], y=test_out[m], name=m, mode="lines+markers"))
    fig.update_layout(height=430, yaxis_title="USD/IDR", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perbandingan model (time-based split 80/20)")
    st.dataframe(res.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    best = res.iloc[0]["Model"]
    best_rmse = float(res.iloc[0]["RMSE"])
    if best == "Naive Baseline":
        st.warning("Model ML belum mengalahkan **Naive Baseline** — USD/IDR bersifat persisten dan "
                   "sulit diprediksi dari fitur ini. Sesuai catatan metodologi di notebook.")
    else:
        st.success(f"Model terbaik: **{best}** (RMSE {best_rmse:,.2f})")

    st.subheader("Feature Importance (XGBoost)")
    st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Label",
                           orientation="h", labels={"Importance": "Importance", "Label": "Fitur"}),
                    use_container_width=True)

# ---- AI Tanya Jawab
with tab_ai:
    st.subheader("💬 Tanya jawab analisis")
    st.caption("Asisten menjawab berdasarkan angka yang dihitung langsung dari data (tanpa API eksternal).")

    corr = master[VARS].corr()
    corr_idr = corr["idr"].drop("idr")
    res, imp, *_ = train_models(master)
    best = res.iloc[0]["Model"]
    best_row = res.iloc[0]

    def answer(q: str) -> str:
        t = q.lower()
        if re.search(r"pola|pergerakan|trend|fluktuat|rata.?rata", t):
            d = master["idr"]
            return (f"Nilai tukar USD/IDR pada {len(master)} hari gabungan memiliki rata-rata "
                    f"{fmt_idr(d.mean())}, terendah {fmt_idr(d.min())}, tertinggi {fmt_idr(d.max())}, "
                    f"deviasi standar Rp{d.std():,.2f}. Pergerakan cukup fluktuatif dengan periode "
                    f"menguat dan melemah.")
        if re.search(r"korelasi|hubungan|kelsa|pearson", t):
            rows = "\n".join(f"- {LABELS[k]}: {v:+.2f}" for k, v in corr_idr.items())
            return "Korelasi tiap variabel terhadap USD/IDR:\n" + rows
        if re.search(r"kuat|paling|dominan|berpengaruh|faktor", t):
            top = corr_idr.abs().idxmax()
            return (f"Variabel dengan hubungan paling kuat ke USD/IDR adalah **{LABELS[top]}** "
                    f"(r = {corr_idr[top]:+.2f}).")
        if re.search(r"model|prediksi|akurasi|mae|rmse|r2|xgboost|machine learning|ml|performa", t):
            tbl = "\n".join(f"- {r['Model']}: MAE {r['MAE']:.2f}, RMSE {r['RMSE']:.2f}, R² {r['R2']:.4f}"
                            for _, r in res.iterrows())
            note = ("Model tidak mengalahkan Naive Baseline, yang menandakan target sulit diprediksi."
                    if best == "Naive Baseline" else f"Model terbaik berdasarkan RMSE: **{best}**.")
            return "Hasil time-based split 80/20:\n" + tbl + "\n\n" + note
        if re.search(r"fitur|feature|importance|penting", t):
            rows = "\n".join(f"- {r['Label']}: {r['Importance']:.3f}" for _, r in imp.sort_values('Importance', ascending=False).iterrows())
            return "Feature importance XGBoost:\n" + rows
        if re.search(r"kesimpulan|summary|ringkas|selesai|jawaban", t):
            return ("Kesimpulan: (1) USD/IDR fluktuatif dengan rata-rata "
                    f"{fmt_idr(master['idr'].mean())}. (2) IHSG & harga emas berkorelasi negatif kuat, "
                    "USD Index positif sedang. (3) IHSG hubungan paling kuat. (4) "
                    f"Pada data ini model ML {'belum' if best=='Naive Baseline' else 'mengalahkan'} "
                    "naive baseline, menegaskan nilai tukar sulit diprediksi satu langkah ke depan.")
        return ("Aku bisa menjelaskan: pola pergerakan USD/IDR, korelasi antarvariabel, variabel paling "
                "berpengaruh, hasil & akurasi model ML, feature importance, atau kesimpulan. "
                "Silakan pilih contoh pertanyaan di bawah.")

    suggestions = [
        "Bagaimana pola pergerakan USD/IDR?",
        "Apa korelasi tiap variabel terhadap USD/IDR?",
        "Variabel mana yang paling kuat?",
        "Seberapa baik model ML memprediksi?",
        "Fitur apa yang paling penting?",
        "Tolong simpulkan analisis ini",
    ]
    picked = st.segmented_control("Contoh pertanyaan", suggestions, default=None)
    q = st.text_input("atau ketik pertanyaan", placeholder="contoh: kenapa IHSG korelasinya negatif?")
    if picked or q:
        st.markdown(f"**Q:** {q or picked}")
        with st.spinner("menyusun jawaban..."):
            st.markdown(answer(q or picked))
