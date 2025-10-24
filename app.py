# Streamlit_Weather.py
# -*- coding: utf-8 -*-

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="TRY/DAT Wetter-Analyse", layout="wide")

st.title("TRY/DAT Wetter-Analyse – Einlesen, Grafiken & Temperaturbereiche")

with st.expander("Hinweise zum Format", expanded=False):
    st.markdown("""
- **DAT-Format (TRY)** mit Spalten:  
  `RW  HW  MM  DD  HH  t  p  WR  WG  N  x  RF  B  D  A  E  IL`  
- **Header in Zeile 33**, **Zeile 34 ignorieren**, **Daten ab Zeile 35** (1-basiert).  
- **Trenner**: Tabs/Whitespace, **Dezimaltrenner**: Komma.  
- Standardmäßig werden Solarspalten **B, D, A** entfernt (optional umschaltbar).  
- **HH** ist bei TRY oft **1..24 = Ende der Stunde** → wird zu 0..23 gemappt.
""")

# ---------------- Sidebar: Einstellungen ----------------
st.sidebar.header("Einstellungen")

header_line = st.sidebar.number_input("Header-Zeile (1-basiert)", min_value=1, value=33, step=1)
skip_comment_line = st.sidebar.checkbox("Zeile 34 als Kommentar überspringen", value=True)
data_start_line = st.sidebar.number_input("Datenstart-Zeile (1-basiert)", min_value=1, value=35, step=1)

decimal_char = st.sidebar.selectbox("Dezimaltrenner", options=[",", "."], index=0)
drop_solar = st.sidebar.checkbox("Solarspalten (B, D, A) entfernen", value=True)
hour_is_end = st.sidebar.checkbox("HH = 1..24 (Ende der Stunde)", value=True)
year = st.sidebar.number_input("Jahr für Zeitstempel (0 = ohne Zeit)", min_value=0, max_value=9999, value=2015, step=1)

show_raw = st.sidebar.checkbox("Rohtext-Vorschau (erste 60 Zeilen)", value=False)
preview_rows = st.sidebar.slider("Vorschau-Zeilen DataFrame", 5, 50, 10, 1)

# Darstellung
st.sidebar.subheader("Darstellung")
use_grid = st.sidebar.checkbox("Gitter-/Querlinien anzeigen", value=True)
ref_lines_str = st.sidebar.text_input("Horizontale Referenzlinien (°C, Komma-getrennt)", value="26,28")
def parse_ref_lines(s: str):
    out = []
    for part in s.split(","):
        part = part.strip().replace(",", ".")
        try:
            out.append(float(part))
        except:
            pass
    return out
ref_lines = parse_ref_lines(ref_lines_str)

# Temperaturbänder (anpassbar)
st.sidebar.subheader("Temperaturbänder")
default_bands = [
    ("≤16°C",      -273.15, 16.0),
    ("16–18°C",     16.0,   18.0),
    ("18–20°C",     18.0,   20.0),
    ("20–22°C",     20.0,   22.0),
    ("22–24°C",     22.0,   24.0),
    ("24–26°C",     24.0,   26.0),
    ("26–28°C",     26.0,   28.0),
    ("28–30°C",     28.0,   30.0),
    (">30°C",       30.0,  1e9),
]
bands = default_bands

# ---------------- Upload ----------------
uploaded = st.file_uploader("DAT-Datei hier ablegen oder auswählen", type=["dat", "txt"], accept_multiple_files=False)

def decode_bytes(data: bytes) -> str:
    """Versuche UTF-8, dann Latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("latin-1", errors="ignore")

def load_dat_from_text(text: str, data_start_line: int, decimal: str) -> pd.DataFrame:
    cols = ["RW","HW","MM","DD","HH","t","p","WR","WG","N","x","RF","B","D","A","E","IL"]
    # Wichtig: KEIN dtype=str → damit decimal=',' wirkt und Zahlen direkt korrekt geparst werden.
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"\s+",
        decimal=decimal,
        header=None,
        skiprows=data_start_line - 1,  # 1-basiert → 0-basiert
        engine="python"
    )
    if df.shape[1] != len(cols):
        raise ValueError(f"Unerwartete Spaltenanzahl: {df.shape[1]} statt {len(cols)}. "
                         f"Prüfe Datenstart-Zeile/Trenner.")
    df.columns = cols

    # Sicherstellen, dass t/WG/x numerisch sind (falls einzelne Zeilen doch Strings enthalten)
    num_cols = ["t","WG","x","B","D","A"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Integers „weich“ casten (lassen NaN zu)
    for c in ["RW","HW","MM","DD","HH","p","WR","N","RF","E","IL"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")

    return df

def build_timestamp(df: pd.DataFrame, year: int, hour_is_end: bool) -> pd.DataFrame:
    if year <= 0:
        return df
    out = df.copy()
    hh = pd.to_numeric(out["HH"], errors="coerce").fillna(0).astype(int)
    if hour_is_end:
        hh = (hh - 1).clip(lower=0, upper=23)
    mm = pd.to_numeric(out["MM"], errors="coerce").fillna(1).astype(int).clip(1, 12)
    dd = pd.to_numeric(out["DD"], errors="coerce").fillna(1).astype(int).clip(1, 31)

    out["ts"] = pd.to_datetime(
        {"year": int(year), "month": mm, "day": dd, "hour": hh},
        errors="coerce"
    )
    return out

def summarize_temp_bands(df: pd.DataFrame, temp_col="t", bands_list=None) -> pd.DataFrame:
    if bands_list is None:
        bands_list = bands
    s = pd.to_numeric(df[temp_col], errors="coerce")
    total = int(s.notna().sum())
    rows, cum = [], 0
    for label, lo, hi in bands_list:
        m = (s >= lo) & (s < hi)
        cnt = int(m.sum())
        cum += cnt
        share = round((cnt / total * 100.0), 2) if total > 0 else np.nan
        rows.append({"Band": label, "Stunden": cnt, "Anteil_%": share, "kumuliert_Stunden": cum})
    return pd.DataFrame(rows)

def add_grid(ax):
    if use_grid:
        ax.grid(True, which="both", axis="both", alpha=0.3)

def add_ref_lines(ax, ref_vals, color=None):
    # Nur zeichnen, wenn Werte vorhanden
    if not ref_vals:
        return
    for v in ref_vals:
        ax.axhline(v, linestyle="--", linewidth=1, alpha=0.6)

def plot_timeseries(df: pd.DataFrame):
    if "ts" not in df.columns or df["ts"].isna().all():
        st.info("Keine gültige Zeitachse verfügbar → Zeitreihe/MinMax werden ausgelassen.")
        return
    d = df.set_index("ts").sort_index()

    # Zeitreihe + 24h Mittel
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    pd.to_numeric(d["t"], errors="coerce").plot(ax=ax1, label="Temperatur [°C]")
    pd.to_numeric(d["t"], errors="coerce").rolling("24H").mean().plot(ax=ax1, label="24h-Mittel")
    add_grid(ax1)
    add_ref_lines(ax1, ref_lines)
    ax1.set_xlabel("Zeit"); ax1.set_ylabel("°C"); ax1.set_title("Temperatur – Zeitreihe")
    ax1.legend(); fig1.tight_layout()
    st.pyplot(fig1)

    # Tages-Min/Max
    daily = pd.to_numeric(d["t"], errors="coerce").resample("D").agg(["min","max"])
    fig2, ax2 = plt.subplots(figsize=(11, 3.6))
    daily["min"].plot(ax=ax2, label="Tagesminimum")
    daily["max"].plot(ax=ax2, label="Tagesmaximum")
    add_grid(ax2)
    add_ref_lines(ax2, ref_lines)
    ax2.set_xlabel("Datum"); ax2.set_ylabel("°C"); ax2.set_title("Tages-Min/Max")
    ax2.legend(); fig2.tight_layout()
    st.pyplot(fig2)

def plot_hist_box(df: pd.DataFrame):
    t_series = pd.to_numeric(df["t"], errors="coerce").dropna()

    # Histogramm
    fig3, ax3 = plt.subplots(figsize=(7.5, 3.8))
    ax3.hist(t_series.values, bins=40)
    add_grid(ax3)
    add_ref_lines(ax3, ref_lines)
    ax3.set_xlabel("Temperatur [°C]"); ax3.set_ylabel("Häufigkeit [h]")
    ax3.set_title("Histogramm Temperatur")
    fig3.tight_layout()
    st.pyplot(fig3)

    # Monats-Boxplot (nur mit ts)
    if "ts" in df.columns and df["ts"].notna().any():
        tmp = df.dropna(subset=["ts"]).copy()
        tmp["Monat"] = tmp["ts"].dt.month
        fig4, ax4 = plt.subplots(figsize=(9.5, 3.8))
        tmp.boxplot(column="t", by="Monat", grid=False, ax=ax4)
        add_grid(ax4)
        add_ref_lines(ax4, ref_lines)
        ax4.set_title("Monats-Boxplot Temperatur"); ax4.set_xlabel("Monat"); ax4.set_ylabel("°C")
        fig4.suptitle(""); fig4.tight_layout()
        st.pyplot(fig4)

def plot_custom_variables(df: pd.DataFrame):
    st.subheader("Individuelle Variablen-Grafik")

    # Welche Spalten sind numerisch?
    numeric_cols = [c for c in df.columns if c != "ts" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.info("Keine numerischen Spalten gefunden.")
        return

    default_selection = [c for c in ["t","WG","RF"] if c in numeric_cols]
    sel = st.multiselect("Variablen (y-Achse)", options=numeric_cols, default=default_selection or [numeric_cols[0]])

    colX1, colX2, colX3 = st.columns([1,1,1])
    with colX1:
        x_choice = st.radio("x-Achse", ["Zeit (ts)", "Index"], index=0 if "ts" in df.columns and df["ts"].notna().any() else 1, horizontal=True)
    with colX2:
        resample = st.selectbox("Resampling", ["Keins", "Täglich (D)", "Wöchentlich (W)", "Monatlich (MS)"], index=0)
    with colX3:
        agg = st.selectbox("Aggregation", ["Mittel", "Min", "Max"], index=0)

    # Optional zweite y-Achse
    use_second_axis = False
    if len(sel) == 2:
        use_second_axis = st.checkbox("Zweite y-Achse für zweite Variable", value=False)

    if not sel:
        st.info("Bitte mindestens eine Variable auswählen.")
        return

    # Daten vorbereiten
    data = df.copy()
    if x_choice.startswith("Zeit") and "ts" in data.columns and data["ts"].notna().any():
        data = data.set_index("ts").sort_index()
        can_resample = True
    else:
        data = data.reset_index(drop=True)
        can_resample = False
        if resample != "Keins":
            st.warning("Resampling erfordert eine gültige Zeitachse – wurde deaktiviert.")
            resample = "Keins"

    # Resampling
    if resample != "Keins" and can_resample:
        rule = {"Täglich (D)":"D", "Wöchentlich (W)":"W", "Monatlich (MS)":"MS"}[resample]
        func = {"Mittel":"mean", "Min":"min", "Max":"max"}[agg]
        data = data[sel].resample(rule).agg(func)
    else:
        data = data[sel]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4.2))
    if len(sel) == 1 or not use_second_axis:
        for c in sel:
            pd.to_numeric(data[c], errors="coerce").plot(ax=ax, label=c)
        add_grid(ax)
        ax.set_xlabel("Zeit" if can_resample or x_choice.startswith("Zeit") else "Index")
        ax.set_ylabel("Wert")
        ax.set_title("Individuelle Variablen")
        ax.legend()
    else:
        # Zwei Achsen: sel[0] links, sel[1] rechts
        c1, c2 = sel[0], sel[1]
        pd.to_numeric(data[c1], errors="coerce").plot(ax=ax, label=c1)
        ax2 = ax.twinx()
        pd.to_numeric(data[c2], errors="coerce").plot(ax=ax2, label=c2, linestyle="--")
        add_grid(ax)
        ax.set_xlabel("Zeit" if can_resample or x_choice.startswith("Zeit") else "Index")
        ax.set_ylabel(c1)
        ax2.set_ylabel(c2)
        ax.set_title("Individuelle Variablen (2 Achsen)")
        # Gemeinsame Legende
        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="best")

    fig.tight_layout()
    st.pyplot(fig)

if uploaded is None:
    st.info("Bitte eine **DAT**-Datei hochladen.")
else:
    raw_bytes = uploaded.read()
    text = decode_bytes(raw_bytes)

    if show_raw:
        st.subheader("Rohtext-Vorschau")
        preview = "\n".join(text.splitlines()[:60])
        st.code(preview, language="text")

    try:
        # Zeile 34 optional entfernen
        if skip_comment_line:
            lines = text.splitlines()
            if len(lines) >= 34:
                del lines[33]  # 1-basiert: 34 → Index 33
                text_for_parse = "\n".join(lines)
            else:
                text_for_parse = text
        else:
            text_for_parse = text

        df = load_dat_from_text(
            text_for_parse,
            data_start_line=int(data_start_line),
            decimal=decimal_char
        )

        # Solarspalten ggf. entfernen
        if drop_solar:
            for c in ["B","D","A"]:
                if c in df.columns:
                    df.drop(columns=c, inplace=True)

        # Zeitstempel
        df = build_timestamp(df, year=int(year), hour_is_end=bool(hour_is_end))

        # Preview
        st.subheader("Datenvorschau")
        st.dataframe(df.head(preview_rows))

        # Basisinfos
        tnum = pd.to_numeric(df["t"], errors="coerce")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Zeilen (h)", len(df))
        with c2:
            st.metric("Min Temp [°C]", f"{tnum.min():.1f}" if tnum.notna().any() else "—")
        with c3:
            st.metric("Max Temp [°C]", f"{tnum.max():.1f}" if tnum.notna().any() else "—")
        with c4:
            st.metric("NaN in t", int(tnum.isna().sum()))
        with c5:
            st.metric("WG Mittel [m/s]", f"{pd.to_numeric(df['WG'], errors='coerce').mean():.2f}" if 'WG' in df else "—")

        st.markdown("---")
        st.subheader("Grafiken")
        plot_timeseries(df)
        plot_hist_box(df)

        st.markdown("---")
        st.subheader("Auswertung Temperaturbereiche")
        bands_df = summarize_temp_bands(df, temp_col="t", bands_list=bands)
        st.dataframe(bands_df)

        # Downloads
        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                label="Temperaturbänder als CSV herunterladen",
                data=bands_df.to_csv(index=False).encode("utf-8"),
                file_name="temperaturbaender_summary.csv",
                mime="text/csv",
            )
        with colB:
            st.download_button(
                label="Gereinigte Daten (aktueller Stand) als CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="daten_aufbereitet.csv",
                mime="text/csv",
            )

        st.markdown("---")
        plot_custom_variables(df)

    except Exception as e:
        st.error(f"Fehler beim Laden/Verarbeiten: {e}")
        st.stop()
