# Streamlit_Weather.py
# -*- coding: utf-8 -*-

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

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
st.sidebar.header("Einstellungen – Import")
header_line = st.sidebar.number_input("Header-Zeile (1-basiert)", min_value=1, value=33, step=1)
skip_comment_line = st.sidebar.checkbox("Zeile 34 als Kommentar überspringen", value=True)
data_start_line = st.sidebar.number_input("Datenstart-Zeile (1-basiert)", min_value=1, value=35, step=1)
decimal_char = st.sidebar.selectbox("Dezimaltrenner", options=[",", "."], index=0)
drop_solar = st.sidebar.checkbox("Solarspalten (B, D, A) entfernen", value=True)
hour_is_end = st.sidebar.checkbox("HH = 1..24 (Ende der Stunde)", value=True)
year = st.sidebar.number_input("Jahr für Zeitstempel (0 = ohne Zeit)", min_value=0, max_value=9999, value=2015, step=1)

show_raw = st.sidebar.checkbox("Rohtext-Vorschau (erste 60 Zeilen)", value=False)
preview_rows = st.sidebar.slider("Vorschau-Zeilen DataFrame", 5, 50, 10, 1)

st.sidebar.header("Darstellung – Allgemein")
use_grid = st.sidebar.checkbox("Gitter-/Querlinien anzeigen", value=True)
show_legend = st.sidebar.checkbox("Legenden anzeigen", value=True)
ref_lines_str = st.sidebar.text_input("Horizontale Referenzlinien (z. B. 26,28)", value="26,28")

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

# ---------------- Sidebar: Logo & Beschriftungen ----------------
st.sidebar.header("Logo-Overlay")
logo_file = st.sidebar.file_uploader("Logo hochladen (PNG/JPG)", type=["png", "jpg", "jpeg"])
logo_pos = st.sidebar.selectbox("Logo-Position", ["oben rechts", "oben links", "unten rechts", "unten links"], index=0)
logo_scale = st.sidebar.slider("Logo-Größe (% der Axenbreite)", 5, 40, 18, 1)
logo_alpha = st.sidebar.slider("Logo-Transparenz", 0.0, 1.0, 0.85, 0.05)
logo_padding = 0.02  # Abstand zum Rand (als Anteil)

# Beschriftungen je Plot
st.sidebar.header("Beschriftungen – Diagramme")

# Zeitreihe
st.sidebar.subheader("Temperatur – Zeitreihe")
title_ts = st.sidebar.text_input("Titel (Zeitreihe)", "Temperatur – Zeitreihe")
xlabel_ts = st.sidebar.text_input("x-Achse (Zeitreihe)", "Zeit")
ylabel_ts = st.sidebar.text_input("y-Achse (Zeitreihe)", "°C")

# Tages-Min/Max
st.sidebar.subheader("Temperatur – Tages-Min/Max")
title_minmax = st.sidebar.text_input("Titel (Min/Max)", "Tages-Min/Max")
xlabel_minmax = st.sidebar.text_input("x-Achse (Min/Max)", "Datum")
ylabel_minmax = st.sidebar.text_input("y-Achse (Min/Max)", "°C")

# Histogramm
st.sidebar.subheader("Histogramm Temperatur")
title_hist = st.sidebar.text_input("Titel (Histogramm)", "Histogramm Temperatur")
xlabel_hist = st.sidebar.text_input("x-Achse (Histogramm)", "Temperatur [°C]")
ylabel_hist = st.sidebar.text_input("y-Achse (Histogramm)", "Häufigkeit [h]")

# Monats-Boxplot
st.sidebar.subheader("Monats-Boxplot Temperatur")
title_box = st.sidebar.text_input("Titel (Boxplot)", "Monats-Boxplot Temperatur")
xlabel_box = st.sidebar.text_input("x-Achse (Boxplot)", "Monat")
ylabel_box = st.sidebar.text_input("y-Achse (Boxplot)", "°C")

# Custom-Plot
st.sidebar.subheader("Individuelle Variablen")
title_custom = st.sidebar.text_input("Titel (Individuell)", "Individuelle Variablen")
xlabel_custom = st.sidebar.text_input("x-Achse (Individuell)", "Zeit / Index")
ylabel_custom = st.sidebar.text_input("y-Achse links (Individuell)", "Wert")
ylabel2_custom = st.sidebar.text_input("y-Achse rechts (bei 2. Achse)", "Wert 2")

# Temperaturbänder
st.sidebar.header("Temperaturbänder")
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

    # Zahlen sicher parsen
    num_cols = ["t","WG","x","B","D","A"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

def add_ref_lines(ax, ref_vals):
    if not ref_vals:
        return
    for v in ref_vals:
        ax.axhline(v, linestyle="--", linewidth=1, alpha=0.6)

def apply_labels(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

def draw_logo(ax, logo_img: Image.Image, position="oben rechts", scale=0.18, alpha=0.85, pad=0.001):
    """
    Blendet ein Logo als "Inset Axes" ein.
    - scale: Anteil der Axenbreite (0..1)
    - pad: Abstand zum Rand in Achsenkoordinaten (0..0.1 sinnvoll)
    """
    if logo_img is None:
        return
    # Größe des Logos erhalten (Seitenverhältnis)
    w, h = logo_img.size
    aspect = h / w if w else 1.0

    # Breite/Höhe in Achsenkoordinaten
    w_ax = scale
    h_ax = scale * aspect

    # Position
    if position == "oben rechts":
        x0, y0 = 1 - w_ax - pad, 1 - h_ax - pad
    elif position == "oben links":
        x0, y0 = pad, 1 - h_ax - pad
    elif position == "unten rechts":
        x0, y0 = 1 - w_ax - pad, pad
    else:  # unten links
        x0, y0 = pad, pad

    inset_ax = ax.inset_axes([x0, y0, w_ax, h_ax])
    inset_ax.imshow(logo_img)
    inset_ax.set_axis_off()
    # Transparenz via Alpha-Maske simulieren (bei PNG mit Transparenz schon okay),
    # ansonsten via zorder + leichtem alpha der Achse:
    for im in inset_ax.get_images():
        im.set_alpha(alpha)

def get_logo_image():
    if logo_file is None:
        return None
    try:
        img = Image.open(logo_file).convert("RGBA")
        return img
    except Exception:
        return None

def plot_timeseries(df: pd.DataFrame, logo_img):
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
    apply_labels(ax1, title_ts, xlabel_ts, ylabel_ts)
    if show_legend:
        ax1.legend()
    draw_logo(ax1, logo_img, position=logo_pos, scale=logo_scale/100.0, alpha=logo_alpha, pad=logo_padding)
    fig1.tight_layout()
    st.pyplot(fig1)

    # Tages-Min/Max
    daily = pd.to_numeric(d["t"], errors="coerce").resample("D").agg(["min","max"])
    fig2, ax2 = plt.subplots(figsize=(11, 3.6))
    daily["min"].plot(ax=ax2, label="Tagesminimum")
    daily["max"].plot(ax=ax2, label="Tagesmaximum")
    add_grid(ax2)
    add_ref_lines(ax2, ref_lines)
    apply_labels(ax2, title_minmax, xlabel_minmax, ylabel_minmax)
    if show_legend:
        ax2.legend()
    draw_logo(ax2, logo_img, position=logo_pos, scale=logo_scale/100.0, alpha=logo_alpha, pad=logo_padding)
    fig2.tight_layout()
    st.pyplot(fig2)

def plot_hist_box(df: pd.DataFrame, logo_img):
    t_series = pd.to_numeric(df["t"], errors="coerce").dropna()

    # Histogramm
    fig3, ax3 = plt.subplots(figsize=(7.5, 3.8))
    ax3.hist(t_series.values, bins=40)
    add_grid(ax3)
    add_ref_lines(ax3, ref_lines)
    apply_labels(ax3, title_hist, xlabel_hist, ylabel_hist)
    draw_logo(ax3, logo_img, position=logo_pos, scale=logo_scale/100.0, alpha=logo_alpha, pad=logo_padding)
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
        apply_labels(ax4, title_box, xlabel_box, ylabel_box)
        fig4.suptitle("")
        draw_logo(ax4, logo_img, position=logo_pos, scale=logo_scale/100.0, alpha=logo_alpha, pad=logo_padding)
        fig4.tight_layout()
        st.pyplot(fig4)

def plot_custom_variables(df: pd.DataFrame, logo_img):
    st.subheader("Individuelle Variablen-Grafik")

    # Numerische Spalten
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

    if resample != "Keins" and can_resample:
        rule = {"Täglich (D)":"D", "Wöchentlich (W)":"W", "Monatlich (MS)":"MS"}[resample]
        func = {"Mittel":"mean", "Min":"min", "Max":"max"}[agg]
        data = data[sel].resample(rule).agg(func)
    else:
        data = data[sel]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    if len(sel) == 1 or not use_second_axis:
        for c in sel:
            pd.to_numeric(data[c], errors="coerce").plot(ax=ax, label=c)
        add_grid(ax)
        apply_labels(ax, title_custom, xlabel_custom, ylabel_custom)
        if show_legend:
            ax.legend()
    else:
        c1, c2 = sel[0], sel[1]
        pd.to_numeric(data[c1], errors="coerce").plot(ax=ax, label=c1)
        ax2 = ax.twinx()
        pd.to_numeric(data[c2], errors="coerce").plot(ax=ax2, label=c2, linestyle="--")
        add_grid(ax)
        apply_labels(ax, title_custom, xlabel_custom, ylabel_custom)
        ax2.set_ylabel(ylabel2_custom if ylabel2_custom.strip() else c2)
        # Gemeinsame Legende
        if show_legend:
            lines = ax.get_lines() + ax2.get_lines()
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="best")

    draw_logo(ax, get_logo_image(), position=logo_pos, scale=logo_scale/100.0, alpha=logo_alpha, pad=logo_padding)
    fig.tight_layout()
    st.pyplot(fig)

# ---------------- Main Flow ----------------
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

        # Vorschau
        st.subheader("Datenvorschau")
        st.dataframe(df.head(preview_rows))

        # Kennzahlen
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

        # Diagramme
        st.markdown("---")
        st.subheader("Grafiken")
        logo_img = get_logo_image()
        plot_timeseries(df, logo_img)
        plot_hist_box(df, logo_img)

        # Temperaturbänder
        st.markdown("---")
        st.subheader("Auswertung Temperaturbereiche")
        bands_df = summarize_temp_bands(df, temp_col="t", bands_list=bands)
        st.dataframe(bands_df)

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
        plot_custom_variables(df, logo_img)

    except Exception as e:
        st.error(f"Fehler beim Laden/Verarbeiten: {e}")
        st.stop()
