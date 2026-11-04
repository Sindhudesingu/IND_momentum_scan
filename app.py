
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="NSE Momentum Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("🇮🇳 NSE Stock Momentum Scanner")
st.caption("Daily momentum screening | NSE Mainboard")

RESULTS_DIR = Path("results")

FILES = {
    "Strict Momentum": RESULTS_DIR / "strict_momentum.csv",
    "Early Momentum": RESULTS_DIR / "early_momentum.csv",
    "All Qualified Stocks": RESULTS_DIR / "latest_all.csv"
}

def load_data(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

strict_df = load_data(FILES["Strict Momentum"])
early_df = load_data(FILES["Early Momentum"])
all_df = load_data(FILES["All Qualified Stocks"])

c1, c2, c3 = st.columns(3)

c1.metric("Strict Momentum", len(strict_df))
c2.metric("Early Momentum", len(early_df))
c3.metric("Qualified Stocks", len(all_df))

st.divider()

selection = st.radio(
    "Select stock category",
    list(FILES.keys()),
    horizontal=True
)

data = {
    "Strict Momentum": strict_df,
    "Early Momentum": early_df,
    "All Qualified Stocks": all_df
}[selection]

st.subheader(selection)

if data.empty:
    st.info("No stocks available in this category yet.")
else:
    st.write(f"Total stocks: {len(data)}")

    search = st.text_input("Search by stock symbol")

    if search:
        if "symbol" in data.columns:
            data = data[
                data["symbol"].astype(str).str.contains(
                    search, case=False, na=False
                )
            ]

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=selection.lower().replace(" ", "_") + ".csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "Data is for screening and research only. "
    "Not a buy or sell recommendation."
)
