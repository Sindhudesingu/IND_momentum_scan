import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Perumal Thunai",
    page_icon="📈",
    layout="wide"
)

st.title("Perumal NSE Stock Momentum Scanner")
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


# Load scanner results
strict_df = load_data(FILES["Strict Momentum"])
early_df = load_data(FILES["Early Momentum"])
all_df = load_data(FILES["All Qualified Stocks"])


# Dashboard metrics
c1, c2, c3 = st.columns(3)

c1.metric("Strict Momentum", len(strict_df))
c2.metric("Early Momentum", len(early_df))
c3.metric("Qualified Stocks", len(all_df))

st.divider()


# Stock category selection
selection = st.radio(
    "Select stock category",
    list(FILES.keys()),
    horizontal=True
)

data = {
    "Strict Momentum": strict_df,
    "Early Momentum": early_df,
    "All Qualified Stocks": all_df
}[selection].copy()

st.subheader(selection)


if data.empty:
    st.info("No stocks available in this category yet.")

else:
    st.write(f"Total stocks: {len(data)}")

    # Search by stock symbol
    search = st.text_input(
        "Search by stock symbol",
        placeholder="Example: RELIANCE, TCS, INFY"
    )

    if search.strip():
        if "Symbol" in data.columns:
            filtered_data = data[
                data["Symbol"].astype(str).str.strip().str.contains(
                    search.strip(),
                    case=False,
                    na=False,
                    regex=False
                )
            ]
        else:
            filtered_data = pd.DataFrame()

        if filtered_data.empty:
            st.warning("No matching stock found.")
        else:
            st.success(f"Found {len(filtered_data)} matching stock(s)")

            if "Company" in filtered_data.columns:
                st.subheader("Company Details")
                st.dataframe(
                    filtered_data[["Symbol", "Company"]],
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader("Stock Details")
            st.dataframe(
                filtered_data,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Search Results",
                data=filtered_data.to_csv(index=False).encode("utf-8"),
                file_name="stock_search_results.csv",
                mime="text/csv"
            )

    else:
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
