
import requests
import pandas as pd
from io import BytesIO

OUTPUT_FILE = "universe.csv"

NSE_EQUITY_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)

NSE_SME_URL = (
    "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,*/*"
}


def download_csv(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def build_nse():
    print("Downloading NSE equity list...")
    equity = download_csv(NSE_EQUITY_URL)

    print("Downloading NSE SME list...")
    sme = download_csv(NSE_SME_URL)

    equity.columns = equity.columns.str.strip().str.upper()
    sme.columns = sme.columns.str.strip().str.upper()

    required = {"SYMBOL", "NAME OF COMPANY", "SERIES"}

    if not required.issubset(equity.columns):
        raise ValueError(
            f"Unexpected NSE equity columns: {list(equity.columns)}"
        )

    equity["SYMBOL"] = equity["SYMBOL"].astype(str).str.strip()
    equity["SERIES"] = (
        equity["SERIES"].astype(str).str.strip().str.upper()
    )

    # Include only normal EQ series.
    equity = equity[equity["SERIES"] == "EQ"].copy()

    # Exclude NSE SME symbols.
    if "SYMBOL" in sme.columns:
        sme_symbols = set(
            sme["SYMBOL"].astype(str).str.strip()
        )
        equity = equity[
            ~equity["SYMBOL"].isin(sme_symbols)
        ]

    # Exclude matching SME ISINs when available.
    if "ISIN NUMBER" in equity.columns and "ISIN NUMBER" in sme.columns:
        sme_isins = set(
            sme["ISIN NUMBER"].dropna().astype(str).str.strip()
        )

        equity = equity[
            ~equity["ISIN NUMBER"].astype(str).str.strip().isin(sme_isins)
        ]

    result = pd.DataFrame({
        "exchange": "NSE",
        "symbol": equity["SYMBOL"],
        "company": equity["NAME OF COMPANY"],
        "yahoo_symbol": equity["SYMBOL"] + ".NS"
    })

    return result.drop_duplicates(subset=["symbol"])


def main():
    nse = build_nse()

    print("NSE eligible symbols:", len(nse))

    if nse.empty:
        raise ValueError("NSE stock universe is empty.")

    nse.to_csv(OUTPUT_FILE, index=False)

    print("NSE universe created successfully.")
    print("Total securities:", len(nse))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
