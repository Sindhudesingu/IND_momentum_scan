
import logging
import requests
import pandas as pd
import bseindia

OUTPUT_FILE = "universe.csv"

NSE_EQUITY_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)
NSE_SME_URL = (
    ""https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv""
)

# Conservative BSE group allowlist.
# Unknown groups are excluded rather than assumed to be mainboard.
BSE_MAINBOARD_GROUPS = {"A", "B", "T", "XT", "Z", "P", "X"}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,*/*"
}


def download_csv(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    from io import BytesIO
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

    # Only normal EQ series.
    equity["SYMBOL"] = equity["SYMBOL"].astype(str).str.strip()
    equity["SERIES"] = equity["SERIES"].astype(str).str.strip().str.upper()

    equity = equity[equity["SERIES"] == "EQ"].copy()

    # Exclude SME symbols and ISINs when those columns are available.
    if "SYMBOL" in sme.columns:
        sme_symbols = set(
            sme["SYMBOL"].astype(str).str.strip()
        )
        equity = equity[~equity["SYMBOL"].isin(sme_symbols)]

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


def build_bse():
    print("Downloading BSE security master...")

    securities = bseindia.all_listed_securities(refresh=True)

    if securities is None or securities.empty:
        raise ValueError("BSE security master is empty.")

    securities.columns = [
        str(c).strip().lower() for c in securities.columns
    ]

    required = {"security_code", "symbol", "group"}
    if not required.issubset(securities.columns):
        raise ValueError(
            f"Unexpected BSE columns: {list(securities.columns)}"
        )

    securities["group"] = (
        securities["group"].astype(str).str.strip().str.upper()
    )

    # Include only recognized mainboard groups.
    securities = securities[
        securities["group"].isin(BSE_MAINBOARD_GROUPS)
    ].copy()

    securities["security_code"] = (
        securities["security_code"].astype(str).str.strip()
    )

    securities = securities[
        securities["security_code"].str.fullmatch(r"\d+")
    ]

    securities["symbol"] = (
        securities["symbol"].astype(str).str.strip()
    )

    result = pd.DataFrame({
        "exchange": "BSE",
        "symbol": securities["symbol"],
        "company": securities["symbol"],
        "yahoo_symbol": securities["security_code"] + ".BO"
    })

    return result.drop_duplicates(subset=["yahoo_symbol"])


def main():
    nse = build_nse()
    print("NSE eligible symbols:", len(nse))

    bse = build_bse()
    print("BSE eligible symbols:", len(bse))

    combined = pd.concat([nse, bse], ignore_index=True)

    combined = combined.dropna(subset=["yahoo_symbol"])
    combined = combined.drop_duplicates(subset=["yahoo_symbol"])

    if combined.empty:
        raise ValueError("No eligible securities found.")

    combined.to_csv(OUTPUT_FILE, index=False)

    print("Universe created successfully.")
    print("Total securities:", len(combined))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
