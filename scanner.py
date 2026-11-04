
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- SETTINGS ----------------

MIN_PRICE = 5.00
MIN_AVG_VOLUME = 50_000
MIN_AVG_TURNOVER = 2_500_000       # ₹25 lakh
MIN_VOLUME_MULTIPLE = 1.5

RSI_MIN = 50
RSI_MAX = 80
PREFERRED_RSI_MIN = 55
PREFERRED_RSI_MAX = 75

MIN_HISTORY = 30
WORKERS = 5

RESULTS_DIR = "results"
UNIVERSE_FILE = "universe.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


# ---------------- INDICATORS ----------------

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    return rsi


# ---------------- SCAN ONE STOCK ----------------

def scan_stock(stock):
    symbol = stock["yahoo_symbol"]

    try:
        data = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20
        )

        if data is None or data.empty:
            return None

        # Handle yfinance's possible multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = {"Close", "Volume"}
        if not required.issubset(data.columns):
            return None

        data = data[["Close", "Volume"]].copy()
        data = data.dropna()
        data = data[~data.index.duplicated(keep="last")]

        if len(data) < MIN_HISTORY:
            return None

        close = data["Close"].astype(float)
        volume = data["Volume"].astype(float)

        # Indicators use completed daily candles returned by the provider.
        data["RSI"] = calculate_rsi(close)
        data["DMA20"] = close.rolling(20).mean()

        # Daily percentage changes
        data["DailyChange"] = close.pct_change() * 100

        latest = data.iloc[-1]
        previous_20 = data.iloc[-21:-1]

        if len(previous_20) < 20:
            return None

        price = float(latest["Close"])
        latest_volume = float(latest["Volume"])
        avg_volume = float(previous_20["Volume"].mean())

        avg_turnover = float(
            (previous_20["Close"] * previous_20["Volume"]).mean()
        )

        rsi = float(latest["RSI"])
        dma20 = float(latest["DMA20"])

        if not all(np.isfinite(x) for x in
                   [price, latest_volume, avg_volume,
                    avg_turnover, rsi, dma20]):
            return None

        if price < MIN_PRICE:
            return None

        if avg_volume < MIN_AVG_VOLUME:
            return None

        if avg_turnover < MIN_AVG_TURNOVER:
            return None

        if latest_volume < MIN_VOLUME_MULTIPLE * avg_volume:
            return None

        if not (RSI_MIN <= rsi <= RSI_MAX):
            return None

        if price <= dma20:
            return None

        # Four most recent completed daily returns
        last_four = data["DailyChange"].dropna().tail(4)

        if len(last_four) != 4:
            return None

        positive_days = int((last_four > 0).sum())
        four_day_return = (
            (price / float(data["Close"].iloc[-5])) - 1
        ) * 100

        strict = bool((last_four >= 3.0).all())

        early = bool(
            four_day_return >= 3.0 and positive_days >= 3
        )

        if not strict and not early:
            return None

        rsi_label = (
            "Preferred" if PREFERRED_RSI_MIN <= rsi <= PREFERRED_RSI_MAX
            else "Moderate / Extended"
        )

        return {
            "Exchange": stock["exchange"],
            "Symbol": stock["symbol"],
            "Company": stock.get("company", ""),
            "Date": data.index[-1].strftime("%Y-%m-%d"),
            "Close": round(price, 2),
            "4_Session_Return_%": round(four_day_return, 2),
            "Positive_Days": positive_days,
            "Day_1_%": round(float(last_four.iloc[-4]), 2),
            "Day_2_%": round(float(last_four.iloc[-3]), 2),
            "Day_3_%": round(float(last_four.iloc[-2]), 2),
            "Day_4_%": round(float(last_four.iloc[-1]), 2),
            "RSI_14": round(rsi, 2),
            "RSI_Status": rsi_label,
            "DMA_20": round(dma20, 2),
            "Avg_Volume_20D": int(avg_volume),
            "Latest_Volume": int(latest_volume),
            "Volume_Multiple": round(latest_volume / avg_volume, 2),
            "Avg_Turnover_20D_Rs": round(avg_turnover, 0),
            "Strict": strict,
            "Early": early
        }

    except Exception as exc:
        logging.warning("Skipped %s: %s", symbol, exc)
        return None


# ---------------- MAIN SCANNER ----------------

def main():
    if not os.path.exists(UNIVERSE_FILE):
        raise FileNotFoundError(
            "universe.csv is missing. We will create it in the next step."
        )

    universe = pd.read_csv(UNIVERSE_FILE)

    required_columns = {
        "exchange", "symbol", "company", "yahoo_symbol"
    }

    if not required_columns.issubset(universe.columns):
        raise ValueError(
            "universe.csv must contain: exchange, symbol, company, yahoo_symbol"
        )

    universe = universe.dropna(subset=["yahoo_symbol"])
    universe = universe.drop_duplicates(subset=["yahoo_symbol"])

    logging.info("Starting scan for %s symbols", len(universe))

    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(scan_stock, row.to_dict())
            for _, row in universe.iterrows()
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    columns = [
        "Exchange", "Symbol", "Company", "Date", "Close",
        "4_Session_Return_%", "Positive_Days",
        "Day_1_%", "Day_2_%", "Day_3_%", "Day_4_%",
        "RSI_14", "RSI_Status", "DMA_20",
        "Avg_Volume_20D", "Latest_Volume", "Volume_Multiple",
        "Avg_Turnover_20D_Rs", "Strict", "Early"
    ]

    all_results = pd.DataFrame(results, columns=columns)

    if not all_results.empty:
        all_results = all_results.sort_values(
            "4_Session_Return_%", ascending=False
        )

    strict_results = all_results[
        all_results["Strict"] == True
    ].copy()

    early_results = all_results[
        (all_results["Early"] == True) &
        (all_results["Strict"] == False)
    ].copy()

    strict_results.to_csv(
        f"{RESULTS_DIR}/strict_momentum.csv", index=False
    )

    early_results.to_csv(
        f"{RESULTS_DIR}/early_momentum.csv", index=False
    )

    all_results.to_csv(
        f"{RESULTS_DIR}/latest_all.csv", index=False
    )

    logging.info(
        "Scan completed. Strict: %s | Early: %s",
        len(strict_results), len(early_results)
    )


if __name__ == "__main__":
    main()
