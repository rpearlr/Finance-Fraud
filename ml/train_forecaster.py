"""
Phase 2 — Portfolio Forecaster (Facebook Prophet)
Run from project root: python ml/train_forecaster.py

Outputs:
  data/raw/stock_prices.csv                    — downloaded from Yahoo Finance
  data/raw/sample_portfolio.csv                — generated sample portfolio
  data/staged/stock_forecast.csv               — 30-day Prophet forecast (stocks)
  data/staged/portfolio_forecast.csv           — 30-day Prophet forecast (portfolio)
  ml/models/prophet_stock_{TICKER}.pkl         — saved Prophet model per stock
  ml/models/prophet_portfolio.pkl              — saved Prophet model for portfolio
  ml/reports/forecast_report.txt              — RMSE, MAPE, evaluation summary
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
# Indian blue-chips — matches the dashboard tickers from Phase 1
TICKERS = ["HDFCBANK.NS", "INFY.NS", "TCS.NS", "RELIANCE.NS", "ICICIBANK.NS"]
FORECAST_DAYS  = 30
HISTORY_DAYS   = 365 * 2    # 2 years of history for good seasonality fitting
TRAIN_RATIO    = 0.85       # 85% train, 15% test (for RMSE/MAPE evaluation)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR     = Path("data/raw")
STAGED_DIR  = Path("data/staged")
MODEL_DIR   = Path("ml/models")
REPORT_DIR  = Path("ml/reports")

for d in [RAW_DIR, STAGED_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Step 1: Download stock data ────────────────────────────────────────────────
def download_stock_data() -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    end   = datetime.today()
    start = end - timedelta(days=HISTORY_DAYS)

    log.info(f"Downloading {len(TICKERS)} tickers from Yahoo Finance...")
    log.info(f"Period: {start.date()} to {end.date()}")

    all_frames = []
    for ticker in TICKERS:
        log.info(f"  Fetching {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            log.warning(f"  No data returned for {ticker} — skipping")
            continue
        df = df[["Close"]].copy()
        df.columns = ["close"]
        df["ticker"] = ticker
        df.index.name = "date"
        df = df.reset_index()
        all_frames.append(df)

    if not all_frames:
        raise ValueError("No stock data downloaded. Check your internet connection.")

    combined = pd.concat(all_frames, ignore_index=True)
    out_path = RAW_DIR / "stock_prices.csv"
    combined.to_csv(out_path, index=False)
    log.info(f"Saved stock data -> {out_path} ({len(combined):,} rows)")
    return combined


# ── Step 2: Generate sample portfolio ─────────────────────────────────────────
def generate_sample_portfolio(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a weighted portfolio value series from the downloaded stocks.
    Weights approximate the dashboard allocation: HDFC 18%, INFY 15%,
    TCS 12%, RELIANCE 11%, ICICI 9%, rest in cash (35%).
    Total AUM: 4.82B INR (matches dashboard KPI).
    """
    log.info("Generating sample portfolio from stock weights...")

    WEIGHTS = {
        "HDFCBANK.NS": 0.182,
        "INFY.NS"    : 0.147,
        "TCS.NS"     : 0.123,
        "RELIANCE.NS": 0.108,
        "ICICIBANK.NS": 0.094,
    }
    TOTAL_AUM_INR = 4_820_000_000   # 4.82B INR

    pivoted = stock_df.pivot(index="date", columns="ticker", values="close").dropna()

    # Normalize each stock to its first price so we track % returns
    normalized = pivoted / pivoted.iloc[0]

    # Weighted portfolio index
    portfolio_index = sum(
        normalized[ticker] * weight
        for ticker, weight in WEIGHTS.items()
        if ticker in normalized.columns
    )
    # Add cash component (flat)
    cash_weight = 1 - sum(WEIGHTS.values())
    portfolio_index += cash_weight

    # Scale to AUM in INR
    portfolio_value = (portfolio_index * TOTAL_AUM_INR).round(2)

    portfolio_df = pd.DataFrame({
        "date": portfolio_index.index,
        "portfolio_value_inr": portfolio_value.values
    })

    out_path = RAW_DIR / "sample_portfolio.csv"
    portfolio_df.to_csv(out_path, index=False)
    log.info(f"Saved portfolio -> {out_path} ({len(portfolio_df):,} rows)")
    log.info(f"  Start value : INR {portfolio_df['portfolio_value_inr'].iloc[0]:,.0f}")
    log.info(f"  End value   : INR {portfolio_df['portfolio_value_inr'].iloc[-1]:,.0f}")
    return portfolio_df


# ── Step 3: Train Prophet on a single series ───────────────────────────────────
def train_prophet(df: pd.DataFrame, date_col: str, value_col: str, label: str):
    """
    Fits Prophet, evaluates on held-out test period, returns model + metrics + forecast.
    df must have columns: date_col (datetime), value_col (float).
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Run: pip install prophet")

    # Prophet requires columns named 'ds' and 'y'
    series = df[[date_col, value_col]].rename(
        columns={date_col: "ds", value_col: "y"}
    )
    series["ds"] = pd.to_datetime(series["ds"])
    series = series.dropna().sort_values("ds").reset_index(drop=True)

    # Train / test split
    split_idx  = int(len(series) * TRAIN_RATIO)
    train_df   = series.iloc[:split_idx]
    test_df    = series.iloc[split_idx:]

    log.info(f"  [{label}] Training Prophet — {len(train_df)} train rows, {len(test_df)} test rows")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,    # daily noise isn't meaningful for stock close prices
        changepoint_prior_scale=0.05,   # controls trend flexibility (lower = smoother)
        seasonality_prior_scale=10,
        interval_width=0.95,        # 95% confidence bands
    )
    model.fit(train_df)

    # Evaluate on test set
    test_forecast = model.predict(test_df[["ds"]])
    y_true = test_df["y"].values
    y_pred = test_forecast["yhat"].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    log.info(f"  [{label}] RMSE={rmse:.2f}  MAPE={mape:.2f}%")

    # Refit on full data for the actual forecast
    model_full = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95,
    )
    model_full.fit(series)

    # Generate 30-day future forecast
    future   = model_full.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model_full.predict(future)

    # Tag which rows are historical vs forecast
    last_historical = series["ds"].max()
    forecast["is_forecast"] = forecast["ds"] > last_historical
    forecast["label"] = label

    return model_full, forecast, {"rmse": rmse, "mape": mape, "label": label}


# ── Step 4: Run forecasting for all tickers + portfolio ────────────────────────
def run_stock_forecasts(stock_df: pd.DataFrame):
    all_forecasts = []
    all_metrics   = []
    models        = {}

    for ticker in stock_df["ticker"].unique():
        ticker_df = stock_df[stock_df["ticker"] == ticker][["date", "close"]].copy()
        ticker_df["date"] = pd.to_datetime(ticker_df["date"])

        model, forecast, metrics = train_prophet(
            ticker_df, date_col="date", value_col="close", label=ticker
        )
        forecast["ticker"] = ticker
        all_forecasts.append(forecast)
        all_metrics.append(metrics)
        models[ticker] = model

        # Save individual model
        model_path = MODEL_DIR / f"prophet_stock_{ticker.replace('.', '_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Saved model -> {model_path}")

    combined_forecast = pd.concat(all_forecasts, ignore_index=True)
    out_path = STAGED_DIR / "stock_forecast.csv"
    combined_forecast[["ds", "ticker", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]]\
        .to_csv(out_path, index=False)
    log.info(f"Saved stock forecast -> {out_path}")

    return models, all_metrics


def run_portfolio_forecast(portfolio_df: pd.DataFrame):
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])

    model, forecast, metrics = train_prophet(
        portfolio_df,
        date_col="date",
        value_col="portfolio_value_inr",
        label="portfolio"
    )

    model_path = MODEL_DIR / "prophet_portfolio.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Saved portfolio model -> {model_path}")

    out_path = STAGED_DIR / "portfolio_forecast.csv"
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]]\
        .to_csv(out_path, index=False)
    log.info(f"Saved portfolio forecast -> {out_path}")

    return model, metrics


# ── Step 5: Write evaluation report ───────────────────────────────────────────
def write_report(stock_metrics: list, portfolio_metrics: dict) -> None:
    lines = [
        "=" * 60,
        "PROPHET FORECASTING EVALUATION REPORT",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Horizon    : {FORECAST_DAYS} days",
        f"Train split: {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test",
        "=" * 60,
        "",
        "-- Stock Price Forecasts --",
        f"  {'Ticker':<20} {'RMSE':>10} {'MAPE':>10}",
        f"  {'-'*42}",
    ]

    for m in stock_metrics:
        lines.append(f"  {m['label']:<20} {m['rmse']:>10.2f} {m['mape']:>9.2f}%")

    avg_mape = np.mean([m["mape"] for m in stock_metrics])
    lines += [
        f"  {'-'*42}",
        f"  {'Average':<20} {'':>10} {avg_mape:>9.2f}%",
        "",
        "-- Portfolio Value Forecast --",
        f"  RMSE : {portfolio_metrics['rmse']:,.2f} INR",
        f"  MAPE : {portfolio_metrics['mape']:.2f}%",
        "",
        "-- Output Files --",
        f"  data/staged/stock_forecast.csv",
        f"  data/staged/portfolio_forecast.csv",
        f"  ml/models/prophet_stock_*.pkl",
        f"  ml/models/prophet_portfolio.pkl",
        "",
        "-- What these numbers mean --",
        "  MAPE < 5%  : Excellent forecast accuracy",
        "  MAPE 5-10% : Good — acceptable for 30-day financial forecast",
        "  MAPE > 10% : High uncertainty — expected for volatile stocks",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    report_path = REPORT_DIR / "forecast_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    log.info(f"Report saved -> {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    log.info("---  Phase 2: Prophet Portfolio Forecaster  ---")

    # Install check
    missing = []
    try: import yfinance
    except ImportError: missing.append("yfinance")
    try: from prophet import Prophet
    except ImportError: missing.append("prophet")

    if missing:
        log.error(f"Missing packages: {', '.join(missing)}")
        log.error(f"Run: pip install {' '.join(missing)}")
        return

    stock_df      = download_stock_data()
    portfolio_df  = generate_sample_portfolio(stock_df)

    log.info("Training stock forecasts (this takes 1-2 min)...")
    _, stock_metrics = run_stock_forecasts(stock_df)

    log.info("Training portfolio forecast...")
    _, portfolio_metrics = run_portfolio_forecast(portfolio_df)

    write_report(stock_metrics, portfolio_metrics)

    log.info("---  Done. Ready for Phase 3 (FastAPI /forecast endpoint).  ---")


if __name__ == "__main__":
    run()