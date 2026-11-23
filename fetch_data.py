"""
Data acquisition for the Belfort factory microgrid paper.

Sources
-------
Solar  : PVGIS hourly PV-production API (European Commission JRC, free, no key)
         https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html
Prices : Three options (pick one):
  1. Energy Charts — Fraunhofer ISE (recommended, no registration, CC BY 4.0)
             https://www.energy-charts.info  |  source: BNetzA / SMARD.de
             Use: python fetch_data.py --year 2019 --real-prices
  2. ENTSO-E Transparency Platform (standard in European research papers)
             https://transparency.entsoe.eu  →  My Account → Security Token
             Use: export ENTSOE_API_KEY=<token> && python fetch_data.py --year 2020
  3. --mock-prices  Synthetic duck-curve prices (pipeline testing only)

Note on RTE v3: operational (real-time) API — returns today's prices only.

Setup
-----
  pip install requests pandas entsoe-py

Usage
-----
  python fetch_data.py --year 2019 --real-prices    # real prices, no registration
  python fetch_data.py --year 2020 --solar-only    # solar only, skip prices
  python fetch_data.py --year 2020 --mock-prices   # synthetic prices (pipeline test)
"""

import argparse
import base64
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── site constants ────────────────────────────────────────────────────────────
SITE_NAME       = "Belfort, France"
SITE_LAT        = 47.637
SITE_LON        = 6.863
SITE_PEAK_KWP   = 2000.0   # 2 MWp rooftop PV array
SYSTEM_LOSS_PCT = 14.0     # PVGIS default system loss (%)

DATA_DIR = Path("data")

# ── ENTSO-E ───────────────────────────────────────────────────────────────────
ENTSOE_AREA_FR = "10YFR-RTE------C"   # France bidding zone EIC code

# ── Energy Charts (Fraunhofer ISE) ────────────────────────────────────────────
# Hourly day-ahead prices for any European bidding zone, no registration.
# License: CC BY 4.0, source: BNetzA | SMARD.de via Fraunhofer ISE.
ENERGY_CHARTS_URL = "https://api.energy-charts.info/price"
ENERGY_CHARTS_BZN = "FR"   # France bidding zone

# ── RTE v3 (operational / today only — kept for completeness) ─────────────────
RTE_TOKEN_URL  = "https://digital.iservices.rte-france.com/token/oauth/"
RTE_PRICES_URL = (
    "https://digital.iservices.rte-france.com"
    "/open_api/wholesale_market/v3/france_power_exchanges"
)


# ─────────────────────────────────────────────────────────────────────────────
#  PVGIS solar production
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pvgis(year: int) -> pd.DataFrame:
    """
    Fetch hourly PV production for the Belfort site from PVGIS-SARAH3 (PVGIS 5.3).

    Returns
    -------
    DataFrame with columns:
        timestamp_utc  (tz-aware, UTC)
        solar_kw       (kW, clipped ≥ 0)
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat":               SITE_LAT,
        "lon":               SITE_LON,
        "raddatabase":       "PVGIS-SARAH3",  # hourly satellite irradiance, 2005-2023
        "pvcalculation":     1,
        "peakpower":         SITE_PEAK_KWP,   # kWp  (P output will be in W)
        "loss":              SYSTEM_LOSS_PCT,
        "pvtechchoice":      "crystSi",
        "mountingplace":     "free",
        "angle":             35,              # optimal tilt for lat ~47.6°
        "aspect":            0,               # south-facing
        "startyear":         year,
        "endyear":           year,
        "outputformat":      "json",
        "optimalinclination": 0,
        "optimalangles":     0,
    }

    print(f"  Querying PVGIS for {SITE_NAME} ({SITE_LAT}°N, {SITE_LON}°E), "
          f"{SITE_PEAK_KWP:.0f} kWp, year {year} …")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()

    records = r.json()["outputs"]["hourly"]
    df = pd.DataFrame(records)

    # PVGIS time: "YYYYMMDD:HHMM" in UTC, minutes are always :10 past the hour.
    # Floor to the hour so we can align with RTE prices (which are on the :00).
    df["timestamp_utc"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True)
    df["timestamp_utc"] = df["timestamp_utc"].dt.floor("h")
    # P is in W for the full system; divide by 1000 to get kW
    df["solar_kw"] = (df["P"] / 1000.0).clip(lower=0.0).astype(np.float32)
    df = df[["timestamp_utc", "solar_kw"]].sort_values("timestamp_utc").reset_index(drop=True)

    n_nonzero = (df["solar_kw"] > 0).sum()
    print(f"  → {len(df)} hourly records, "
          f"peak {df['solar_kw'].max():.0f} kW, "
          f"{n_nonzero} sunlight hours")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  RTE éCO2mix day-ahead prices
# ─────────────────────────────────────────────────────────────────────────────

def _rte_token(client_id: str, client_secret: str) -> str:
    """Obtain a short-lived OAuth2 bearer token."""
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        RTE_TOKEN_URL,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _rte_fetch_month(token: str, year: int, month: int) -> list[dict]:
    """Fetch one month of day-ahead prices; return list of {timestamp_utc, price_eur_kwh}."""
    # RTE API uses ISO-8601 with timezone offset (CET = +01:00)
    start = f"{year}-{month:02d}-01T00:00:00+01:00"
    next_month = month % 12 + 1
    next_year  = year + (1 if month == 12 else 0)
    end   = f"{next_year}-{next_month:02d}-01T00:00:00+01:00"

    r = requests.get(
        RTE_PRICES_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={"start_date": start, "end_date": end, "type": "day_ahead"},
        timeout=60,
    )
    r.raise_for_status()

    rows = []
    for block in r.json().get("france_power_exchanges", []):
        for v in block.get("values", []):
            ts = pd.Timestamp(v["start_date"]).tz_convert("UTC")
            # RTE prices are in EUR/MWh; convert to EUR/kWh for the env
            rows.append({"timestamp_utc": ts, "price_eur_kwh": v["value"] / 1000.0})
    return rows


def fetch_rte_prices(year: int, client_id: str, client_secret: str) -> pd.DataFrame:
    """
    Fetch full-year hourly day-ahead prices from RTE éCO2mix (France).

    Returns
    -------
    DataFrame with columns:
        timestamp_utc   (tz-aware, UTC)
        price_eur_kwh   (EUR/kWh, converted from EUR/MWh)
    """
    print(f"  Authenticating with RTE API …")
    token = _rte_token(client_id, client_secret)

    all_rows: list[dict] = []
    for month in range(1, 13):
        print(f"  Fetching RTE prices {year}-{month:02d} …", end=" ", flush=True)
        try:
            rows = _rte_fetch_month(token, year, month)
        except requests.HTTPError as e:
            if e.response.status_code == 401:   # token expired mid-run
                token = _rte_token(client_id, client_secret)
                rows = _rte_fetch_month(token, year, month)
            else:
                raise
        all_rows.extend(rows)
        print(f"{len(rows)} values")
        time.sleep(0.4)  # respect API rate limit

    df = (pd.DataFrame(all_rows)
            .drop_duplicates("timestamp_utc")
            .sort_values("timestamp_utc")
            .reset_index(drop=True))
    df["price_eur_kwh"] = df["price_eur_kwh"].astype(np.float32)

    mean_mwh = df["price_eur_kwh"].mean() * 1000
    print(f"  → {len(df)} records, mean {mean_mwh:.1f} EUR/MWh, "
          f"min {df['price_eur_kwh'].min()*1000:.1f}, "
          f"max {df['price_eur_kwh'].max()*1000:.1f} EUR/MWh")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ENTSO-E historical day-ahead prices (primary source for the paper)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_entsoe_prices(year: int, api_key: str) -> pd.DataFrame:
    """
    Fetch full-year French day-ahead prices from ENTSO-E Transparency Platform.

    Registration: https://transparency.entsoe.eu → My Account → Security Token
    (free, takes ~2 minutes)

    Returns
    -------
    DataFrame with columns:
        timestamp_utc   (tz-aware, UTC, hourly)
        price_eur_kwh   (EUR/kWh, converted from EUR/MWh)
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        raise ImportError("Run:  pip install entsoe-py")

    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Paris")
    end   = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Paris")

    print(f"  Querying ENTSO-E for France day-ahead prices {year} …")
    prices_mwh = client.query_day_ahead_prices(ENTSOE_AREA_FR, start=start, end=end)

    # entsoe-py returns a Series indexed by local time; resample to UTC hourly
    prices_utc = (prices_mwh
                  .resample("h")
                  .mean()
                  .tz_convert("UTC"))
    # keep only the target year
    prices_utc = prices_utc[
        (prices_utc.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
        (prices_utc.index <  pd.Timestamp(f"{year+1}-01-01", tz="UTC"))
    ]

    df = pd.DataFrame({
        "timestamp_utc": prices_utc.index,
        "price_eur_kwh": (prices_utc.values / 1000.0).astype(np.float32),
    }).reset_index(drop=True)

    mean_mwh = df["price_eur_kwh"].mean() * 1000
    print(f"  → {len(df)} hourly records, mean {mean_mwh:.1f} EUR/MWh, "
          f"min {df['price_eur_kwh'].min()*1000:.1f}, "
          f"max {df['price_eur_kwh'].max()*1000:.1f} EUR/MWh")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Energy Charts (Fraunhofer ISE) — French day-ahead prices, no registration
# ─────────────────────────────────────────────────────────────────────────────

def fetch_energycharts_prices(year: int) -> pd.DataFrame:
    """
    Fetch French day-ahead prices from the Fraunhofer ISE Energy Charts API.

    No API key required. License: CC BY 4.0.
    Source: BNetzA | SMARD.de via https://www.energy-charts.info

    Cite as:
        Fraunhofer ISE (2024). Energy Charts.
        https://www.energy-charts.info

    Returns
    -------
    DataFrame with columns:
        timestamp_utc   (tz-aware, UTC, hourly)
        price_eur_kwh   (EUR/kWh, converted from EUR/MWh)
    """
    print(f"  Querying Energy Charts (Fraunhofer ISE) for FR day-ahead prices {year} …")
    r = requests.get(
        ENERGY_CHARTS_URL,
        params={"bzn": ENERGY_CHARTS_BZN, "start": f"{year - 1}-12-31", "end": f"{year + 1}-01-01"},
        timeout=120,
    )
    r.raise_for_status()
    payload = r.json()

    timestamps = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    prices_mwh = np.array(
        [float(p) if p is not None else np.nan for p in payload["price"]],
        dtype=np.float32,
    )

    df = pd.DataFrame({"timestamp_utc": timestamps, "price_eur_kwh": prices_mwh / 1000.0})
    df = df.dropna(subset=["price_eur_kwh"])
    # hourly mean (guards against sub-hourly products) and drop duplicates
    df = (df.set_index("timestamp_utc").resample("h").mean()
            .dropna().reset_index())
    df["price_eur_kwh"] = df["price_eur_kwh"].astype(np.float32)

    mean_mwh = df["price_eur_kwh"].mean() * 1000
    print(f"  → {len(df)} hourly records, mean {mean_mwh:.1f} EUR/MWh, "
          f"min {df['price_eur_kwh'].min()*1000:.1f}, "
          f"max {df['price_eur_kwh'].max()*1000:.1f} EUR/MWh")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic price fallback (for pipeline testing before ENTSO-E token)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_prices(timestamps_utc: pd.Series) -> pd.DataFrame:
    """
    Generate synthetic French day-ahead prices calibrated to 2020 RTE levels.
    Duck-curve shape (cheap midday, expensive morning/evening) with seasonal
    scaling (winter ~20% higher than summer) and mild day-to-day noise.

    Returns DataFrame with columns: timestamp_utc, price_eur_kwh
    """
    rng = np.random.default_rng(0)
    ts_cet = timestamps_utc.dt.tz_convert("Europe/Paris")
    hours  = ts_cet.dt.hour.to_numpy()
    months = ts_cet.dt.month.to_numpy()

    # Hourly shape in EUR/MWh (duck curve, loosely calibrated to FR 2020 ~40 EUR/MWh avg)
    hourly_shape = np.array([
        45, 42, 40, 38, 39, 44, 55, 65,   # 0-7h
        58, 48, 35, 27, 22, 20, 22, 28,   # 8-15h  (midday solar dip)
        38, 52, 68, 75, 70, 62, 55, 50,   # 16-23h (evening peak)
    ], dtype=float)

    # Seasonal multiplier: winter peak demand, summer cheap nuclear+solar
    seasonal = np.array([1.25, 1.20, 1.10, 1.00, 0.95, 0.90,
                         0.88, 0.90, 0.95, 1.05, 1.15, 1.30])

    base      = hourly_shape[hours] * seasonal[months - 1]
    noise     = rng.normal(1.0, 0.08, size=len(base)).clip(0.5, 1.5)
    price_mwh = (base * noise).clip(min=0.0)

    return pd.DataFrame({
        "timestamp_utc":  timestamps_utc.reset_index(drop=True),
        "price_eur_kwh":  (price_mwh / 1000.0).astype(np.float32),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Merge and save
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_save(solar: pd.DataFrame, prices: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Inner-join solar and price on UTC hour, add CET local-time columns, save to CSV.

    Output columns:
        timestamp_utc, timestamp_cet, date (YYYY-MM-DD), hour (0-23 CET),
        solar_kw, price_eur_kwh
    """
    merged = pd.merge(
        solar.rename(columns={"timestamp_utc": "ts"}),
        prices.rename(columns={"timestamp_utc": "ts"}),
        on="ts", how="inner",
    )
    merged["timestamp_utc"] = merged["ts"]
    merged["timestamp_cet"] = merged["ts"].dt.tz_convert("Europe/Paris")
    merged["date"] = merged["timestamp_cet"].dt.strftime("%Y-%m-%d")
    merged["hour"] = merged["timestamp_cet"].dt.hour
    merged = (merged
              .drop(columns="ts")
              [["timestamp_utc", "timestamp_cet", "date", "hour",
                "solar_kw", "price_eur_kwh"]]
              .sort_values("timestamp_utc")
              .reset_index(drop=True))

    # Report DST transition days (23 or 25 hours — excluded by HourlyDataset)
    day_counts = merged.groupby("date")["hour"].count()
    incomplete = day_counts[day_counts != 24]
    if not incomplete.empty:
        print(f"  Note: {len(incomplete)} DST-transition day(s) have ≠24 hours "
              f"({list(incomplete.index)}) — they will be skipped by the data loader.")

    path = DATA_DIR / f"merged_hourly_{year}.csv"
    merged.to_csv(path, index=False)
    n_complete = (day_counts == 24).sum()
    print(f"\n  Saved: {path}  "
          f"({len(merged)} rows, {n_complete} complete 24-h days)")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch PVGIS solar + ENTSO-E prices for the Belfort microgrid paper"
    )
    parser.add_argument("--year", type=int, default=2019,
                        help="Calendar year to fetch (default: 2019; "
                             "PVGIS-SARAH3 covers 2005-2023; "
                             "OPSD covers 2015-2021)")
    parser.add_argument("--solar-only", action="store_true",
                        help="Fetch only PVGIS solar data; skip prices")
    parser.add_argument("--real-prices", action="store_true",
                        help="Download prices from Fraunhofer ISE Energy Charts "
                             "(no registration; CC BY 4.0; any year 2015-present)")
    parser.add_argument("--mock-prices", action="store_true",
                        help="Generate synthetic French duck-curve prices "
                             "(pipeline test only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached CSV already exists")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    print(f"\n=== Data acquisition for {SITE_NAME}, year {args.year} ===\n")

    # ── solar ─────────────────────────────────────────────────────────────────
    solar_path = DATA_DIR / f"pvgis_belfort_{args.year}.csv"
    if solar_path.exists() and not args.force:
        print(f"  Cache hit: {solar_path}  (use --force to re-download)")
        solar = pd.read_csv(solar_path)
        solar["timestamp_utc"] = pd.to_datetime(solar["timestamp_utc"], utc=True)
    else:
        solar = fetch_pvgis(args.year)
        solar.to_csv(solar_path, index=False)
        print(f"  Saved: {solar_path}")

    if args.solar_only:
        print("\nDone (solar only). Add ENTSO-E token and re-run for prices.\n")
        return

    # ── prices ────────────────────────────────────────────────────────────────
    prices_path = DATA_DIR / f"prices_{args.year}.csv"

    if args.mock_prices:
        print("  Generating synthetic French duck-curve prices (--mock-prices) …")
        prices = _mock_prices(solar["timestamp_utc"])
        prices.to_csv(prices_path, index=False)
        print(f"  Saved: {prices_path}")
        print("  NOTE: Replace with real prices before final paper experiments.")

    elif args.real_prices:
        if prices_path.exists() and not args.force:
            print(f"  Cache hit: {prices_path}  (use --force to re-download)")
            prices = pd.read_csv(prices_path)
            prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
        else:
            prices = fetch_energycharts_prices(args.year)
            prices.to_csv(prices_path, index=False)
            print(f"  Saved: {prices_path}")

    else:
        api_key = os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            print(
                "\n[ERROR] No price source specified. Options:\n"
                "  --opsd-prices        Real prices, no registration needed\n"
                "  --mock-prices        Synthetic prices (pipeline test only)\n"
                "  export ENTSOE_API_KEY=<token> then re-run  (ENTSO-E)\n"
            )
            raise SystemExit(1)

        if prices_path.exists() and not args.force:
            print(f"  Cache hit: {prices_path}  (use --force to re-download)")
            prices = pd.read_csv(prices_path)
            prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
        else:
            prices = fetch_entsoe_prices(args.year, api_key)
            prices.to_csv(prices_path, index=False)
            print(f"  Saved: {prices_path}")

    # ── merge ─────────────────────────────────────────────────────────────────
    merge_and_save(solar, prices, args.year)
    print("\nNext step:  python train.py --real-data\n")


if __name__ == "__main__":
    main()
