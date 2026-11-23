"""
Real day-ahead PV forecast errors (revision: R2-M5, R4-4).

Downloads, for France, the TSO day-ahead solar generation forecast and the
actual solar generation (Energy-Charts, Fraunhofer ISE, CC BY 4.0) and
stores the hourly ratio  r(t) = forecast(t) / actual(t)  on local-time days.

The ratio is applied to the site's PVGIS output in `real_forecast` as
    S_hat(t) = clip( S(t) * r(t), 0, P_peak ),
i.e. the site forecast inherits the *empirical* relative day-ahead error of
a real operational forecast, with its true diurnal, day-to-day and seasonal
structure, instead of the synthetic AR(1) error. Hours with small actual
national generation (< 5% of the daily maximum) have ill-conditioned ratios
and are given the ratio of the nearest well-conditioned hour.

National aggregation smooths errors spatially, so these errors are smaller
than those of a single-site forecast; the synthetic sigma = 0.3 case is kept
as the pessimistic scenario.

Output: data/fr_solar_forecast_<year>.csv  (date, hour, forecast_mw, actual_mw, ratio)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL_FC = "https://api.energy-charts.info/public_power_forecast"
URL_ACT = "https://api.energy-charts.info/public_power"
DATA = Path("data")


def _get(url, params, tries=8):
    import time
    for k in range(tries):
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 429:
            time.sleep(30 * (k + 1))
            continue
        r.raise_for_status()
        time.sleep(3)
        return r.json()
    r.raise_for_status()


def fetch_year(year):
    fc, act = [], []
    for mo in range(1, 13):
        s = f"{year}-{mo:02d}-01T00:00Z"
        e = f"{year + (mo == 12)}-{(mo % 12) + 1:02d}-01T00:00Z"
        j = _get(URL_FC, dict(country="fr", production_type="solar",
                              forecast_type="day-ahead", start=s, end=e))
        fc.append(pd.Series(j["forecast_values"],
                            index=pd.to_datetime(j["unix_seconds"], unit="s", utc=True)))
        j = _get(URL_ACT, dict(country="fr", start=s, end=e))
        sol = [p for p in j["production_types"] if p["name"] == "Solar"][0]
        act.append(pd.Series(sol["data"],
                             index=pd.to_datetime(j["unix_seconds"], unit="s", utc=True)))
    f = pd.concat(fc).astype(float).resample("h").mean()
    a = pd.concat(act).astype(float).resample("h").mean()
    df = pd.DataFrame(dict(forecast_mw=f, actual_mw=a)).dropna()
    df = df[~df.index.duplicated()]
    loc = df.index.tz_convert("Europe/Paris")
    df["date"] = loc.strftime("%Y-%m-%d")
    df["hour"] = loc.hour
    df = df[loc.year == year]
    out = []
    for d, g in df.groupby("date"):
        if len(g) != 24:
            continue
        g = g.sort_values("hour").copy()
        amax = g.actual_mw.max()
        ok = g.actual_mw > 0.05 * max(amax, 1.0)
        r = np.where(ok, g.forecast_mw / g.actual_mw.clip(lower=1e-6), np.nan)
        if np.isfinite(r).any():
            s = pd.Series(r).ffill().bfill().to_numpy()
        else:
            s = np.ones(24)
        g["ratio"] = np.clip(s, 0.0, 5.0)
        out.append(g)
    res = pd.concat(out)[["date", "hour", "forecast_mw", "actual_mw", "ratio"]]
    res.to_csv(DATA / f"fr_solar_forecast_{year}.csv", index=False)
    print(year, "days", res.date.nunique())
    return res


_CACHE = {}


def real_forecast(solar, year, date, p_peak=2000.0):
    """Site forecast with the empirical day-ahead error of the given day."""
    if year not in _CACHE:
        d = pd.read_csv(DATA / f"fr_solar_forecast_{year}.csv")
        _CACHE[year] = {k: g.sort_values("hour").ratio.to_numpy()
                        for k, g in d.groupby("date")}
    r = _CACHE[year].get(str(date))
    if r is None:
        return None
    return np.clip(np.asarray(solar, float) * r, 0.0, p_peak)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2018, 2019, 2020, 2021, 2022, 2023])
    for y in ap.parse_args().years:
        fetch_year(y)
