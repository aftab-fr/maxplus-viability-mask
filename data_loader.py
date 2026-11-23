"""
HourlyDataset — loads the merged_hourly_YYYY.csv produced by fetch_data.py
and serves 24-slot daily profiles to the RL environment.
"""

from pathlib import Path
import numpy as np
import pandas as pd


class HourlyDataset:
    """
    Wraps a merged_hourly_<year>.csv file.
    Provides (solar_kw[24], price_eur_kwh[24]) profiles indexed by calendar date.

    DST-transition days (23 or 25 h) are silently skipped — they account for
    ~2 days per year and would break the fixed-24-slot episode assumption.
    """

    def __init__(self, csv_path="data/merged_hourly_2019.csv", verbose=True):
        paths = [Path(p) for p in (csv_path if isinstance(csv_path, (list, tuple))
                                   else [csv_path])]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run  python fetch_data.py --year <YEAR> --real-prices  first."
                )
        csv_path = ", ".join(str(p) for p in paths)

        frames = []
        for p in paths:
            f = pd.read_csv(p)
            year = int(p.stem.split("_")[-1])
            f["date"] = pd.to_datetime(f["date"]).dt.date
            # keep only local dates that belong to the file's own year
            frames.append(f[[d.year == year for d in f["date"]]])
        df = pd.concat(frames, ignore_index=True)

        self._days: dict = {}
        skipped = 0
        for date, group in df.groupby("date"):
            g = group.sort_values("hour").reset_index(drop=True)
            if len(g) != 24:
                skipped += 1
                continue
            self._days[date] = (
                g["solar_kw"].to_numpy(dtype=np.float32),
                g["price_eur_kwh"].to_numpy(dtype=np.float32),
            )

        if not self._days:
            raise ValueError(f"No complete 24-h days found in {csv_path}.")

        self.dates = sorted(self._days.keys())
        if verbose:
            print(f"[HourlyDataset] {len(self.dates)} complete days loaded "
                  f"from {csv_path}  ({skipped} incomplete/DST days skipped)")

    # ── public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.dates)

    def get_day(self, date) -> tuple[np.ndarray, np.ndarray]:
        """Return (solar_kw[24], price_eur_kwh[24]) for a specific date."""
        return self._days[date]

    def random_day(self, rng: np.random.Generator | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Return a uniformly random day's profiles (used during RL training)."""
        if rng is None:
            rng = np.random.default_rng()
        date = self.dates[rng.integers(len(self.dates))]
        return self._days[date]

    def seasonal_summary(self) -> pd.DataFrame:
        """
        DataFrame with per-day statistics — useful for paper figures.
        Columns: date, month, solar_peak_kw, solar_daily_kwh,
                 price_mean_eur_mwh, price_peak_eur_mwh
        """
        rows = []
        for date, (sol, pri) in self._days.items():
            rows.append({
                "date":               date,
                "month":              date.month,
                "solar_peak_kw":      float(sol.max()),
                "solar_daily_kwh":    float(sol.sum()),      # 1 h intervals
                "price_mean_eur_mwh": float(pri.mean()) * 1000,
                "price_peak_eur_mwh": float(pri.max())  * 1000,
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def monthly_avg_profile(self, month: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the average hourly solar and price profile for a given month (1-12).
        Useful for illustrative figures in the paper.
        """
        days = [d for d in self.dates if d.month == month]
        if not days:
            raise ValueError(f"No data for month {month}")
        solar_stack = np.stack([self._days[d][0] for d in days])
        price_stack = np.stack([self._days[d][1] for d in days])
        return solar_stack.mean(axis=0), price_stack.mean(axis=0)
