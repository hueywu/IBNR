"""
completion_factors.py

Builds health insurance completion factors from a claim-level extract using
the chain-ladder method. See README.md for full documentation.

Required CSV columns: ClaimAmount, DOS_YYYYMM, DOP_YYYYMM

Usage:
    python completion_factors.py <input_file> [options]

Options:
    --lookback N        Most recent service months per age-to-age factor (default 12, range 6-24)
    --exclude-high N    Highest-ratio outliers to drop per window (default 1, range 0-3)
    --exclude-low N     Lowest-ratio outliers to drop per window (default 1, range 0-3)
    --min-periods N     Min ratios after trimming to compute average; else 1.0 (default 6)
    --max-lag N         Lag assumed fully developed; defaults to max observed lag in data
    --output FILE       Write completion factor table to CSV
"""

import argparse
import sys

import numpy as np
import pandas as pd


def load_claims(filepath: str) -> pd.DataFrame:
    """Load the raw claims CSV and keep only the columns we need."""
    df = pd.read_csv(filepath)

    required_cols = {"ClaimAmount", "DOS_YYYYMM", "DOP_YYYYMM"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    return df[["ClaimAmount", "DOS_YYYYMM", "DOP_YYYYMM"]].copy()


def yyyymm_to_period(series: pd.Series) -> pd.Series:
    """Convert an int/str column like 202403 into a pandas Period (monthly)."""
    return pd.to_datetime(series.astype(str), format="%Y%m").dt.to_period("M")


def add_lag_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'lag' column = number of months between DOS and DOP."""
    dos = yyyymm_to_period(df["DOS_YYYYMM"])
    dop = yyyymm_to_period(df["DOP_YYYYMM"])

    # Subtracting two monthly Periods gives an integer number of months.
    df = df.copy()
    df["DOS_period"] = dos
    df["lag"] = (dop - dos).apply(lambda offset: offset.n)

    # Negative lags shouldn't exist (you can't pay before service occurs),
    # but real-world data is messy -- treat them as lag 0 (paid in the
    # same month as service) rather than dropping them.
    bad = df["lag"] < 0
    if bad.any():
        print(f"Warning: {bad.sum()} claims have a payment date before their "
              f"service date -- reassigning to lag 0.", file=sys.stderr)
        df.loc[bad, "lag"] = 0

    return df


def build_incremental_triangle(df: pd.DataFrame, max_dop_period) -> pd.DataFrame:
    """
    Build a triangle of INCREMENTAL paid amounts, sized to reflect what's
    actually OBSERVABLE for each service month -- not just the lags where
    claims happen to exist.

    The number of lag rows is determined by the OLDEST service month's
    observable window: (max_dop_period - earliest DOS_period). For
    example, if the oldest service month is 2022-07 and the most recent
    payment data is from 2025-09, that's a 38-month window, so the
    triangle will have rows for lag 0 through lag 38 -- even if no claims
    in the data actually take 38 months to pay. Rows/cells with no claims
    are filled with 0.0 (a real, observed "$0 paid" -- as opposed to the
    "not yet observable" N/A cells handled by mask_unobservable_cells()).

    Columns = service month (DOS_period)
    Rows    = lag (0, 1, 2, ... max observable lag for the oldest month)
    Values  = total $ paid for claims with that DOS and that lag
    """
    earliest_month = df["DOS_period"].min()
    max_lag = (max_dop_period - earliest_month).n

    triangle = df.pivot_table(
        index="lag",
        columns="DOS_period",
        values="ClaimAmount",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Make sure every lag row 0..max_lag exists, even if some have no data.
    for lag in range(max_lag + 1):
        if lag not in triangle.index:
            triangle.loc[lag] = 0.0

    triangle = triangle.sort_index(axis=0).sort_index(axis=1)
    return triangle


def build_cumulative_triangle(incremental: pd.DataFrame) -> pd.DataFrame:
    """Cumulative paid-to-date = running sum down the rows (across lag)."""
    return incremental.cumsum(axis=0)


def mask_unobservable_cells(triangle: pd.DataFrame, max_dop_period) -> pd.DataFrame:
    """
    Replace cells that represent data we genuinely don't have yet with NaN
    ("N/A"), as opposed to data we have and that happens to be zero.

    A service month `m` can only have observed payments at lag `L` if
    `m + L <= max_dop_period` (the most recent payment period present in
    the dataset). For example, if the latest paid date in the file is
    Dec 2024, then for service month Nov 2024 we only have lag 0 and
    lag 1 -- lag 2 and beyond are unknown (not zero), so they're shown
    as N/A rather than as real values.

    This is purely for DISPLAY / interpretation. The underlying numeric
    triangle (used to compute age-to-age factors) is left untouched --
    compute_age_to_age_factors() already restricts itself to observable
    cells via the same (month + lag <= max_dop_period) test.
    """
    masked = triangle.astype(float).copy()
    for lag in masked.index:
        for month in masked.columns:
            if (month + lag) > max_dop_period:
                masked.loc[lag, month] = np.nan
    return masked


def compute_age_to_age_factors(
    cumulative: pd.DataFrame,
    max_dop_period,
    lookback: int,
    exclude_high: int,
    exclude_low: int,
    min_periods: int = 6,
) -> pd.DataFrame:
    """
    For each lag transition (lag -> lag+1):

      1. Compute one age-to-age ratio per service month:
             ratio = cumulative[lag+1, month] / cumulative[lag, month]
         restricted to service months where lag+1 is actually observable
         (month + (lag+1) <= max_dop_period).

      2. Keep only the most recent `lookback` service months among those
         (`lookback` is adjustable -- typical range 6 to 24).

      3. Drop the `exclude_high` largest and `exclude_low` smallest ratios
         (each adjustable -- typical range 0 to 3).

      4. If fewer than `min_periods` ratios remain after trimming, there
         isn't enough data to trust an average -- assume NO further
         development for this transition (age_to_age_factor = 1.0).
         Otherwise, average what's left -> the age-to-age factor for this
         transition.

    Returns a DataFrame indexed by the *starting* lag of each transition
    (e.g. row 0 is the lag0 -> lag1 factor), with columns:
        n_window           -- how many service months were in the lookback window
        n_used             -- how many ratios remained after trimming
        age_to_age_factor
        assumed_default    -- True if age_to_age_factor was set to 1.0 because
                              n_used < min_periods
    """
    lags = sorted(cumulative.index)
    months = sorted(cumulative.columns)  # oldest -> newest

    results = {}

    for lag in lags[:-1]:
        next_lag = lag + 1

        # Service months for which lag+1 is observable (not in the future).
        usable_months = [m for m in months if (m + next_lag) <= max_dop_period]

        # Most recent `lookback` of those (selection based on recency).
        window_months = usable_months[-lookback:] if usable_months else []

        ratios = []
        for m in window_months:
            denom = cumulative.loc[lag, m]
            numer = cumulative.loc[next_lag, m]
            if denom:
                ratios.append(numer / denom)

        n_window = len(ratios)

        if n_window == 0:
            results[lag] = {
                "n_window": 0, "n_used": 0,
                "age_to_age_factor": 1.0, "assumed_default": True,
            }
            continue

        ratios_sorted = sorted(ratios)

        # Don't drop more than leaves at least one observation.
        max_drop = max(0, n_window - 1)
        eh = min(exclude_high, max_drop)
        el = min(exclude_low, max_drop - eh)

        trimmed = ratios_sorted[el: n_window - eh] if (eh + el) < n_window else ratios_sorted
        n_used = len(trimmed)

        if n_used < min_periods:
            # Not enough data left to trust an average -- assume no
            # further development for this transition.
            results[lag] = {
                "n_window": n_window, "n_used": n_used,
                "age_to_age_factor": 1.0, "assumed_default": True,
            }
        else:
            results[lag] = {
                "n_window": n_window, "n_used": n_used,
                "age_to_age_factor": float(np.mean(trimmed)), "assumed_default": False,
            }

    out = pd.DataFrame.from_dict(results, orient="index")
    out.index.name = "lag"
    return out


def compute_age_to_ultimate(age_to_age: pd.Series, max_lag: int) -> pd.DataFrame:
    """
    Chain the age-to-age factors together, from the oldest lag to the most
    recent, to build the cumulative "age-to-ultimate" factor at each lag.

        age_to_ultimate[max_lag] = 1.0  (assumed fully developed -- the
                                          "tail" assumption)
        age_to_ultimate[lag] = age_to_age[lag] * age_to_ultimate[lag + 1]

    The completion factor at a given lag is then:

        completion_factor[lag] = 1 / age_to_ultimate[lag]
    """
    age_to_ultimate = {max_lag: 1.0}

    # Walk backwards from the tail so each step has the next lag's
    # age-to-ultimate factor already computed -- this is the chaining
    # ("accumulate from oldest to most recent") step.
    for lag in range(max_lag - 1, -1, -1):
        link = age_to_age.get(lag, np.nan)
        if np.isnan(link):
            # No data to estimate this transition -- assume no further
            # development at this step.
            link = 1.0
        age_to_ultimate[lag] = link * age_to_ultimate[lag + 1]

    atu_series = pd.Series(age_to_ultimate).sort_index()
    completion = 1.0 / atu_series

    result = pd.DataFrame({
        "age_to_ultimate_factor": atu_series,
        "completion_factor": completion,
    })
    result.index.name = "lag_months"
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute completion factors from a claims extract.")
    parser.add_argument("input_file", help="Path to the claims CSV")
    parser.add_argument("--max-lag", type=int, default=None,
                         help="Lag (in months) at which claims are assumed to be fully "
                              "developed (completion factor = 1.0). Defaults to the "
                              "maximum lag actually observed in the data. The "
                              "development triangle itself always uses ALL available "
                              "data regardless of this setting.")
    parser.add_argument("--lookback", type=int, default=12,
                         help="Number of most-recent service months to use per "
                              "age-to-age factor (default: 12; reasonable range 6-24)")
    parser.add_argument("--exclude-high", type=int, default=1,
                         help="Number of highest-ratio outliers to drop from each "
                              "lookback window before averaging (default: 1)")
    parser.add_argument("--exclude-low", type=int, default=1,
                         help="Number of lowest-ratio outliers to drop from each "
                              "lookback window before averaging (default: 1)")
    parser.add_argument("--min-periods", type=int, default=6,
                         help="Minimum number of ratios that must remain after "
                              "trimming for an age-to-age factor to be averaged. "
                              "If fewer remain, the factor defaults to 1.0 (no "
                              "further development assumed). Default: 6")
    parser.add_argument("--output", help="Optional path to write the completion factor table as CSV")
    args = parser.parse_args()

    if not (6 <= args.lookback <= 24):
        parser.error("--lookback should be between 6 and 24")
    if not (0 <= args.exclude_high <= 3):
        parser.error("--exclude-high should be between 0 and 3")
    if not (0 <= args.exclude_low <= 3):
        parser.error("--exclude-low should be between 0 and 3")
    if args.exclude_high + args.exclude_low >= args.lookback:
        parser.error("--exclude-high + --exclude-low must be less than --lookback "
                      "(otherwise nothing is left to average)")
    if args.min_periods < 1:
        parser.error("--min-periods must be at least 1")

    print(f"Loading claims from {args.input_file} ...")
    df = load_claims(args.input_file)
    print(f"  {len(df):,} claims loaded")

    df = add_lag_column(df)
    max_dop_period = yyyymm_to_period(df["DOP_YYYYMM"]).max()
    earliest_month = df["DOS_period"].min()
    claims_max_lag = int(df["lag"].max())
    triangle_max_lag = (max_dop_period - earliest_month).n
    print(f"  Most recent payment period in data: {max_dop_period}")
    print(f"  Earliest service month in data: {earliest_month}")
    print(f"  Maximum lag actually observed among claims: {claims_max_lag} months")
    print(f"  Triangle will extend to lag {triangle_max_lag} months "
          f"(the oldest service month's full observable window)")

    max_lag = args.max_lag if args.max_lag is not None else claims_max_lag
    if max_lag > triangle_max_lag:
        parser.error(f"--max-lag ({max_lag}) cannot exceed the triangle's "
                      f"maximum lag ({triangle_max_lag})")

    print("\nBuilding development triangle "
          "(rows sized to the oldest service month's observable window) ...")
    incremental = build_incremental_triangle(df, max_dop_period)
    cumulative = build_cumulative_triangle(incremental)

    incremental_display = mask_unobservable_cells(incremental, max_dop_period)
    cumulative_display = mask_unobservable_cells(cumulative, max_dop_period)

    print("\nIncremental paid triangle (rows = lag in months, cols = service month, "
          "N/A = not yet observable):")
    print(incremental_display.round(0))

    print("\nCumulative paid triangle (rows = lag in months, cols = service month, "
          "N/A = not yet observable):")
    print(cumulative_display.round(0))

    print(f"\nComputing age-to-age factors "
          f"(lookback={args.lookback}, exclude_high={args.exclude_high}, "
          f"exclude_low={args.exclude_low}) ...")
    age_to_age = compute_age_to_age_factors(
        cumulative,
        max_dop_period,
        lookback=args.lookback,
        exclude_high=args.exclude_high,
        exclude_low=args.exclude_low,
        min_periods=args.min_periods,
    )
    print(age_to_age.round(4))

    print(f"\nChaining age-to-age factors into age-to-ultimate and completion factors "
          f"(assuming fully developed at lag {max_lag}) ...")
    completion_table = compute_age_to_ultimate(age_to_age["age_to_age_factor"], max_lag)
    print(completion_table.round(4))

    if args.output:
        completion_table.to_csv(args.output)
        print(f"\nCompletion factor table written to {args.output}")


if __name__ == "__main__":
    main()
