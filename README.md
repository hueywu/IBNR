# IBNR
Healthcare Claims Reserve Setting - IBNR Tutorial from Real Life Practictioner (Licensed Actuary)

# Completion Factors from Claims Data

A hands-on walkthrough of building **completion factors** from a raw claims extract — the multipliers used in health insurance reserving to estimate how much of a service month's ultimate cost has been paid to date.

> **Note:** Completion factors are one input into reserve development, not a complete reserve estimate. See [Scope and Limitations](#scope-and-limitations).

The underlying method — the chain-ladder — is well established and implemented in libraries such as R's `ChainLadder` package and Python's `chainladder-python`. This repo focuses on building it from scratch at the claim level, with practitioner guardrails (rolling lookback window, outlier trimming, minimum credibility threshold) that are typically learned on the job rather than found in library docs.

---

## What's a completion factor?

There's a lag between when a service occurs (DOS) and when it's paid (DOP). A completion factor captures what fraction of a service month's ultimate cost has been paid by a given number of months after service:

```
completion factor (at lag N) = paid-to-date at lag N / ultimate paid
```

Dividing paid-to-date by the completion factor estimates the **ultimate cost** for that month — a key input to IBNR liability estimates.

---

## Method

Five steps, implemented in `completion_factors.py`:

**Step 1 — Compute lag.** Months between DOS and DOP for each claim. Claims with a negative lag (payment before service) are reassigned to lag 0 with a warning.

**Step 2 — Build an incremental triangle.** Columns = service months, rows = lag months (0 through the full observable window of the oldest service month — no artificial cap). Cells not yet observable are shown as N/A, not zero.

**Step 3 — Cumulate.** Running sum down each column gives paid-to-date through lag N. N/A cells are preserved to keep the distinction between "observed zero" and "not yet known."

**Step 4 — Age-to-age factors.** For each lag transition (N → N+1):
- Compute one ratio per service month: `cumulative[lag+1] / cumulative[lag]`, observable months only.
- Use the most recent `--lookback` months (default 12, range 6–24).
- Drop the `--exclude-high` highest and `--exclude-low` lowest ratios (default 1 each, range 0–3).
- If fewer than `--min-periods` ratios remain (default 6), default to 1.0 — not enough data to trust an average.
- Average the rest.

**Step 5 — Chain to completion factors.** Multiply age-to-age factors from oldest to most recent lag to get the cumulative age-to-ultimate factor. Invert for the completion factor:

```
completion_factor[lag] = 1 / age_to_ultimate_factor[lag]
```

---

## Data

Point the script at any CSV containing these three columns:

| Column        | Description                              |
|---------------|------------------------------------------|
| `ClaimAmount` | Dollar amount of the claim               |
| `DOS_YYYYMM`  | Date of service as integer (e.g. 202403) |
| `DOP_YYYYMM`  | Date of payment as integer (e.g. 202405) |

Any additional columns in your file are ignored. The more service months of history you have, the more credible the resulting factors — at least 18–24 months of claims data is recommended.

---

## Usage

```bash
pip install -r requirements.txt
python completion_factors.py path/to/your/claims.csv
```

| Flag             | Description                                                              | Default | Range |
|------------------|--------------------------------------------------------------------------|---------|-------|
| `--lookback`     | Most recent service months used per age-to-age factor                    | 12      | 6–24  |
| `--exclude-high` | Highest-ratio outliers dropped per window                                | 1       | 0–3   |
| `--exclude-low`  | Lowest-ratio outliers dropped per window                                 | 1       | 0–3   |
| `--min-periods`  | Minimum ratios after trimming to compute an average; else defaults to 1.0 | 6      | 1+    |
| `--max-lag`      | Lag assumed fully developed; defaults to max observed lag in data        | —       | —     |
| `--output`       | Write completion factor table to CSV                                     | —       | —     |

`--exclude-high + --exclude-low` must be less than `--lookback`.

---

## Sample output

```
            age_to_ultimate_factor  completion_factor
lag_months
0                           6.3741             0.1569
1                           1.8425             0.5427
2                           1.1901             0.8403
3                           1.1027             0.9069
...
15                          1.0000             1.0000
```

A service month 1 month old is ~54% complete; by lag 4 it's ~93%; fully developed by lag 15 in this dataset.

---

## Scope and limitations

**Completion factors are one component of reserve development, not a complete reserve estimate.**

A full IBNR analysis involves additional actuarial judgment beyond what's shown here. The scope is defined in **ASOP No. 5 — Incurred Health and Disability Claims**, and includes:

- Adjustments for changes in claim processing speed, payment patterns, or data systems
- Trend adjustments for utilization, unit cost, and benefit design changes
- Large claim identification and treatment
- IBNR vs. IBNEP segmentation
- Credibility considerations when data is thin
- Validation against expected loss ratios and other benchmarks
- Appointed Actuary requirements for statutory filings

This repo demonstrates the completion factor step only. Translating these factors into a reserve or incurred estimate requires further analysis consistent with ASOP No. 5.

---

## Questions or want to go further?

For questions about this tutorial, or to discuss applying completion factor modeling within a full reserve development process, contact Huey "Huiqing" Wu at wuhuey@outlook.com.

---

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to use, share, and adapt this material for any purpose, including commercially, as long as you give appropriate credit to the original author.
