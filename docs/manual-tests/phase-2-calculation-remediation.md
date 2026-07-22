# Phase 2 calculation-remediation acceptance record

> Phase 2 accepted. All automated and manual checks passed. The remediation branch was merged
> into main.

## Merge record

- Pull request: [#6 — Audit phase 2 remediation](https://github.com/HulkLikeCode/ChatAlpaca1/pull/6)
- Audit baseline: `a1cfe451b827dd4f83288f50d94b477ce01788ed`
- Remediation branch: `audit-phase-2-remediation`
- Remediation head: `11e3e2876ed626148012bc0d6919d753a32d07ef`
- Merge commit: `dcc20d69104dbf66c05fb5efd32efe520d35dbf1`
- Target branch: `main`
- Merged: July 22, 2026 at 2:59 AM ET

The accepted boundary contains the 34 Phase 2 audit records. `HOLD-004` and `MON-012` remained
excluded, and the accepted `RET-007` methodology was touched only through regression coverage.

## Automated results

- [GitHub Actions CI run #49](https://github.com/HulkLikeCode/ChatAlpaca1/actions/runs/29898525632)
  completed successfully against remediation head `11e3e2876ed626148012bc0d6919d753a32d07ef`.
  The workflow used Python 3.12 and passed `ruff format --check .`, `ruff check .`, and `pytest -q`.
- The final pre-merge local verification passed `ruff format --check .` with 58 files already
  formatted, passed `ruff check .`, and passed `pytest -q` with 307 tests. Pytest reported one
  non-failing third-party `websockets` legacy deprecation warning.
- `git diff --check` passed before merge.

## Manual-test outcomes

The project owner confirmed the required Phase 2 manual checks passed:

- Scenario views used and disclosed one common valuation date, and historical replay displayed
  the aligned endpoint dates and jointly complete observation evidence.
- Exact Holdings retained numeric sorting and raw values while displaying the approved labels,
  observation dates, and adaptive precision for `1`, `1.25`, and `0.00000001` shares.
- Hypothetical-analysis disclosures correctly distinguished benchmark-relative exposure,
  covariance-based volatility, component risk contribution, and the simplified adjacent
  retirement metric from the full retirement model.
- The affected disclosures, tables, and labels rendered correctly at both narrow and wide browser
  widths.
- Monitor preserved paper-mode behavior and displayed the approved quote provenance and fallback
  hierarchy without changing any live-trading gate.

No manual-test discrepancies remained open at acceptance.
