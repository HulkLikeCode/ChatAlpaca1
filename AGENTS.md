# ChatAlpaca1 agent guidance

- Treat `README.md` and `docs/architecture/` and `docs/decisions/` as authoritative product and architecture guidance.
- Preserve the transaction ledger as the accounting source of truth for cash, holdings, FIFO lots, basis, history, contributions, withdrawals, income, fees, and taxes.
- Keep Streamlit as the UI layer; put financial calculations and market-data logic in reusable modules outside Streamlit pages.
- Use versioned database migrations for schema changes. Preserve paper mode and all existing live-trading safety gates.
- Never commit credentials. Inspect relevant code and tests before editing; do not edit files for analysis- or planning-only requests.
- Avoid broad rewrites unless code-audit evidence justifies incremental migration as impractical. Request approval before materially changing the architecture.
- Run the documented checks: `ruff format --check .`, `ruff check .`, and `pytest -q`.
- In handoff, report changed files, tests run and failures, assumptions, and remaining manual tests.
