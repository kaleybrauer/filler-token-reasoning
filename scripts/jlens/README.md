# scripts/jlens — Jacobian-lens side investigation (NOT part of the paper)

Everything in this directory, plus `--jlens` options elsewhere, `outputs/jlens/` and
`results/unsupervised_decode_2fact_allpos/jlens_*/`, is a separate line of work: fitting
Anthropic's Jacobian lens (transformer-circuits.pub/2026/workspace) for DeepSeek V3 and
comparing it with the logit lens the paper uses. It was done out of research interest
after the paper's results were in. No paper figure, table or claim depends on it, and no
reviewer asked for it.

Outcome (2026-09-08, see `REFIT_RUNBOOK.md`): with the paper's Frobenius pre-filter applied,
the J-lens reads the same operands at the same layers and positions as the logit lens on the
2-fact task; the paper's logit-lens results stand unchanged.
