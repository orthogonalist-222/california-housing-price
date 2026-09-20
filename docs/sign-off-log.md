# Sign-off log

Every gate crossing: what was produced, who approved it, the verdict, and the
**evidence** — the observation that proved it, not the intention behind it.
Append-only; a row is never rewritten.

Roles in this project: `data scientist`, `ML engineer`, `tech lead`,
`model validator` (fresh-session assurance only — no self-sign-off).

| Date | Story | Gate | Producer | Approver | Verdict | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-20 | M1-S1 | `uv sync --dev` provisions | tech lead | tech lead | PASS | Python 3.12.13, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.5.3 |
| 2026-09-20 | M1-S1 | `uv run pytest -q` | tech lead | tech lead | PASS | 6 passed |
| 2026-09-20 | M1-S1 | Layout gate, control | tech lead | tech lead | PASS | `layout gate OK: 14 tracked files, all declared.` (exit 0) |
| 2026-09-20 | M1-S1 | Layout gate, **red-team** | tech lead | tech lead | PASS | See RT-001 below |
| 2026-09-20 | M1-S1 | Story-lint, **red-team** | tech lead | tech lead | PASS | See RT-002 below |

## Red-team records

A safeguard that has never been seen rejecting anything is an assumption. Each
record holds **both halves**: the planted defect and the observed refusal, then
the removal and the restored pass.

### RT-001 — the layout gate refuses, distinguishably (2026-09-20)

Planted: `src/calhousing/scratch.py` (undeclared) and `kaggle.json` (a fake
credential file), both force-staged — `.gitignore` refused `kaggle.json` first,
so `git add -f` was needed to reach the gate at all.

Observed:

```
FORBIDDEN TRACKED PATH: kaggle.json
    Kaggle credentials must never be committed
UNDECLARED TRACKED PATH: src/calhousing/scratch.py
    Not covered by DECLARED in tools/check_layout.py. ...

layout gate FAILED: 1 forbidden, 1 undeclared, out of 16 tracked files.
exit=1
```

Two distinct signatures, as designed. Both files removed from the index and
from disk; the gate then reported `layout gate OK: 14 tracked files, all
declared.` with exit 0.

### RT-002 — the story-lint refuses, both failure modes (2026-09-20)

The CI job's shell logic run locally against four inputs:

| Input | Result |
| --- | --- |
| branch `docs/m3-kickoff` (ad-hoc prefix) | REFUSED — not `feat/m<M>-s<S>-<slug>` |
| branch `feat/m1-s1-scaffold`, title `M1-S2: wrong id` | REFUSED — title does not start with `M1-S1: ` |
| branch `feat/m1-s1-scaffold`, title `M1-S1: Repo scaffold, CI and protocol docs` | OK |
| branch `feat/m4-s2.1-hotfix`, title `M4-S2.1: fix the kernel` | OK — point stories are permitted |

The first row is the failure this gate exists for: the sibling program's
`docs/m3-kickoff`-style PRs are precisely what the protocol calls "a unit that
has left the map".
