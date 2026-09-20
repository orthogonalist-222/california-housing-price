@~/.claude/templates/UNIVERSAL_PROTOCOL.md
PROTOCOL_MODE: learning

# california-housing-price - working memory

Protocol version last confirmed by the user: **2.18 (2026-09-17)**.

The rituals below govern **chartered work** - a numbered story from the plan.
Ad-hoc questions in this repo get the knowledge and none of the ceremony.

## What this project is

Predict `median_house_value` on `camnugent/california-housing-prices`, with the
*preprocessing pipeline* as the deliverable and the model as its passenger. The
user's background is PySpark ML pipelines; the notebook's job is to show the
same discipline in scikit-learn.

## Environment facts

| Fact | Value |
| --- | --- |
| Package manager | `uv` (`C:\Users\longt\.local\bin\uv.exe`) - **not** the system Python |
| System Python | 3.9.13, empty. Never used. `uv` provisions 3.12. |
| Local target | 3.12 (`.python-version`) |
| CI matrix | 3.11 (Kaggle's notebook image) and 3.12 |
| GitHub | `orthogonalist-222/california-housing-price`, public. Repo-local `user.name`/`user.email` are set to that identity; the global default is `Phu-Hong-Duong` and stays untouched. |
| Kaggle | owner slug `duonghongphu`; CLI at `...\Python39\Scripts\kaggle.exe`, credentials in `~/.kaggle/kaggle.json` |
| Sibling repo | `../Wine Quality Dataset` - the house template for uv layout, generated notebooks and `tools/stage_kaggle.py` |

## Commands

```
uv sync --dev                          # provision
uv run pytest -q                       # tests
uv run python tools/check_layout.py    # layout gate
```

Raw data (never tracked):

```
kaggle datasets download -d camnugent/california-housing-prices -p data/raw --unzip
```

## Known traps

- **The system Python 3.9 has no packages at all.** `python -m pytest` outside
  `uv run` fails with `ModuleNotFoundError` that looks like a broken install.
  Everything goes through `uv run`.
- **`git config user.*` is repo-local here.** A commit made after a `git init`
  elsewhere in this tree would carry the wrong identity; check before pushing.
- **The layout gate is a CI job, not a hook.** A file added and committed
  without a `DECLARED` entry passes locally and fails on the PR. Run
  `uv run python tools/check_layout.py` before pushing.
- **`git push` used the wrong GitHub account, and said so only as a 403.**
  Git Credential Manager is configured system-wide
  (`C:/Program Files/Git/etc/gitconfig`) and handed out `loadbearingcode-222`
  while `gh`'s active account was `orthogonalist-222`:
  `Permission to orthogonalist-222/... denied to loadbearingcode-222`. Three
  accounts are logged into `gh` on this machine, so the symptom reads as a
  permissions problem on the repo rather than an identity mix-up. Fixed
  **repo-locally** so the machine-wide setting is untouched:

  ```
  git config --local credential.helper ""
  git config --local --add credential.helper "!gh auth git-credential"
  ```

  Ten-second check: `gh api user --jq .login` vs the name in the 403.
  Undo: `git config --local --unset-all credential.helper`.
- **`git add -A --renormalize .` does not stage new files.** `--renormalize`
  only revisits paths git already tracks, so a freshly written field note was
  silently left out of a commit that otherwise looked complete. Use a plain
  `git add -A`, then read `git status --short` for `??` lines.

## Ledgers

- `docs/sign-off-log.md` - every gate crossing, with its evidence.
- `docs/findings-register.md` - F-### findings, closed only by evidence.
- `HANDOFF.md` - session log, **untracked** by design.
