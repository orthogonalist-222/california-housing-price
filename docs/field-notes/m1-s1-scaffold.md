# M1-S1 — Repo scaffold, CI and protocol docs

**Built.** An empty but opinionated repo: `uv` project on Python 3.12 (CI also
runs 3.11, which is Kaggle's image), a `src/calhousing/` package that so far
contains only its own docstring, two ledgers, and — the actual content of this
story — **two gates that can refuse.**

**Why this way.**

*The layout gate.* The protocol says a path the declared tree does not name may
not be tracked. That rule is worthless as prose: nobody adds a stray directory
on purpose, they add one file at a time until the tree in the README and the
tree on disk are different documents, and by then the README is decoration.
`tools/check_layout.py` holds the tree as an allowlist of globs and refuses
anything else in `git ls-files`.

Two details are load-bearing:

- **`fnmatch` is wrong here.** Its `*` matches `/`, so `docs/field-notes/*.md`
  would bless `docs/field-notes/nested/sneak.md` — exactly the drift the gate
  exists to catch. The globs are translated to anchored regexes where `*` means
  `[^/]*` and only `**/` crosses a directory boundary.
- **FORBIDDEN is checked before DECLARED, and says something different.** A
  committed `kaggle.json` and a stray `scratch.py` are not the same problem and
  must not read alike. "undeclared path" would send someone to add an allowlist
  entry for their credentials.

*The story-lint.* One story → one branch → one PR title, checked in CI. A
convention nobody enforces drifts within a single milestone.

**The concept: a safeguard nobody has watched refuse anything is an assumption.**
Both gates were red-teamed before this story closed — planted defect, observed
refusal, defect removed, both halves logged together. The planted `kaggle.json`
needed `git add -f` to stage at all, because `.gitignore` caught it first; that
is the belt-and-braces working, not a reason to drop either layer.

**What to look at.** `tools/check_layout.py` — specifically `_to_regex`, and the
`classify` split into two lists. Then `tests/test_layout.py`, and note what it
refuses to assert: not the number of tracked files, not that any particular file
exists. Those are era-facts that change every story, and a guard that fires on
correct behaviour teaches the next session to edit the assertion instead of the
code. It asserts the *invariant*: undeclared is refused, declared is not, and
the two refusals are distinguishable.

**What to try.** Create `src/calhousing/anything.py`, `git add` it, and run
`uv run python tools/check_layout.py`. Then add the matching entry to `DECLARED`
and watch it pass. That two-step is the whole mechanism — adding the file and
declaring it are deliberately separate acts.
