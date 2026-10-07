# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

MNE-BIDS-Pipeline is a configurable, cached processing pipeline for M/EEG data stored in
[BIDS](https://bids-specification.readthedocs.io): users write a Python config file and
run `mne_bids_pipeline --config=config.py`, and the pipeline runs preprocessing (Maxwell
filtering, filtering, ICA/SSP, epoching, artifact rejection), sensor-level analysis
(evokeds, decoding, time-frequency, covariances), source-level analysis (BEM, forward,
inverse), and group averaging, writing derivatives and an HTML report per subject.
MNE-Python does the processing and MNE-BIDS does the BIDS I/O; this package orchestrates
them.

## Follow MNE-Python's conventions

This package is a subsidiary of MNE-Python. Unless something below says otherwise, follow
[MNE-Python's AGENTS.md](https://github.com/mne-tools/mne-python/blob/main/AGENTS.md)
(read it rather than guessing): keep changes small, naming, imports, compact tests,
license rules for adapted code, and in particular its
[policy on AI assistance](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md#policy-on-ai-assistance-in-contributions):

- Work test-first: write (or extend) a test that fails for the right reason, then make it
  pass. Promote anything a throwaway script caught into a real test.
- Do not open pull requests, push, or commit unless explicitly asked; the human submitting
  the change must review, understand, and disclose AI use in the PR description.
- Keep changes minimal and scoped to the request; mention, don't silently fix, unrelated
  problems you notice.

What does *not* carry over from MNE-Python:

- There are no towncrier fragments. The changelog is `docs/source/dev.md`: add a bullet
  under the right heading of the unreleased version, in the existing format (ending in
  `(#1234 by @handle)`), linking any config option as
  ``[`name`][mne_bids_pipeline._config.name]``. Uncomment a heading if needed (the
  `[//N]: # (...)` lines; `dev.md.inc.template` is the template for a new version).
- There is no public Python API, `docdict`, or numpydoc-documented functions: the
  user-facing API is the config options and the command-line interface. Functions are
  private and keyword-only with type hints, and `ty` checks them.
- Config options are not deprecated MNE-style; when renaming or removing one, add it to
  `_REMOVED_NAMES` in `_config_import.py` so users get a helpful error, and note it under
  "Behavior changes".
- The default branch is `main`.

## Layout and how things fit together

- `mne_bids_pipeline/_config.py` — every config option, its type annotation (validated
  with pydantic in `_config_import.py`), its default, and its docstring, which *is* the
  user documentation (rendered by `docs/source/settings/gen_settings.py`, organized by
  the `# %%` / `# #` section comments). `_config_import.py` loads the user's config on
  top of these defaults and does the checks pydantic can't express.
- `mne_bids_pipeline/steps/<group>/_NN_<name>.py` — one module per step, run in order.
  Each has `get_config(*, config, ...)` building a `SimpleNamespace` `cfg` with only the
  options that step needs, `get_input_fnames_*` returning the files it reads, a worker
  function decorated with `@failsafe_run(get_input_fnames=...)` that pops its inputs off
  `in_files` and returns `_prep_out_files(...)`, and `main(*, config)` that parallelizes
  over subjects/sessions/runs with `parallel_func`.
- `_run.py` — `failsafe_run` handles `on_error` and the joblib-backed cache: a step is
  skipped when its `cfg` and the hashes (or mtimes) of its input files are unchanged. So
  every option a step uses must go through its `cfg`, and every file it reads must be in
  its input fnames, or results silently go stale. `test_documented.py` checks the `cfg`
  fields are both set and read. `_update_for_splits` handles split FIF files.
- `_config_utils.py` — subject/session/run/task enumeration and other helpers shared
  across steps; `_import_data.py` — reading raw data via MNE-BIDS; `_report.py` —
  report sections; `_parallel.py` — joblib/Dask (including `dask_cluster` for HPC).
- `docs/` — MkDocs (not Sphinx), built from the dataset test outputs (see below).

## Adding a config option

Add it to `_config.py` in the right section with a type annotation, default, and
docstring (with an `???+ example` block where useful); pass it through the `get_config`
of every step that uses it; add validation beyond the type hint to `_config_import.py`
if needed; exercise it in a test config if it changes processing; and add a changelog
entry linking it. `test_documented.py` fails if an option is undocumented or unused.

## Tests and docs

`pytest mne_bids_pipeline -m "not dataset_test"` runs the fast unit tests. The real
coverage is the dataset tests in `tests/test_run.py`: `TEST_SUITE` maps a key to a
dataset and a config in `tests/configs/config_<key>.py`, each running the full pipeline
on (mostly OpenNeuro) data via `pytest -k <key>`; add `--download` to fetch the data to
`~/mne_data` first (see `tests/datasets.py`). These are slow and some need FreeSurfer, so
CircleCI runs them; locally, pick the smallest dataset that exercises your change
(`ds000248_ica` and `eeg_matchingpennies` are good starting points). Warnings are errors
(`tests/conftest.py`): add a narrowly scoped `ignore` line there (with a comment naming
the source) only for third-party noise.

Steps are cached in the derivatives' `_cache` directory, so rerunning a dataset test
does not necessarily rerun the steps you changed: a step reruns only when its `cfg`, its
input files, or the source of its own `@failsafe_run`-decorated function changed, not
when a helper it calls (in `_config_utils.py`, `_report.py`, ...) did. Pass `--no-cache`
to `mne_bids_pipeline` or delete the dataset's derivatives to force a rerun, and check
the log to see which steps actually ran rather than assuming a passing test exercised
your change. Conversely, CircleCI runs each dataset twice and fails if the second run
isn't fast, so a change that defeats the cache (e.g. an unstable `cfg` value or output
file) is a bug.

The dataset tests also serve as the documentation's examples: CircleCI builds the docs
(MkDocs, not Sphinx) from the derivatives those test jobs produce, and
`docs/source/examples/gen_examples.py` turns each test config and its HTML reports into
an example page. So a test config is also user-facing example code, and changing it
changes the docs. A new test config that should appear as an example must follow the
naming rules in `tests/configs/README.md` and be added to `docs/mkdocs.yml`.
`docs/build-docs.sh` builds the docs (`mkdocs build --strict`) once the dataset tests
have run locally. Run `prek run --all-files` (ruff, ruff-format, ty, tombi, codespell)
before handing work back.
