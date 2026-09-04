"""Test that all config values are documented."""

import ast
import logging
import os
import re
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from mne_bids_pipeline._config_import import _get_default_config, _import_config
from mne_bids_pipeline._config_template import create_template_config
from mne_bids_pipeline._config_utils import _bids_kwargs, _get_step_modules
from mne_bids_pipeline._docs import (
    _EXECUTION_OPTIONS,
    _FORCE_EMPTY,
    _ParseConfigSteps,
    get_kwargs_helper_fields,
    get_step_cfg_fields,
    get_step_reads,
)
from mne_bids_pipeline._logging import _log_context
from mne_bids_pipeline.tests.datasets import DATASET_OPTIONS
from mne_bids_pipeline.tests.test_run import TEST_SUITE

root_path = Path(__file__).parent.parent


def test_options_documented() -> None:
    """Test that all options are suitably documented."""
    # use ast to parse _config.py for assignments
    with open(root_path / "_config.py") as fid:
        contents_str = fid.read()
    contents = ast.parse(contents_str)
    assert isinstance(contents, ast.Module), type(contents)
    in_config_list = [
        item.target.id
        for item in contents.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]
    assert len(set(in_config_list)) == len(in_config_list)
    in_config = set(in_config_list)
    del in_config_list
    # ensure we clean our namespace correctly
    config = _get_default_config()
    config_names = set(d for d in dir(config) if not d.startswith("_"))
    assert in_config == config_names
    settings_path = root_path.parent / "docs" / "source" / "settings"
    sys.path.append(str(settings_path))
    try:
        from gen_settings import main  # type: ignore
    finally:
        sys.path.pop()
    main()
    assert settings_path.is_dir()
    in_doc: dict[str, set[str]] = dict()
    key = "        - "
    for dirpath, _, fnames in os.walk(settings_path):
        for fname in fnames:
            if not fname.endswith(".md"):
                continue
            # This is a .md file
            # convert to relative path
            fname = os.path.join(os.path.relpath(dirpath, settings_path), fname)
            assert fname not in in_doc
            in_doc[fname] = set()
            with open(settings_path / fname) as fid:
                for line in fid:
                    if not line.startswith(key):
                        continue
                    # The line starts with our magic key
                    val = line[len(key) :].strip()
                    for other in in_doc:
                        why = f"Duplicate docs in {fname} and {other} for {val}"
                        assert val not in in_doc[other], why
                    in_doc[fname].add(val)
    what = "docs/source/settings/*.md docs created by gen_settings.py"
    in_doc_all = set()
    for vals in in_doc.values():
        in_doc_all.update(vals)
    assert in_doc_all.difference(in_config) == set(), f"Extra values in {what}"
    assert in_config.difference(in_doc_all) == set(), f"Values missing from {what}"


def test_config_options_passed_to_any_steps() -> None:
    """Test that all config options are used somewhere."""
    config = _get_default_config()
    config_names = set(d for d in dir(config) if not d.startswith("__"))
    for key in ("_epochs_split_size", "_raw_split_size"):
        config_names.add(key)
    for key in _EXECUTION_OPTIONS:
        config_names.remove(key)
    # a stale entry here would silently blank out a step list forever
    stale = sorted(set(_FORCE_EMPTY) - config_names - set(_EXECUTION_OPTIONS))
    assert stale == [], f"No longer config options, drop from _FORCE_EMPTY: {stale}"
    pcs = _ParseConfigSteps(force_empty=())
    missing_from_config = sorted(set(pcs.steps) - config_names)
    assert missing_from_config == [], f"Missing from config: {missing_from_config}"
    missing_from_steps = sorted(config_names - set(pcs.steps))
    assert missing_from_steps == [], f"Missing from steps: {missing_from_steps}"
    for key, val in pcs.steps.items():
        assert val, f"No steps for {key}"
    # Spot check for EXTRA_FUNCS
    assert "sensor/_03_decoding_time_by_time" in pcs.steps["cov_rank"]


def _step_modules() -> dict[ModuleType, str]:
    """Get each step module and its "group/name" label, without duplicates."""
    modules = sum(_get_step_modules().values(), ())  # "all" repeats other groups
    return {
        module: "/".join(module.__name__.split(".")[-2:])
        for module in dict.fromkeys(modules)
    }


def _format_errors(errors: list[tuple[str, str]], what: str, hint: str) -> str:
    use_len = max(len(err[0]) for err in errors)
    error_str = "\n".join(f"- {err[0].ljust(use_len)} : {err[1]}" for err in errors)
    return f"{len(errors)} {what}:\n\n{error_str}\n\n{hint}"


def test_cfg_fields_used_in_steps() -> None:
    """Test that every cfg field a step builds is actually read by that step."""
    # `_bids_kwargs` is deliberately all-or-nothing: it is the BIDS identity of
    # the data, and the steps that use any of it get all of it. That is the same
    # reasoning behind listing those options in `_FORCE_EMPTY` -- see the comment
    # there. Everything else has to earn its place in `cfg`.
    bids_fields = get_kwargs_helper_fields(_bids_kwargs)
    errors = []
    for module, step in _step_modules().items():
        unread = get_step_cfg_fields(module) - get_step_reads(module) - bids_fields
        errors.extend(
            (f"mne_bids_pipeline/steps/{step}.py", var) for var in sorted(unread)
        )
    if errors:  # pragma: no cover
        raise AssertionError(
            _format_errors(
                errors,
                "cfg field(s) passed to a step that never reads them",
                "Passing a config option a step does not use makes the docs claim a "
                "dependency that is not real, and needlessly invalidates that step's "
                "cache when the option changes. Drop the field from get_config(), or "
                "if it really is used, teach _docs.get_step_reads() how to see it.",
            )
        )


def test_cfg_fields_passed_to_steps() -> None:
    """Test that every cfg field a step reads is actually built by that step."""
    # `get_tasks()` returns early on `config.all_tasks`, which `_bids_kwargs`
    # always supplies, so the `config.task` fallback inside it is unreachable
    # when it is handed a cfg rather than the real config.
    guarded = {"task"}
    errors = []
    for module, step in _step_modules().items():
        missing = get_step_reads(module) - get_step_cfg_fields(module) - guarded
        errors.extend(
            (f"mne_bids_pipeline/steps/{step}.py", var) for var in sorted(missing)
        )
    if errors:  # pragma: no cover
        raise AssertionError(
            _format_errors(
                errors,
                "cfg field(s) read by a step that never builds them",
                "This is an AttributeError waiting to happen: the step reads "
                "cfg.<field> but its get_config() never sets it. Add it to "
                "get_config(), or stop reading it.",
            )
        )


def test_datasets_in_doc() -> None:
    """Test that all datasets in tests are in the doc."""
    # There are four things to keep in sync:
    #
    # 1. CircleCI caches, tests, etc. (actually multiple things!)
    # 2. docs/mkdocs.yml
    # 3. tests/datasets.py:DATASET_OPTIONS (imported above)
    # 4. tests/test_run.py:TEST_SUITE (imported above)
    #
    # So let's make sure they stay in sync.

    # 1. Read cache, test, etc. entries from CircleCI
    with open(root_path.parent / ".circleci" / "config.yml") as fid:
        circle_yaml_src = fid.read()
    circle_yaml = yaml.safe_load(circle_yaml_src)
    caches_list = [job[6:] for job in circle_yaml["jobs"] if job.startswith("cache_")]
    caches = set(caches_list)
    assert len(caches_list) == len(caches)
    tests_list = [job[5:] for job in circle_yaml["jobs"] if job.startswith("test_")]
    assert len(tests_list) == len(set(tests_list))
    tests = set(tests_list)
    # Rather than going circle_yaml['workflows']['commit']['jobs'] and
    # make sure everything is consistent there (too much work), let's at least
    # check that we get the correct number using `.count`.
    counts = dict(ERP_CORE=7, ds000248=6, ds003392=3)
    counts_noartifact = dict(ds000248=1)  # 1 is actually a test, not for docs
    for name in sorted(caches):
        get = f"Get {name}"
        n_found = circle_yaml_src.count(get)
        assert n_found == 1, get
        dl = f"$DOWNLOAD_DATA {name}"
        n_found = circle_yaml_src.count(dl)
        assert n_found == 1, dl
        # jobs: save_cache:
        sc = f"key: data-cache-{name}"
        n_found = circle_yaml_src.count(sc)
        assert n_found == 1, sc
        # jobs: restore_cache:
        rc = f"- data-cache-{name}-"
        n_found = circle_yaml_src.count(rc)
        count = counts.get(name, 1) + 1  # one restore
        assert n_found == count, f"{rc} ({n_found} != {count})"
        # jobs: save_cache: paths:
        pth = f"- ~/mne_data/{name}"
        n_found = circle_yaml_src.count(pth)
        assert n_found == 1, pth
        # jobs:
        cj = f"  cache_{name}:"
        n_found = circle_yaml_src.count(cj)
        assert n_found == 1, cj
        tj = f"  test_{name}"
        n_found = circle_yaml_src.count(tj)
        count = counts.get(name, 1)
        assert n_found == count, f"{tj} ({n_found} != {count})"
        # jobs: test_*: steps: store_artifacts
        sa = f"path: /home/circleci/reports/{name}"
        n_found = circle_yaml_src.count(sa)
        this_count = count - counts_noartifact.get(name, 0)
        assert n_found == this_count, f"{sa} ({n_found} != {this_count})"
        # jobs: test_*: steps: persist_to_workspace
        pw = re.compile(
            f"- mne_data/derivatives/mne-bids-pipeline/{name}[^\\.]+\\*.html"
        )
        n_found = len(pw.findall(circle_yaml_src))
        assert n_found == this_count, f"{pw} ({n_found} != {this_count})"
        # jobs: test_*: steps: run test
        cp = re.compile(rf" command: \$RUN_TESTS[ -rc]+{name}.*")
        n_found = len(cp.findall(circle_yaml_src))
        assert n_found == count, f"{cp} ({n_found} != {count})"

    # 3. Read examples from docs (being careful about tags we can't read)
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node: yaml.Node) -> None:
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None,
        SafeLoaderIgnoreUnknown.ignore_unknown,
    )

    with open(root_path.parent / "docs" / "mkdocs.yml") as fid:
        examples = yaml.load(fid.read(), Loader=SafeLoaderIgnoreUnknown)
    examples = [n for n in examples["nav"] if list(n)[0] == "Examples"][0]
    examples = [ex for ex in examples["Examples"] if isinstance(ex, str)]
    examples = [ex.split("/")[-1].split(".")[0] for ex in examples]
    assert len(examples) == len(set(examples))
    examples = set(examples)
    examples.remove("examples")

    # 4. DATASET_OPTIONS
    dataset_names_list = list(DATASET_OPTIONS)
    dataset_names = set(dataset_names_list)
    assert len(dataset_names_list) == len(dataset_names)

    # 5. TEST_SUITE
    test_names_list = list(TEST_SUITE)
    test_names = set(test_names_list)
    assert len(test_names_list) == len(test_names)

    # Some have been split into multiple test runs, so trim down to the same
    # set as caches
    for key in ("ERP_CORE", "ds000248"):
        tests = set(
            job if not job.startswith(key) else job[: len(key)] for job in tests
        )
        dataset_names = set(
            name if not name.startswith(key) else name[: len(key)]
            for name in dataset_names
        )
        examples = set(
            ex if not ex.startswith(key) else ex[: len(key)] for ex in examples
        )
        test_names = set(
            test if not test.startswith(key) else test[: len(key)]
            for test in test_names
        )
    assert tests == caches, "CircleCI tests != CircleCI caches"
    assert tests == examples, "CircleCI tests != docs/mkdocs.yml Examples"
    assert tests == dataset_names, "CircleCI tests != tests/datasets.py"
    assert tests == test_names, "CircleCI tests != tests/test_run.py"


def _replace_config_value_in_file(fpath: Path, config_key: str, new_value: str) -> None:
    """Assign a value to a config key in a file, and uncomment the line if needed."""
    lines = fpath.read_text().split("\n")
    pattern = re.compile(rf"(?:# )?({config_key}: .* = )(?:.*)")
    for ix, line in enumerate(lines):
        if pattern.match(line):
            lines[ix] = pattern.sub(
                rf"\1{new_value}",
                line,  # omit comment marker, change default value to `new_value`
            )
            break
    fpath.write_text("\n".join(lines))


@pytest.fixture
def no_log() -> Generator[None, None, None]:
    """Disable logging."""
    with _log_context(logging.CRITICAL):
        yield


def test_config_template_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_log: None
) -> None:
    """Ensure our config template is syntactically valid (importable)."""
    monkeypatch.setenv("BIDS_ROOT", str(tmp_path))
    fpath = tmp_path / "foo.py"
    create_template_config(fpath)
    # `ch_types` fails pydantic validation (its default is `[]` but its annotation
    # requires length > 0)
    with pytest.raises(
        ValueError, match="ch_types\n  Value should have at least 1 item"
    ):
        _import_config(config_path=fpath)
    # Give `ch_types` a value so pydantic will succeed...
    _replace_config_value_in_file(fpath, "ch_types", '["meg"]')
    # ...but now `_check_config` will raise an error that `conditions` cannot be None
    # unless `task_is_rest = True` (which defaults to False)
    with pytest.raises(ValueError, match="the `conditions` parameter is empty"):
        _import_config(config_path=fpath)
    # give a non-None value for `conditions`, now importing the config should work
    _replace_config_value_in_file(fpath, "conditions", '["foo"]')
    _import_config(config_path=fpath)
