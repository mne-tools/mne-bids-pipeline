"""Test that all config values are documented."""
import ast
from pathlib import Path
import os
import re
import yaml

from mne_bids_pipeline.tests.datasets import DATASET_OPTIONS
from mne_bids_pipeline.tests.test_run import TEST_SUITE
from mne_bids_pipeline._config_import import _get_default_config

root_path = Path(__file__).parent.parent


def test_options_documented():
    """Test that all options are suitably documented."""
    # use ast to parse _config.py for assignments
    with open(root_path / "_config.py", "r") as fid:
        contents = fid.read()
    contents = ast.parse(contents)
    in_config = [
        item.target.id for item in contents.body
        if isinstance(item, ast.AnnAssign)
    ]
    assert len(set(in_config)) == len(in_config)
    in_config = set(in_config)
    # ensure we clean our namespace correctly
    config = _get_default_config()
    config_names = set(d for d in dir(config) if not d.startswith('_'))
    assert in_config == config_names
    settings_path = root_path.parent / "docs" / "source" / "settings"
    assert settings_path.is_dir()
    in_doc = set()
    key = "::: mne_bids_pipeline._config."
    allowed_duplicates = set(
        [
            "source_info_path_update",
        ]
    )
    for dirpath, _, fnames in os.walk(settings_path):
        for fname in fnames:
            if not fname.endswith(".md"):
                continue
            # This is a .md file
            with open(Path(dirpath) / fname, "r") as fid:
                for line in fid:
                    if not line.startswith(key):
                        continue
                    # The line starts with our magic key
                    val = line[len(key):].strip()
                    if val not in allowed_duplicates:
                        assert val not in in_doc, "Duplicate documentation"
                    in_doc.add(val)
    assert in_doc.difference(in_config) == set(), "Extra values in doc"
    assert in_config.difference(in_doc) == set(), "Values missing from doc"


def test_datasets_in_doc():
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
    with open(root_path.parent / '.circleci' / 'config.yml', 'r') as fid:
        circle_yaml_src = fid.read()
    circle_yaml = yaml.safe_load(circle_yaml_src)
    caches = [
        job[6:] for job in circle_yaml['jobs'] if job.startswith('cache_')
    ]
    assert len(caches) == len(set(caches))
    caches = set(caches)
    tests = [
        job[5:] for job in circle_yaml['jobs'] if job.startswith('test_')
    ]
    assert len(tests) == len(set(tests))
    tests = set(tests)
    # Rather than going circle_yaml['workflows']['commit']['jobs'] and
    # make sure everything is consistent there (too much work), let's at least
    # check that we get the correct number using `.count`.
    counts = dict(ERP_CORE=7, ds000248=6)
    counts_noartifact = dict(ds000248=3)  # 3 are actually tests, not for docs
    for name in sorted(caches):
        get = f'Get {name}'
        n_found = circle_yaml_src.count(get)
        assert n_found == 1, get
        dl = f'$DOWNLOAD_DATA {name}'
        n_found = circle_yaml_src.count(dl)
        assert n_found == 1, dl
        # jobs: save_cache:
        sc = f'key: data-cache-{name}'
        n_found = circle_yaml_src.count(sc)
        assert n_found == 1, sc
        # jobs: restore_cache:
        rc = f'- data-cache-{name}-'
        n_found = circle_yaml_src.count(rc)
        count = counts.get(name, 1) + 1  # one restore
        assert n_found == count, f'{rc} ({n_found} != {count})'
        # jobs: save_cache: paths:
        pth = f'- ~/mne_data/{name}'
        n_found = circle_yaml_src.count(pth)
        assert n_found == 1, pth
        # jobs:
        cj = f'  cache_{name}:'
        n_found = circle_yaml_src.count(cj)
        assert n_found == 1, cj
        tj = f'  test_{name}'
        n_found = circle_yaml_src.count(tj)
        count = counts.get(name, 1)
        assert n_found == count, f'{tj} ({n_found} != {count})'
        # jobs: test_*: steps: store_artifacts
        sa = f'path: /home/circleci/reports/{name}'
        n_found = circle_yaml_src.count(sa)
        this_count = count - counts_noartifact.get(name, 0)
        assert n_found == this_count, f'{sa} ({n_found} != {this_count})'
        # jobs: test_*: steps: persist_to_workspace
        pw = re.compile(f'- mne_data/derivatives/mne-bids-pipeline/{name}[^\\.]+\\*.html')  # noqa: E501
        n_found = len(pw.findall(circle_yaml_src))
        assert n_found == this_count, f'{pw} ({n_found} != {this_count})'
        # jobs: test_*: steps: run test
        cp = re.compile(f"""\
            DS={name}.*
            \\$RUN_TESTS \\${{DS}}.*
            mkdir -p ~/reports/\\${{DS}}
            cp -av ~/mne_data/derivatives/mne-bids-pipeline/\\${{DS}}/[^\\.]+.html""")  # noqa: E501
        n_found = len(cp.findall(circle_yaml_src))
        assert n_found == this_count, f'{cp} ({n_found} != {this_count})'

    # 3. Read examples from docs (being careful about tags we can't read)
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(root_path.parent / 'docs' / 'mkdocs.yml', 'r') as fid:
        examples = yaml.load(fid.read(), Loader=SafeLoaderIgnoreUnknown)
    examples = [n for n in examples['nav'] if list(n)[0] == 'Examples'][0]
    examples = [ex for ex in examples['Examples'] if isinstance(ex, str)]
    examples = [ex.split('/')[-1].split('.')[0] for ex in examples]
    assert len(examples) == len(set(examples))
    examples = set(examples)

    # 4. DATASET_OPTIONS
    dataset_names = list(DATASET_OPTIONS)
    assert len(dataset_names) == len(set(dataset_names))

    # 5. TEST_SUITE
    test_names = list(TEST_SUITE)
    assert len(test_names) == len(set(test_names))

    # Some have been split into multiple test runs, so trim down to the same
    # set as caches
    for key in ('ERP_CORE', 'ds000248'):
        tests = set(
            job if not job.startswith(key) else job[:len(key)]
            for job in tests
        )
        dataset_names = set(
            name if not name.startswith(key) else name[:len(key)]
            for name in dataset_names
        )
        examples = set(
            ex if not ex.startswith(key) else ex[:len(key)]
            for ex in examples
        )
        test_names = set(
            test if not test.startswith(key) else test[:len(key)]
            for test in test_names
        )
    assert tests == caches, 'CircleCI tests != CircleCI caches'
    assert tests == examples, 'CircleCI tests != docs/mkdocs.yml Examples'
    assert tests == dataset_names, 'CircleCI tests != tests/datasets.py'
    assert tests == test_names, 'CircleCI tests != tests/test_run.py'
