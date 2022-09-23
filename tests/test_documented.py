"""Test that all config values are documented."""
import ast
from pathlib import Path
import os
import sys
import yaml

test_dir_path = Path(__file__).parent
root_path = test_dir_path.parent

sys.path.insert(0, str(test_dir_path))

from datasets import DATASET_OPTIONS
from run_tests import TEST_SUITE


def test_options_documented():
    """Test that all options are suitably documented."""
    # use ast to parse config.py for assignments
    with open(root_path / "config.py", "r") as fid:
        contents = fid.read()
    contents = ast.parse(contents)
    in_config = [
        item.target.id for item in contents.body if isinstance(item, ast.AnnAssign)
    ]
    assert len(set(in_config)) == len(in_config)
    in_config = set(in_config)
    settings_path = root_path / "docs" / "source" / "settings"
    in_doc = set()
    key = "::: config."
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
                    val = line[len(key) :].strip()
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
    # 4. tests/run_tests.py:TEST_SUITE (imported above)
    #
    # So let's make sure they stay in sync.

    # 1. Read cache, test, etc. entries from CircleCI
    with open(root_path / '.circleci' / 'config.yml', 'r') as fid:
        circle_yaml = yaml.safe_load(fid.read())
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
    # TODO: Go through circle_yaml['workflows']['commit']['jobs'] and
    # make sure everything is consitent there. It's going to be a bit of a
    # pain but probably worth it to be safe...

    # 3. Read examples from docs (being careful about tags we can't read)
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(root_path / 'docs' / 'mkdocs.yml', 'r') as fid:
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
    assert tests == test_names, 'CircleCI tests != tests/run_tests.py'
