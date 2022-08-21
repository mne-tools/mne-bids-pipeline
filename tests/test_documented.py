"""Test that all config values are documented."""
import ast
from pathlib import Path
import os

root_path = Path(__file__).parent.parent


def test_documented():
    """Test that all options are suitably documented."""
    # use ast to parse config.py for assignments
    with open(root_path / 'config.py', 'r') as fid:
        contents = fid.read()
    contents = ast.parse(contents)
    in_config = [
        item.target.id for item in contents.body
        if isinstance(item, ast.AnnAssign)]
    assert len(set(in_config)) == len(in_config)
    in_config = set(in_config)
    settings_path = root_path / 'docs' / 'source' / 'settings'
    in_doc = set()
    key = '::: config.'
    allowed_duplicates = set([
        'source_info_path_update',
    ])
    for dirpath, _, fnames in os.walk(settings_path):
        for fname in fnames:
            if not fname.endswith('.md'):
                continue
            # This is a .md file
            with open(Path(dirpath) / fname, 'r') as fid:
                for line in fid:
                    if not line.startswith(key):
                        continue
                    # The line starts with our magic key
                    val = line[len(key):].strip()
                    if val not in allowed_duplicates:
                        assert val not in in_doc, 'Duplicate documentation'
                    in_doc.add(val)
    assert in_doc.difference(in_config) == set(), 'Extra values in doc'
    assert in_config.difference(in_doc) == set(), 'Values missing from doc'
