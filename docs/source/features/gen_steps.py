#!/bin/env python
"""Generate steps.md."""

import importlib
import os
from pathlib import Path
import sys

pre = """\
# Processing steps

The following table provides a concise summary of each step in the Study
Template. All scripts exist in the `scripts`/ directory.
"""

this_dir = Path(__file__).parent
config_dir = this_dir.parent.parent.parent.resolve(strict=True)
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))
from run import _get_script_modules
# We need to provide some valid config
os.environ['_MNE_BIDS_PIPELINE_STRICT_RESOLVE'] = 'false'
config_path = str(config_dir / 'tests' / 'configs' / 'config_ds000248.py')
script_modules = _get_script_modules(config=config_path)

# Construct the lines of steps.md
lines = [pre]
for di, (dir_, modules) in enumerate(script_modules.items(), 1):
    if dir_ == 'all':
        continue  # this is an alias
    dir_module = importlib.import_module(f'scripts.{dir_}')
    dir_header = dir_module.__doc__.split('\n')[0].rstrip('.')
    dir_body = dir_module.__doc__.split('\n', maxsplit=1)
    if len(dir_body) > 1:
        dir_body = dir_body[1].strip()
    else:
        dir_body = ''
    lines.append(f'## {di}. {dir_header}\n')
    if dir_body:
        lines.append(f'{dir_body}\n')
    lines.append('| Processing step | Description |')
    lines.append('|:----------------|:------------|')
    # the "all" option
    dir_name, script_title = dir_, f'Run all {dir_header.lower()} steps.'
    lines.append(f'`{dir_name}` | {script_title} |')
    for module in modules:
        script_name = f'{dir_name}/{Path(module.__file__).name}'[:-3]
        script_title = module.__doc__.split('\n')[0]
        lines.append(f'`{script_name}` | {script_title} |')
    lines.append('')
with open(Path(__file__).parent / 'steps.md', 'w') as fid:
    fid.write('\n'.join(lines))
