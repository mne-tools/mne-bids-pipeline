#!/bin/env python
"""Generate steps.md."""

import importlib
from pathlib import Path
from mne_bids_pipeline._config_utils import _get_step_modules

pre = """\
# Processing steps

The following table provides a concise summary of each step in the Study
Template. All steps exist in the `steps`/ directory.
"""

print('Generating steps …')
step_modules = _get_step_modules()

# Construct the lines of steps.md
lines = [pre]
for di, (dir_, modules) in enumerate(step_modules.items(), 1):
    if dir_ == 'all':
        continue  # this is an alias
    dir_module = importlib.import_module(f'mne_bids_pipeline.steps.{dir_}')
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
    dir_name, step_title = dir_, f'Run all {dir_header.lower()} steps.'
    lines.append(f'`{dir_name}` | {step_title} |')
    for module in modules:
        step_name = f'{dir_name}/{Path(module.__file__).name}'[:-3]
        step_title = module.__doc__.split('\n')[0]
        lines.append(f'`{step_name}` | {step_title} |')
    lines.append('')
with open(Path(__file__).parent / 'steps.md', 'w') as fid:
    fid.write('\n'.join(lines))
