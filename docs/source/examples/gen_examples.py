#!/bin/env python

from collections import defaultdict
import contextlib
import logging
import shutil
from pathlib import Path
import sys
from typing import Union, Iterable

import mne_bids_pipeline
from mne_bids_pipeline._config_import import _import_config
import mne_bids_pipeline.tests.datasets
from mne_bids_pipeline.tests.test_run import TEST_SUITE
from mne_bids_pipeline.tests.datasets import DATASET_OPTIONS
from tqdm import tqdm

this_dir = Path(__file__).parent
root = Path(mne_bids_pipeline.__file__).parent.resolve(strict=True)
logger = logging.getLogger()



def _bool_to_icon(x: Union[bool, Iterable]) -> str:
    if x:
        return '✅'
    else:
        return '❌'


@contextlib.contextmanager
def _task_context(task):
    old_argv = sys.argv
    if task:
        sys.argv = [sys.argv[0], f'--task={task}']
    try:
        yield
    finally:
        sys.argv = old_argv


def _gen_demonstrated_funcs(example_config_path: Path) -> dict:
    """Generate dict of demonstrated functionality based on config."""
    # Here we use a defaultdict, and for keys that might vary across configs
    # we should use an `funcs[key] = funcs[key] or ...` so that we effectively
    # OR over all configs.
    funcs = defaultdict(lambda: False)
    tasks = ['']
    if example_config_path.stem == 'config_ERP_CORE':
        tasks[:] = ['N400', 'ERN', 'LRP', 'MMN', 'N2pc', 'N170', 'P3']
    for task in tasks:
        with _task_context(task):
            config = _import_config(
                config_path=example_config_path,
                overrides=None,
                check=False,
                log=False,
            )
        ch_types = [c.upper() for c in config.ch_types]
        funcs['MEG processing'] = "MEG" in ch_types
        funcs['EEG processing'] = "EEG" in ch_types
        key = 'Maxwell filter'
        funcs[key] = funcs[key] or config.use_maxwell_filter
        funcs['Frequency filter'] = config.l_freq or config.h_freq
        key = 'SSP'
        funcs[key] = funcs[key] or (config.spatial_filter == "ssp")
        key = 'ICA'
        funcs[key] = funcs[key] or (config.spatial_filter == "ica")
        funcs['Evoked contrasts'] = config.contrasts
        any_decoding = config.decode and config.contrasts
        key = 'Time-by-time decoding'
        funcs[key] = funcs[key] or any_decoding
        key = 'Time-generalization decoding'
        funcs[key] = funcs[key] or (
            any_decoding and
            config.decoding_time_generalization
        )
        key = 'CSP decoding'
        funcs[key] = funcs[key] or (
            any_decoding and
            config.decoding_csp
        )
        funcs['Time-frequency analysis'] = config.time_frequency_conditions
        funcs['BEM surface creation'] = config.recreate_bem
        funcs['Template MRI'] = config.use_template_mri
    return funcs


# Copy reports to the correct place.
datasets_without_html = []
logger.warning(f'  Copying reports to {this_dir}/<dataset_name> …')
for test_name, test_dataset_options in TEST_SUITE.items():
    if 'ERP_CORE' in test_name:
        dataset_name = test_dataset_options['dataset']
    else:
        dataset_name = test_name
    del test_dataset_options

    example_target_dir = this_dir / dataset_name
    example_target_dir.mkdir(exist_ok=True)

    example_source_dir = Path(
        f'~/mne_data/derivatives/mne-bids-pipeline/{dataset_name}'
    ).expanduser()

    html_report_fnames = list(example_source_dir.rglob('*.html'))

    if not html_report_fnames:
        datasets_without_html.append(dataset_name)
        continue

    fname_iter = tqdm(
        html_report_fnames,
        desc=f'  {test_name}',
        unit='file',
        leave=False,
    )
    for fname in fname_iter:
        shutil.copy(src=fname, dst=example_target_dir)

# Now, generate the respective markdown example descriptions.
all_demonstrated = dict()
logger.warning('  Generating example markdown files …')
ds_iter = tqdm(
    list(TEST_SUITE.items()),
    desc='  ',
    unit='file',
    leave=False,
)
for test_dataset_name, test_dataset_options in ds_iter:
    if 'ERP_CORE' in test_dataset_name:
        dataset_name = test_dataset_options['dataset']
    else:
        dataset_name = test_dataset_name

    dataset_options_key = test_dataset_options.get(
        'dataset', test_dataset_name.split('_')[0])
    if dataset_name in all_demonstrated:
        logger.warning(
            f'Duplicate dataset name {test_dataset_name} -> {dataset_name}, '
            'skipping')
        continue
    del test_dataset_options, test_dataset_name


    if dataset_name in datasets_without_html:
        logger.warning(f'Dataset {dataset_name} has no HTML report.')
        continue

    options = DATASET_OPTIONS[dataset_options_key]

    report_str = '\n## Generated output\n\n'
    example_target_dir = this_dir / dataset_name

    fnames_reports = sorted(
        [f for f in example_target_dir.glob('*.html')
         if 'proc-clean' not in f.name and
         'proc-ica' not in f.name and 'proc-ssp' not in f.name]
    )

    fnames_cleaning = sorted(
        [f for f in example_target_dir.glob('*')
         if 'proc-clean' in f.name or
         'proc-ica' in f.name or 'proc-ssp' in f.name]
    )

    fnames_other = sorted(
        set(example_target_dir.glob('*')) -
        set(fnames_cleaning) -
        set(fnames_reports)
    )

    report_str += '???+ info "Summary reports"\n'
    for fname in fnames_reports:
        link_target = Path(dataset_name) / fname.name
        report_str += (f'    <a href="{link_target}" target="_blank" '
                       f'class="report-button md-button md-button--primary">'
                       f'{fname.name} :fontawesome-solid-square-poll-vertical:</a>\n\n')

    if fnames_cleaning:
        report_str += '??? info "Data cleaning"\n'
    for fname in fnames_cleaning:
        link_target = Path(dataset_name) / fname.name
        report_str += (f'    <a href="{link_target}" target="_blank" '
                       f'class="report-button md-button md-button--primary">'
                       f'{fname.name} :fontawesome-solid-square-poll-vertical:</a>\n\n')

    if fnames_other:
        report_str += '??? info "Other output"\n'
    for fname in fnames_other:
        link_target = Path(dataset_name) / fname.name
        report_str += (f'    <a href="{link_target}" target="_blank" '
                       f'class="report-button md-button md-button--primary">'
                       f'{fname.name} :fontawesome-solid-square-poll-vertical:</a>\n\n')

    if options['openneuro']:
        url = f'https://openneuro.org/datasets/{options["openneuro"]}'
    elif options['git']:
        url = options['git']
    elif options['web']:
        url = options['web']
    else:
        url = ''

    source_str = (f'## Dataset source\n\nThis dataset was acquired from '
                  f'[{url}]({url})\n')

    if options['openneuro']:
        download_str = (
            f'\n??? example "How to download this dataset"\n'
            f'    Run in your terminal:\n'
            f'    ```shell\n'
            f'    openneuro-py download \\\n'
            f'                 --dataset={options["openneuro"]} \\\n')
        for count, include in enumerate(options['include'], start=1):
            download_str += f'                 --include={include}'
            if count < len(options['include']) or options['exclude']:
                download_str += ' \\\n'
        for count, exclude in enumerate(options['exclude'], start=1):
            download_str += f'                 --exclude={exclude}'
            if count < len(options['exclude']):
                download_str += ' \\\n'
        download_str += '\n    ```\n'

        if options['exclude']:
            download_str += ("\nNote that we have to explicitly exclude\n"
                             "files due to a problem with OpenNeuro's storage."
                             "\n")
    else:
        download_str = ''

    # TODO: For things like ERP_CORE_ERN, decoding_csp are not populated
    # properly by the root config
    config_path = root / 'tests' / 'configs' / f'config_{dataset_name}.py'
    config = config_path.read_text(encoding='utf-8-sig').strip()
    descr_end_idx = config[2:].find('"""')
    config_descr = '# ' + config[:descr_end_idx+1].replace('"""', '').strip()
    config_descr += '\n\n'
    config_options = config[descr_end_idx+1:].replace('"""', '').strip()
    config_str = (f'\n## Configuration\n\n'
                  f'<details><summary>Click to expand</summary>\n\n'
                  f'```python\n'
                  f'{config_options}\n'
                  f'```\n'
                  f'</details>\n\n')
    demonstrated_funcs = _gen_demonstrated_funcs(config_path)
    all_demonstrated[dataset_name] = demonstrated_funcs
    del config, config_options
    funcs = ['## Demonstrated features\n',
             'Feature | This example',
             '--------|:-----------:']
    for key, val in demonstrated_funcs.items():
        funcs.append(f'{key} | {_bool_to_icon(val)}')
    demonstrated_funcs_str = '\n'.join(funcs) + '\n\n'

    out_path = this_dir / f'{dataset_name}.md'
    with out_path.open('w', encoding='utf-8') as f:
        f.write(config_descr)
        f.write(demonstrated_funcs_str)
        f.write(source_str)
        f.write(download_str)
        f.write(config_str)
        f.write(report_str)

# Finally, write our examples.html file

_example_header = """\
# Examples

Here you will find a number of examples using publicly available
datasets, mostly taken from [OpenNeuro](https://openneuro.org).

For a first example, see the results obtained with the
[MNE sample dataset](ds000248_base.md).

## Demonstrated features

"""

out_path = this_dir / 'examples.md'
with out_path.open('w', encoding='utf-8') as f:
    f.write(_example_header)
    header_written = False
    for dataset_name, funcs in all_demonstrated.items():
        if not header_written:
            f.write('Dataset | ' + ' | '.join(funcs.keys()) + '\n')
            f.write('--------|' + '|'.join([':---:'] * len(funcs)) + '\n')
            header_written = True
        f.write(f'[{dataset_name}]({dataset_name}.md) | ' +
                ' | '.join(_bool_to_icon(v) for v in funcs.values()) + '\n')
