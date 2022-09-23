import os
import shutil
import sys
from pathlib import Path
import runpy
import logging
from typing import Union, Iterable

this_dir = Path(__file__).parent
root = this_dir.parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

logger = logging.getLogger()


dataset_opts_path = root / 'tests' / 'datasets.py'
run_tests_path = root / 'tests' / 'run_tests.py'

dataset_options = runpy.run_path(dataset_opts_path)['DATASET_OPTIONS']
test_options = runpy.run_path(run_tests_path)['TEST_SUITE']


def _bool_to_icon(x: Union[bool, Iterable]) -> str:
    if x:
        return '✅'
    else:
        return '❌'


def _gen_demonstrated_funcs(example_config_path: Path) -> dict:
    """Generate dict of demonstrated functionality based on config."""
    env = os.environ
    env['MNE_BIDS_STUDY_CONFIG'] = str(example_config_path.expanduser())

    # Set one of the various tasks for ERP CORE, as we currently raise if none
    # was provided
    if example_config_path.name == 'config_ERP_CORE.py':
        env['MNE_BIDS_STUDY_TASK'] = 'N400'

    example_config = runpy.run_path(example_config_path)
    env['BIDS_ROOT'] = example_config['bids_root']

    config_module_path = root / 'config.py'
    config = runpy.run_path(config_module_path)

    ch_types = [c.upper() for c in config['ch_types']]
    funcs = dict()
    funcs['MEG processing'] = "MEG" in ch_types
    funcs['EEG processing'] = "EEG" in ch_types
    funcs['Maxwell filter'] = config["use_maxwell_filter"]
    funcs['Frequency filter'] = config["l_freq"] or config["h_freq"]
    funcs['SSP'] = config["spatial_filter"] == "ssp"
    funcs['ICA'] = config["spatial_filter"] == "ica"
    funcs['Evoked contrasts'] = config["contrasts"]
    funcs['Time-by-time decoding'] = config["decode"] and config["contrasts"]
    funcs['Time-generalization decoding'] = \
        config["decoding_time_generalization"] and config["contrasts"]
    funcs['Time-frequency analysis'] = config["time_frequency_conditions"]
    funcs['BEM surface creation'] = config["recreate_bem"]
    funcs['Template MRI'] = config["use_template_mri"]
    return funcs


# Copy reports to the correct place.
datasets_without_html = []
for test_name, test_dataset_options in test_options.items():
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

    for fname in html_report_fnames:
        logger.info(f'Copying {fname} to {example_target_dir}')
        shutil.copy(src=fname, dst=example_target_dir)

# Now, generate the respective markdown example descriptions.
all_demonstrated = dict()
for test_dataset_name, test_dataset_options in test_options.items():
    if 'ERP_CORE' in test_dataset_name:
        dataset_name = test_dataset_options['dataset']
    else:
        dataset_name = test_dataset_name

    dataset_options_key = test_dataset_options.get(
        'dataset', test_dataset_name.split('_')[0])
    del test_dataset_options, test_dataset_name

    if dataset_name in datasets_without_html:
        logger.warning(f'Dataset {dataset_name} has no HTML report.')
        continue

    logger.warning(f'Generating markdown file for dataset: {dataset_name}')

    options = dataset_options[dataset_options_key]

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

    config_path = root / 'tests' / 'configs' / f'config_{dataset_name}.py'
    config = config_path.read_text(encoding='utf-8-sig').strip()
    descr_end_idx = config[2:].find('"""')
    config_descr = '# ' + config[:descr_end_idx+1].replace('"""', '').strip()
    config_descr += '\n\n'
    config_options = config[descr_end_idx+1:].replace('"""', '').strip()
    config_str = (f'\n## Configuration\n\n'
                  f'```python\n'
                  f'{config_options}\n'
                  f'```\n')
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
[MNE sample dataset](ds000248.md).
"""

out_path = this_dir / 'examples.md'
with out_path.open('w', encoding='utf-8') as f:
    f.write(_example_header)

# Eventually we could add a table with code like the following (but the flow
# is not great / it's too big currently):
#
#     header_written = False
#     for dataset_name, funcs in all_demonstrated.items():
#         if not header_written:
#             f.write('Dataset | ' + ' | '.join(funcs.keys()) + '\n')
#             f.write('--------|' + '|'.join([':---:'] * len(funcs)) + '\n')
#             header_written = True
#         f.write(f'[{dataset_name}]({dataset_name}.md) | ' +
#                 ' | '.join(_bool_to_icon(v) for v in funcs.values()) + '\n')
