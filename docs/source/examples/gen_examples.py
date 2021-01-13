import os
import shutil
from pathlib import Path
import runpy
from typing import Union, Iterable


RUNNING_ON_CIRCLE = os.environ.get('CIRCLECI')
if RUNNING_ON_CIRCLE == 'true':
    RUNNING_ON_CIRCLE = True
else:
    RUNNING_ON_CIRCLE = False


dataset_opts_path = Path('tests/datasets.py')
run_tests_path = Path('tests/run_tests.py')

dataset_options = runpy.run_path(dataset_opts_path)['DATASET_OPTIONS']
example_datasets = sorted(runpy.
                          run_path(run_tests_path)['TEST_SUITE']
                          .keys())


def gen_demonstrated_funcs_str(example_config_path: Path) -> str:
    """Generate a list of the demonstrated functionality based on the config.
    """
    env = os.environ
    env['MNE_BIDS_STUDY_CONFIG'] = str(example_config_path.expanduser())

    example_config = runpy.run_path(example_config_path)
    env['BIDS_ROOT'] = example_config['bids_root']

    config_module_path = Path('config.py')
    config = runpy.run_path(config_module_path)

    def _bool_to_icon(x: Union[bool, Iterable]) -> str:
        if x:
            return '✅'
        else:
            return '❌'

    funcs = ['## Demonstrated features\n',
             'Feature | This example',
             '--------|:-----------:']

    ch_types = [c.upper() for c in config['ch_types']]
    funcs.append(f'MEG processing | {_bool_to_icon("MEG" in ch_types)}')
    funcs.append(f'EEG processing | {_bool_to_icon("EEG" in ch_types)}')
    funcs.append(f'Maxwell filter | '
                 f'{_bool_to_icon(config["use_maxwell_filter"])}')
    funcs.append(f'Frequency filter | '
                 f'{_bool_to_icon(config["l_freq"] or config["h_freq"])}')
    funcs.append(f'SSP | {_bool_to_icon(config["use_ssp"])}')
    funcs.append(f'ICA | {_bool_to_icon(config["use_ica"])}')
    funcs.append(f'Evoked contrasts | {_bool_to_icon(config["contrasts"])}')
    funcs.append(f'Time-by-time decoding | {_bool_to_icon(config["decode"])}')
    funcs.append(f'Time-frequency analysis | '
                 f'{_bool_to_icon(config["time_frequency_conditions"])}')
    funcs.append(f'BEM surface creation | '
                 f'{_bool_to_icon(config["recreate_bem"])}')

    funcs = '\n'.join(funcs) + '\n'
    return funcs


# Copy reports to the correct place.
datasets_without_html = []
for dataset_name in example_datasets:
    example_target_dir = Path(f'docs/source/examples/{dataset_name}')
    example_target_dir.mkdir(exist_ok=True)

    if RUNNING_ON_CIRCLE:
        example_source_dir = Path(f'~/reports/{dataset_name}').expanduser()
    else:
        example_source_dir = (Path(f'~/mne_data/{dataset_name}'
                                  f'/derivatives/mne-study-template')
                              .expanduser())
    html_report_fnames = list(example_source_dir.rglob('*.html'))

    if not html_report_fnames:
        datasets_without_html.append(dataset_name)
        continue

    for fname in html_report_fnames:
        shutil.copy(src=fname, dst=example_target_dir)

# Now, generate the respective markdown example descriptions.
for dataset_name in example_datasets:
    if dataset_name in datasets_without_html:
        continue

    options = dataset_options[dataset_name]

    report_str = '\n## Generated output\n\n'
    example_target_dir = Path(f'docs/source/examples/{dataset_name}')
    for fname in example_target_dir.glob('*'):
        link_target = Path(dataset_name) / fname.name
        report_str += (f'<a href="{link_target}" target="_blank" '
                       f'class="report-button md-button md-button--primary">'
                       f'{fname.name} :fontawesome-solid-poll:</a>\n\n')

    if options['openneuro']:
        url = f'https://openneuro.org/datasets/{options["openneuro"]}'
    else:
        url = options['git']
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

    config_path = Path(f'tests/configs/config_{dataset_name}.py')
    config = config_path.read_text(encoding='utf-8-sig').strip()
    descr_end_idx = config[2:].find('"""')
    config_descr = '# ' + config[:descr_end_idx+1].replace('"""', '').strip()
    config_descr += '\n\n'
    config_options = config[descr_end_idx+1:].replace('"""', '').strip()
    config_str = (f'\n## Configuration\n\n'
                  f'```python\n'
                  f'{config_options}\n'
                  f'```\n')
    demonstrated_funcs_str = gen_demonstrated_funcs_str(config_path)
    del config, config_options

    out_path = Path(f'docs/source/examples/{dataset_name}.md')
    with out_path.open('w', encoding='utf-8') as f:
        f.write(config_descr)
        f.write(demonstrated_funcs_str)
        f.write(report_str)
        f.write(source_str)
        f.write(download_str)
        f.write(config_str)
