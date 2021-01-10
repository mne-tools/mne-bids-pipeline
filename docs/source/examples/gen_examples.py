from pathlib import Path
import runpy

download_test_data_path = Path('tests/download_test_data.py')
run_tests_path = Path('tests/run_tests.py')

dataset_options = runpy.run_path(download_test_data_path)['DATASET_OPTIONS']
example_datasets = sorted(runpy.
                          run_path(run_tests_path)['TEST_SUITE']
                          .keys())

for dataset_name in example_datasets:
    options = dataset_options[dataset_name]


    # out_path = Path(f'docs/source/examples/{dataset_name}.md')
    # with open(out_path, 'w', encoding='utf-8') as f:
    #     f.write('Hallo\n')


"""
Dataset source: [https://openneuro.org/datasets/ds000248](https://openneuro.org/datasets/ds000248)

??? example "How to download this dataset"
    Run in your terminal:

    ```shell
    openneuro-py download \
                 --dataset=ds000248 \
                 --include=sub-01 \
                 --include=sub-emptyroom \
                 --include=derivatives/freesurfer/subjects \
                 --exclude=derivatives/freesurfer/subjects/fsaverage/mri/aparc.a2005s+aseg.mgz \
                 --exclude=derivatives/freesurfer/subjects/fsaverage/mri/aparc+aseg.mgz \
                 --exclude=derivatives/freesurfer/subjects/fsaverage/mri/aparc.a2009s+aseg.mgz \
                 --exclude=derivatives/freesurfer/subjects/fsaverage/xhemi/mri/aparc+aseg.mgz \
                 --exclude=derivatives/freesurfer/subjects/sub-01/mri/aparc+aseg.mgz \
                 --exclude=derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz \
                 --exclude=derivatives/freesurfer/subjects/sub-01/mri/aparc.DKTatlas+aseg.mgz \
                 --exclude=derivatives/freesurfer/subjects/sub-01/mri/aparc.a2009s+aseg.mgz
    ```
    Note that we have to explicitly exclude numerous files due to a problem
    with OpenNeuro's storage.

## Configuration

```python
study_name = 'ds000248'
subjects = ['01']
rename_events = {'Smiley': 'Emoji',
                 'Button': 'Switch'}
conditions = ['Auditory', 'Visual', 'Auditory/Left', 'Auditory/Right']
contrasts = [('Visual', 'Auditory'),
             ('Auditory/Right', 'Auditory/Left')]

ch_types = ['meg']
mf_reference_run = '01'
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
process_er = True
noise_cov = 'emptyroom'

bem_mri_images = 'FLASH'
recreate_bem = True
```

## Generated report

<div class="example-report">
    <iframe src="reports/ds000248/sub-01_task-audiovisual_report.html"
            title="Example report for ds000248"
            width="100%" height="600px">
    </iframe>
</div>
"""