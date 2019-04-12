[![CircleCI](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline.svg?style=svg)](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline)


1 Be sure MNE-python is installed
---------------------------------

First, you need to make sure you have mne-python installed and working on your system. See [installation instructions](http://martinos.org/mne/stable/install_mne_python.html) and once is done you should be able to run in a terminal this:

	$ python -c "import mne; mne.sys_info()"

Once you have mne-python installed on your machine you need the analysis script that you'll need to adjust to your need. You can [download](https://github.com/mne-tools/mne-study-template/archive/master.zip) the current version of these script, or get them through git:

	$ git clone https://github.com/mne-tools/mne-study-template.git

For source analysis you'll also need freesurfer, follow the instructions on [their website](https://surfer.nmr.mgh.harvard.edu/).


2 Set your data to a proper place
-------------------------------

In our example, we will use .fif raw data you can find here:
https://osf.io/m9nwz/

The name of the study will be "Localizer".

Let's create a folder called "Dynacomp_LocalizerData" wherever you want on your computer.

For exemple here:
'C:/Users/IE258305/Documents/Data/Dynacomp_LocalizerData/'

Then, in this Dynacomp_LocalizerData folder, you need to create three folders "MEG", "system_calibration_files" and "subjects" as follow:

![xx](https://image.noelshack.com/fichiers/2019/15/4/1554998135-path.png)


The "MEG" folder will contain a folder for each participant
The "system_calibration_files" folder will contain the calibration files (also on OSF)
The "MEG" folder will contain participant MRI files.

Here is an example of what the MEG folder should contain:

![xx](https://image.noelshack.com/fichiers/2019/15/4/1554998137-path1.png)

Then you put the raw data for each subject in their own folder. The raw data file name should respect this format:
subjectID_StudyName_raw.fif

![xx](https://image.noelshack.com/fichiers/2019/15/4/1554998137-path2.png)


Then, you will need to update the [config.py](config.py) file. This
file is meant to contain study specific parameters. 


3 Preprocessing steps
-------------------

[config.py](config.py) -> The only file you need to modify in principle. This file contain all your parameters. 

[01-import_and_filter.py](01-import_and_filter.py) ->
Read raw data and apply lowpass or/and highpass filtering.

[02-apply_maxwell_filter.py](02-apply_maxwell_filter.py) ->
Run maxfilter and do lowpass filter at 40 Hz.

[03-extract_events.py](03-extract_events.py) ->
Extract events or annotations or markers from the data and save it to disk. Uses events from stimulus channel STI101.

[04-make_epochs.py](04-make_epochs.py) ->
Extract epochs.

[05a-run_ica.py](05a-run_ica.py) ->
Run Independant Component Analysis (ICA) for artifact correction.

[06a-apply_ica.py](06a-apply_ica.py) ->
Blinks and ECG artifacts are automatically detected and the corresponding ICA components are removed from the data.

[07-make_evoked.py](07-make_evoked.py) ->
Extract evoked data for each condition.


