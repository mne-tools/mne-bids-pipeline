[![CircleCI](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline.svg?style=svg)](https://circleci.com/gh/brainthemind/CogBrainDyn_MEG_Pipeline)

# 0 Credits 

This example pipeline for MEG/EEG data processing with MNE python was build jointly by the [Cognition and Brain Dynamics Team] (https://brainthemind.com/) and the [MNE Python Team](https://martinos.org/mne/stable/index.html),
based on scripts originally developed for this publication:

Jas, M., Larson, E., Engemann, D. A., Leppäkangas, J., Taulu, S., Hämäläinen, M., & Gramfort, A. (2018). A reproducible MEG/EEG group study with the MNE software: recommendations, quality assessments, and good practices. Frontiers in neuroscience, 12. 


# 1 Make sure MNE-python is installed

First, you need to make sure you have mne-python installed and working on your system. See [installation instructions](http://martinos.org/mne/stable/install_mne_python.html). Once is done, you should be able to run this in a terminal:

	$ python -c "import mne; mne.sys_info()"

Get the scripts through git:

	$ git clone https://github.com/mne-tools/mne-study-template.git
	
If you do not know how to use git, download the scripts [here](https://github.com/mne-tools/mne-study-template/archive/master.zip). 

For source analysis you'll also need freesurfer, follow the instructions on [their website](https://surfer.nmr.mgh.harvard.edu/).


# 2 Set your data to a proper place

In our example, we will use .fif raw data you can find here:
https://osf.io/m9nwz/

The name of the study will be "Localizer".

Let's create a folder called "ExampleData" wherever you want on your computer.

In the ExampleData folder, you need to create three subfolders:  "MEG", "system_calibration_files" and "subjects" as follow:

![xx](https://image.noelshack.com/fichiers/2019/15/4/1554998135-path.png)


The "MEG" folder will contain a folder for each participant
The "system_calibration_files" folder will contain the calibration files (download them from OSF)
The "subjects" folder will contain participant MRI files.

Here is an example of what the MEG folder should contain:

![xx](https://image.noelshack.com/fichiers/2019/15/4/1554998137-path1.png)

Then you put the raw data for each subject in their own folder. The raw data file name should respect this format:
subjectID_StudyName_raw.fif

or, if your data has multiple runs:
subjectID_StudyNamerun01_raw.fif


![xx](https://image.noelshack.com/fichiers/2019/15/4/1554998137-path2.png)

# 3 Adapt config.py

All specific settings to be used in your analysis are defined in [config.py](config.py).
See the comments for explanations and recommendations. 


# 4 Preprocessing steps

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

[05b-run_ssp.py](05a-run_ica.py) ->
As an alternative to ICA, you can use SSP projections to correct for eye blink and heart artifacts. 
Use either 5a/6a, or 5b/6b. 

[06b-apply_ssp.py](06a-apply_ica.py) ->
Apply ssp projections and obtain the cleaned epochs. 

[07-make_evoked.py](07-make_evoked.py) ->
Extract evoked data for each condition.

[08-group_average_sensors.py](08-group_average_sensors.py) ->
Make a group average of the time domain data.

