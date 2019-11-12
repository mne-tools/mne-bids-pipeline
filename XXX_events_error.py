#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:00:05 2019

@author: sh254795
"""

# problem: with minimal event duration  = 0.002 not all events are found
# with lower durations, a lot more events are found, but the epocheing function
# doesn't like it either

Input:  /neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/2019_MEG_Pipeline/BIDS/derivatives/CBD_pipeline/sub-SB04/meg/sub-SB04_task-Localizer_sss_raw.fif
Opening raw data file /neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/2019_MEG_Pipeline/BIDS/derivatives/CBD_pipeline/sub-SB04/meg/sub-SB04_task-Localizer_sss_raw.fif...
    Range : 9000 ... 143999 =     18.000 ...   287.998 secs
Ready.
Current compensation grade : 0
Reading 0 ... 134999  =      0.000 ...   269.998 secs...
  Concatenating runs
Used Annotations descriptions: ['coherent/up', 'incoherent/1', 'incoherent/2']
  Epoching

ipdb> Traceback (most recent call last):

  File "<ipython-input-94-6df42cc8cbdc>", line 1, in <module>
    debugfile('/home/sh254795/Documents/REPOS/mne-study-template/04-make_epochs.py', wdir='/home/sh254795/Documents/REPOS/mne-study-template')

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/spyder_kernels/customize/spydercustomize.py", line 856, in debugfile
    debugger.run("runfile(%r, args=%r, wdir=%r)" % (filename, args, wdir))

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/bdb.py", line 434, in run
    exec(cmd, globals, locals)

  File "<string>", line 1, in <module>

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/spyder_kernels/customize/spydercustomize.py", line 827, in runfile
    execfile(filename, namespace)

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/spyder_kernels/customize/spydercustomize.py", line 110, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)

  File "/home/sh254795/Documents/REPOS/mne-study-template/04-make_epochs.py", line 135, in <module>
    main()

  File "/home/sh254795/Documents/REPOS/mne-study-template/04-make_epochs.py", line 131, in main
    itertools.product(config.subjects_list, config.sessions))

  File "/home/sh254795/Documents/REPOS/mne-study-template/04-make_epochs.py", line 130, in <genexpr>
    parallel(run_func(subject, session) for subject, session in

  File "/home/sh254795/Documents/REPOS/mne-study-template/04-make_epochs.py", line 103, in run_epochs
    reject=config.reject)

  File "</home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/mne/externals/decorator.py:decorator-gen-204>", line 2, in __init__

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/mne/utils/_logging.py", line 89, in wrapper
    return function(*args, **kwargs)

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/mne/epochs.py", line 1812, in __init__
    verbose=verbose)

  File "</home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/mne/externals/decorator.py:decorator-gen-195>", line 2, in __init__

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/mne/utils/_logging.py", line 89, in wrapper
    return function(*args, **kwargs)

  File "/home/sh254795/anaconda3/envs/mne/lib/python3.6/site-packages/mne/epochs.py", line 312, in __init__
    raise RuntimeError('Event time samples were not unique')

RuntimeError: Event time samples were not unique