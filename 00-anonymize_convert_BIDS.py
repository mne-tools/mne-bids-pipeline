#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:54:21 2019

This script takes the raw data and sets up a BIDS structure in a specified 
repository (bids_root).

It reads values from the study config file. 

Have a look at the documentation here: https://mne.tools/mne-bids/index.html

@author: sh254795
"""


#%%
import glob
import os.path as op
 
import mne
from mne_bids import write_raw_bids, make_bids_basename, write_anat
 

from datetime import datetime
import re

import config

bids_root = '/neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/2019_MEG_Pipeline/BIDS/'
subjects_dir = '/neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/2019_MEG_Pipeline/MRI/derivatives/fsreconall/'
trans_dir =  '/neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/2019_MEG_Pipeline/MEG/trans/'

base_path = '/neurospin/meg/meg_tmp/Dynacomp_Ciuciu_2011/2019_MEG_Pipeline/MEG/'
subjects = ['SB01', 'SB02','SB03','SB04','SB05','SB06','SB07','SB08','SB09','SB10','SB11','SB12']


tasks = ['Localizer','empty'] # ,'empty'

do_anonymize = True
plot = True

for ss, subject in enumerate(subjects):
    t1w = subjects_dir + "/%s/mri/T1.mgz" % subject
    trans = trans_dir + '/%s/Coregistration-trans.fif' % subject
    # Take care of MEG
    for task in tasks:
        raw_fname = op.join(base_path,task,subject, '%s_raw.fif' % task)
        raw = mne.io.read_raw_fif(raw_fname,allow_maxshield=True)
        
        meas_date = datetime.fromtimestamp(
                    raw.info['meas_date'][0]).strftime('%Y%m%d')  
        
        if do_anonymize:
            raw.anonymize()
            
        raw.info['subject_info'] = dict(id=ss)

        # create basename
        bids_basename = make_bids_basename(subject=subject, task=task)
       
        # read bad channels
        bad_chans_file_name = op.join(base_path,task,subject,'bad_channels.txt')
        bad_chans_file = open(bad_chans_file_name,"r") 
        bad_chans = bad_chans_file.readlines()
        bad_chans_file.close()
        
        bads = []
        
        for i in  bad_chans:            
            if task in i:
                bads = re.findall(r'\d+|\d+.\d+', i)
                # print(bads)
        if bads:
            for b, bad in  enumerate(bads):
                bads[b] = 'MEG' + str(bad)
                
        raw.info['bads'] = bads
        print("added bads: ", raw.info['bads'])
        
        # read events
        if task == 'empty':    
            er_bids_basename = make_bids_basename(subject='emptyroom', task='noise', prefix=subject)       
            write_raw_bids(raw, er_bids_basename,
                           output_path=bids_root,
                           overwrite=True) 
        
        else:
          events = mne.find_events(raw, stim_channel=config.stim_channel,
                                 consecutive=True,
                                 min_duration=config.min_event_duration,
                                 shortest_event=config.shortest_event)
         
          write_raw_bids(raw, bids_basename,
                           output_path=bids_root,
                           events_data=events,
                           event_id = config.event_id,
                           overwrite=True)
        
        if task != 'empty':               
            # Take care of anatomy
            anat_dir = write_anat(bids_root, subject, t1w, acquisition="t1w",
                       trans=trans, raw=raw, overwrite=True)        
            
            
            #%% plot to check landmarks
            if plot :
               
                import numpy as np
                import matplotlib.pyplot as plt
    
                from nilearn.plotting import plot_anat
                from mne.source_space import head_to_mri
                from mne_bids import get_head_mri_trans
    
                # Get Landmarks from MEG file, 0, 1, and 2 correspond to LPA, NAS, RPA
                # and the 'r' key will provide us with the xyz coordinates
                pos = np.asarray((raw.info['dig'][0]['r'],
                                  raw.info['dig'][1]['r'],
                                  raw.info['dig'][2]['r']))
                
                bids_fname = bids_basename + '_meg.fif'
                estim_trans = get_head_mri_trans(bids_fname=bids_fname,  # name of the MEG file
                                     bids_root=bids_root  # root of our BIDS dir
                                     )
    
    
                
                # We use a function from MNE-Python to convert MEG coordinates to MRI space
                # for the conversion we use our estimated transformation matrix and the
                # MEG coordinates extracted from the raw file. `subjects` and `subjects_dir`
                # are used internally, to point to the T1-weighted MRI file: `t1_mgh_fname`
                mri_pos = head_to_mri(pos=pos,
                                      subject=subject,
                                      mri_head_t=estim_trans,
                                      subjects_dir=subjects_dir
                                      )
                
                # Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
                t1_nii_fname = op.join(anat_dir, 'sub-' + subject + '_acq-t1w_T1w.nii.gz')
                # sub-SB01_acq-t1w_T1w
                
                # Plot it
                fig, axs = plt.subplots(3, 1)
                for point_idx, label in enumerate(('LPA', 'NAS', 'RPA')):
                    plot_anat(t1_nii_fname, axes=axs[point_idx],
                              cut_coords=mri_pos[point_idx, :],
                              title=label)
                plt.show()

