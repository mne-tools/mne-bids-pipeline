#!/bin/bash

set -exo pipefail
which python
openneuro-py --version
mri_convert --version
mne_bids --version
mne sys_info
