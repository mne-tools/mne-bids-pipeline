#!/bin/bash
set -exo pipefail

pip install --upgrade --progress-bar off pip setuptools
pip install --upgrade --progress-bar off -v \
    "autoreject @ https://api.github.com/repos/autoreject/autoreject/zipball/master" \
    "mne[hdf5] @ https://api.github.com/repos/mne-tools/mne-python/zipball/main" \
    "mne-bids[full] @ https://api.github.com/repos/mne-tools/mne-bids/zipball/main" \
    numba \
    onnxruntime \
    PySide6 \
    awscli \
    -e .[tests]
