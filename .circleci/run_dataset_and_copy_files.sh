#!/bin/bash

set -eo pipefail

DS_RUN=$1
if [[ "$2" == "" ]]; then
  DS="$DS_RUN"
else
  DS="$2"
fi
pytest mne_bids_pipeline --junit-xml=test-results/junit-results.xml -k ${DS_RUN}
mkdir -p ~/reports/${DS}
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/*/*.html ~/reports/${DS}/
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/*/*.json ~/reports/${DS}/
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/*/*.tsv ~/reports/${DS}/
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*.xlsx ~/reports/${DS}/
ls -al test-results/*.xml
