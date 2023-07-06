#!/bin/bash

set -eo pipefail

DS_RUN=$1
if [[ "$2" == "" ]]; then
  DS="$DS_RUN"
else
  DS="$2"
fi
if [[ "$3" == "--no-copy" ]]; then
  COPY_FILES="false"
else
  COPY_FILES="true"
fi

pytest mne_bids_pipeline --junit-xml=test-results/junit-results.xml -k ${DS_RUN}

if [[ "$COPY_FILES" == "false" ]]; then
  exit 0
fi
mkdir -p ~/reports/${DS}
# these should always exist
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/**/*.html ~/reports/${DS}/
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*.xlsx ~/reports/${DS}/
# these are allowed to be optional
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/**/*.json ~/reports/${DS}/ || :
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/**/*.tsv ~/reports/${DS}/ || :
ls -al test-results/*.xml
