#!/bin/bash

set -eo pipefail

COPY_FILES="true"
RERUN_TEST="true"
while getopts "cr" option; do
   echo $option
   case $option in
      c)
        COPY_FILES="false";;
      r)
        RERUN_TEST="false";;
   esac
done
shift "$(($OPTIND -1))"

DS_RUN=$1
if [[ -z $1 ]]; then
  echo "Missing dataset argument"
  exit 1
fi
if [[ "$DS_RUN" == "ERP_CORE_"* ]]; then
  DS="ERP_CORE"
else
  DS="$1"
fi

SECONDS=0
pytest mne_bids_pipeline --junit-xml=test-results/junit-results.xml -k ${DS_RUN}
echo "Runtime: ${SECONDS} seconds"

# rerun test (check caching)!
SECONDS=0
if [[ "$RERUN_TEST" == "false" ]]; then
  echo "Skipping rerun test"
  RUN_TIME=0
else
  pytest mne_bids_pipeline --cov-append -k $DS_RUN
  RUN_TIME=$SECONDS
  echo "Runtime: ${RUN_TIME} seconds (should be < 20)"
fi
test $RUN_TIME -lt 20

if [[ "$COPY_FILES" == "false" ]]; then
  echo "Not copying files"
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
