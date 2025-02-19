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
EMPH="\e[35m\e[1m"  # bold magenta foreground
RESET="\e[0m"
pytest mne_bids_pipeline --junit-xml=test-results/junit-results.xml -k ${DS_RUN}
# Add emphasis and echo
echo -e "${EMPH}Clean test runtime: ${SECONDS} seconds${RESET}"
echo

# rerun test (check caching)!
SECONDS=0
RERUN_LIMIT=60
if [[ "$RERUN_TEST" == "false" ]]; then
  echo -e "${EMPH}Skipping cache rerun test${RESET}"
  RUN_TIME=0
else
  pytest mne_bids_pipeline --cov-append -k $DS_RUN
  RUN_TIME=$SECONDS
  echo -e "${EMPH}Cached test runtime: ${RUN_TIME} seconds (should be <= $RERUN_LIMIT)${RESET}"
fi
test $RUN_TIME -le $RERUN_LIMIT

if [[ "$COPY_FILES" == "false" ]]; then
  echo -e "${EMPH}Not copying files${RESET}"
  exit 0
fi
echo
echo -e "${EMPH}Copying files${RESET}"
mkdir -p ~/reports/${DS}
# these should always exist
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/**/*.html ~/reports/${DS}/
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*.xlsx ~/reports/${DS}/
# these are allowed to be optional
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/**/*.json ~/reports/${DS}/ || :
cp -av ~/mne_data/derivatives/mne-bids-pipeline/${DS}/*/**/*.tsv ~/reports/${DS}/ || :
ls -al test-results/*.xml
