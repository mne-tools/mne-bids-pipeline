#!/bin/bash -e

set -o pipefail

# skip job if not in a [circle full] or on main
export COMMIT_MESSAGE=$(git log --format=oneline -n 1);
COMMIT_MESSAGE_ESCAPED=${COMMIT_MESSAGE//\'/\\\'}  # escape '
export COMMIT_MESSAGE_ESCAPED=${COMMIT_MESSAGE_ESCAPED//\"/\\\'}  # escape "
export CIRCLE_REQUESTED_JOB=$(echo "$COMMIT_MESSAGE_ESCAPED" | sed -nE 's/^.*\[circle ([^]]+)\].*$/\1/p')

echo "CIRCLE_JOB=$CIRCLE_JOB"
echo "COMMIT_MESSAGE=$COMMIT_MESSAGE"
echo "COMMIT_MESSAGE_ESCAPED=$COMMIT_MESSAGE_ESCAPED"
echo "CIRCLE_REQUESTED_JOB=$CIRCLE_REQUESTED_JOB"

# On a PR
if [[ -v CIRCLE_PULL_REQUEST ]]; then
    # Skip if circle skip has been requested
    if [[ "$CIRCLE_JOB" != "setup_env" && ( "$COMMIT_MESSAGE" == *"[skip circle]"* || "$COMMIT_MESSAGE" == *"[circle skip]"* ) ]]; then
        echo "Skip detected, exiting job ${CIRCLE_JOB} for PR ${CIRCLE_PULL_REQUEST}."
        circleci-agent step halt
        exit 0
    # If no jobs have been specifically requested via [circle jobname], run all tests.
    elif [[ "$CIRCLE_JOB" != "setup_env" && "$CIRCLE_JOB" != "build_docs" && -n $CIRCLE_REQUESTED_JOB && "$CIRCLE_JOB" != *"$CIRCLE_REQUESTED_JOB"* ]]; then
        echo "Skip detected, exiting job ${CIRCLE_JOB} for PR ${CIRCLE_PULL_REQUEST}."
        circleci-agent step halt
        exit 0
    fi
    echo "Running job ${CIRCLE_JOB} for PR ${CIRCLE_PULL_REQUEST}"
else
    echo "Running job ${CIRCLE_JOB} for ${CIRCLE_BRANCH} branch"
fi

# Set up image
echo "export RUN_TESTS=\".circleci/run_dataset_and_copy_files.sh\"" >> "$BASH_ENV"
echo "export DOWNLOAD_DATA=\"coverage run -m mne_bids_pipeline._download\"" >> "$BASH_ENV"

# Similar CircleCI setup to mne-python (Xvfb, minimal commands, env vars)
wget -q https://raw.githubusercontent.com/mne-tools/mne-python/main/tools/setup_xvfb.sh
bash setup_xvfb.sh
sudo apt install -qq tcsh libxft2
wget -q https://raw.githubusercontent.com/mne-tools/mne-python/main/tools/get_minimal_commands.sh
source get_minimal_commands.sh
mkdir -p ~/mne_data
echo "set -e" >> "$BASH_ENV"
echo 'export OPENBLAS_NUM_THREADS=2' >> "$BASH_ENV"
echo 'shopt -s globstar' >> "$BASH_ENV"  # Enable recursive globbing via **
echo 'export MNE_DATA=$HOME/mne_data' >> "$BASH_ENV"
echo 'export DISPLAY=:99' >> "$BASH_ENV"
echo 'export XDG_RUNTIME_DIR=/tmp/runtime-circleci' >> "$BASH_ENV"
echo 'export MPLBACKEND=Agg' >> "$BASH_ENV"
echo "export MNE_3D_OPTION_MULTI_SAMPLES=1" >> "$BASH_ENV"
echo "export MNE_BIDS_PIPELINE_FORCE_TERMINAL=true" >> "$BASH_ENV"
echo "export FORCE_COLOR=1" >> "$BASH_ENV"  # for rich to use color in logs
