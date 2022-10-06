#!/bin/bash -ef

# skip job if not in a [circle full] or on main
export REGEXP="r'(?<=\[circle )(.*)(?=\])'"
export COMMIT_MESSAGE=$(git log --format=oneline -n 1);
COMMIT_MESSAGE_ESCAPED=${COMMIT_MESSAGE//\'/\\\'}  # escape '
export COMMIT_MESSAGE_ESCAPED=${COMMIT_MESSAGE_ESCAPED//\"/\\\'}  # escape "
export CIRCLE_REQUESTED_JOB=$(
    python -c "import re; matches = re.findall(pattern=${REGEXP}, string='${COMMIT_MESSAGE_ESCAPED}'); match = '' if not matches else matches[0]; print(match)"
);

echo "CIRCLE_JOB=$CIRCLE_JOB"
echo "COMMIT_MESSAGE=$COMMIT_MESSAGE"
echo "COMMIT_MESSAGE_ESCAPED=$COMMIT_MESSAGE_ESCAPED"
echo "CIRCLE_REQUESTED_JOB=$CIRCLE_REQUESTED_JOB"

# Leave here for reference. This only runs all jobs upon [circle full] in the
# commit message, or jobs that were explicitly requersted.
#
# On a PR, only run setup_env, build_docs, and the requested job(s). If no
# jobs have been specifically requested, run ds000247.
# if [[
#     -v CIRCLE_PULL_REQUEST &&
#     "$CIRCLE_REQUESTED_JOB" != "full" &&
#     "$CIRCLE_JOB" != "setup_env" &&
#     "$CIRCLE_JOB" != "build_docs"
#    ]] && [[
#        (-n $CIRCLE_REQUESTED_JOB &&  # Specific job requested -> run only that one
#         "$CIRCLE_JOB" != *"$CIRCLE_REQUESTED_JOB"*) ||
#        (-z $CIRCLE_REQUESTED_JOB &&  # no specific job requested -> run only ds000247
#         "$CIRCLE_JOB" != *"ds000247")
#     ]]; then
#         echo "Skip detected, exiting job ${CIRCLE_JOB} for PR ${CIRCLE_PULL_REQUEST}."
#         circleci-agent step halt
# # Otherwise, run everything
# elif [[ -v CIRCLE_PULL_REQUEST ]]; then
#     echo "Running job ${CIRCLE_JOB} for PR ${CIRCLE_PULL_REQUEST}"
# else
#     echo "Running job ${CIRCLE_JOB} for ${CIRCLE_BRANCH} branch"
# fi

# On a PR, only run setup_env, build_docs, and the requested job(s). If no
# jobs have been specifically requested via [circle jobname], run all tests.
if [[
    -v CIRCLE_PULL_REQUEST &&
    "$CIRCLE_JOB" != "setup_env" &&
    "$CIRCLE_JOB" != "build_docs"
   ]] && [[
        -n $CIRCLE_REQUESTED_JOB &&  # Specific job requested -> run only that one
        "$CIRCLE_JOB" != *"$CIRCLE_REQUESTED_JOB"*
    ]]; then
        echo "Skip detected, exiting job ${CIRCLE_JOB} for PR ${CIRCLE_PULL_REQUEST}."
        circleci-agent step halt
# Otherwise, run everything
elif [[ -v CIRCLE_PULL_REQUEST ]]; then
    echo "Running job ${CIRCLE_JOB} for PR ${CIRCLE_PULL_REQUEST}"
else
    echo "Running job ${CIRCLE_JOB} for ${CIRCLE_BRANCH} branch"
fi

# Set up image
sudo ln -s /usr/lib/x86_64-linux-gnu/libxcb-util.so.0 /usr/lib/x86_64-linux-gnu/libxcb-util.so.1
wget -q -O- http://neuro.debian.net/lists/focal.us-tn.libre | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9
# Xvfb from upstream
wget -q https://raw.githubusercontent.com/mne-tools/mne-python/main/tools/setup_xvfb.sh
bash setup_xvfb.sh
# commands from upstream
wget -q https://raw.githubusercontent.com/mne-tools/mne-python/main/tools/get_minimal_commands.sh
source get_minimal_commands.sh
# Similar CircleCI setup to upstream
sudo apt install -qq tcsh git-annex-standalone python3.10-venv python3-venv
python3.10 -m venv ~/python_env
mkdir -p ~/mne_data
echo "set -e" >> $BASH_ENV;
echo 'export OPENBLAS_NUM_THREADS=3' >> $BASH_ENV;
echo 'shopt -s globstar' >> $BASH_ENV;  # Enable recursive globbing via **
echo 'export MNE_DATA=$HOME/mne_data' >> $BASH_ENV;
echo "export PATH=~/.local/bin/:$PATH" >> $BASH_ENV
echo 'export DISPLAY=:99' >> $BASH_ENV;
echo 'export XDG_RUNTIME_DIR=/tmp/runtime-circleci' >> $BASH_ENV;
echo 'export MPLBACKEND=Agg' >> $BASH_ENV;
echo "source ~/python_env/bin/activate" >> $BASH_ENV
mkdir -p ~/.local/bin
ln -s ~/python_env/bin/python ~/.local/bin/python

# Disable fancy 3D rendering options
echo 'export MNE_3D_OPTION_ANTIALIAS=false' >> $BASH_ENV;
echo 'export MNE_3D_OPTION_DEPTH_PEELING=false' >> $BASH_ENV;
echo 'export MNE_3D_OPTION_SMOOTH_SHADING=false' >> $BASH_ENV;

wget -q https://raw.githubusercontent.com/mne-tools/mne-python/main/tools/get_minimal_commands.sh
source get_minimal_commands.sh
mkdir -p ~/mne_data

# start xvfb if testing
if [[ "$CIRCLE_JOB" == "test_"* ]] || [[ "$CIRCLE_JOB" == "setup_env" ]] || [[ "$CIRCLE_JOB" == "build_docs" ]]; then
    echo "Starting Xvfb for ${CIRCLE_JOB}"
    /sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset -nolisten tcp -nolisten unix
fi

echo "export RUN_TESTS=\"python mne_bids_pipeline/tests/run_tests.py\"" >> $BASH_ENV
