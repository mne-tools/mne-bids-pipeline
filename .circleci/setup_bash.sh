#!/bin/bash -ef

# skip job if not in a [circle full] or on main
export COMMIT_MESSAGE=$(git log --format=oneline -n 1);
echo "CIRCLE_JOB=$CIRCLE_JOB"
echo "COMMIT_MESSAGE=$COMMIT_MESSAGE"

# On a PR, only run setup_env, build_docs, and ds00247
if [[ -v CIRCLE_PULL_REQUEST ]] &&
    [[ "$COMMIT_MESSAGE" != *"[circle full]"* ]] &&
    [[ "$CIRCLE_JOB" != *"ds000247" ]] &&
    [[ "$CIRCLE_JOB" != "setup_env" ]] &&
    [[ "$CIRCLE_JOB" != "build_docs" ]]; then
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
wget -q -O- http://neuro.debian.net/lists/buster.us-tn.libre | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9
sudo apt-get -qq update
sudo apt install -qq tcsh git-annex-standalone libosmesa6 libglx-mesa0 libopengl0 libglx0 libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0
echo "set -e" >> $BASH_ENV;
echo "export OPENBLAS_NUM_THREADS=4" >> $BASH_ENV;
echo "shopt -s globstar" >> $BASH_ENV;  # Enable recursive globbing via **
echo "export PATH=~/.local/bin:$PATH" >> $BASH_ENV;
PATH=~/.local/bin:$PATH
echo "export MNE_DATA=/home/circleci/mne_data" >> $BASH_ENV;
echo "export DISPLAY=:99" >> $BASH_ENV;
echo "export XDG_RUNTIME_DIR=/tmp/runtime-circleci" >> $BASH_ENV
wget -q https://raw.githubusercontent.com/mne-tools/mne-python/main/tools/get_minimal_commands.sh
source get_minimal_commands.sh
mkdir -p ~/mne_data

# start xvfb if testing
if [[ "$CIRCLE_JOB" == "test_"* ]] || [[ "$CIRCLE_JOB" == "setup_env" ]] || [[ "$CIRCLE_JOB" == "build_docs" ]]; then
    echo "Starting Xvfb for ${CIRCLE_JOB}"
    /sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset -nolisten tcp -nolisten unix
fi