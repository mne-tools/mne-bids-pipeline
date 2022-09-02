FROM continuumio/miniconda3 as base
# FROM python:3.9

RUN git clone --depth 1 https://github.com/mne-tools/mne-bids-pipeline /mne-bids-pipeline

# Setup user and working directories
# RUN groupadd -r user && useradd -r -g user mne-user
RUN mkdir -p /work /data
# RUN chown mne-user /work /data

WORKDIR /work
VOLUME ["/work", "/data"]

# Create the environment:
RUN conda create --name=mnenv --channel=conda-forge mne
RUN conda install -n mnenv pip

# Activate conda environment
SHELL ["conda", "run", "-n", "mnenv", "/bin/bash", "-c"]
RUN python -c "import mne; mne.sys_info()"

# Install requirements for MNE-BIDS-Pipeline
RUN pip install -r https://raw.githubusercontent.com/mne-tools/mne-bids-pipeline/main/requirements.txt

# Environment variables: Replace PIPELINE_CONFIG for your own pipeline config
ENV PIPELINE_ROOT=/mne-bids-pipeline
ENV PIPELINE_CONFIG=/work/config.py

RUN echo -e '#!/bin/bash\nconda run -n mnenv python /mne-bids-pipeline/run.py --config=$PIPELINE_CONFIG' > /usr/bin/run && \
    chmod +x /usr/bin/run

# USER mne-user
# ENTRYPOINT sh
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mnenv", "/bin/bash"]
# ENTRYPOINT run

# FROM base as test
# CMD ["cd", "/mne-bids-pipeline"]
# CMD ["conda", "run", "--no-capture-output", "-n", "mnenv", "make", "check"]
