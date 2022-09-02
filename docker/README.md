# Containers

The MNE-BIDS-Pipeline can be run as a docker container. The file `docker/Dockerfile` is such an example.
After [installing docker](https://docs.docker.com/get-docker/), open a terminal in the `docker` folder and build the image:

```
sudo docker build -t mne .
```

In this example, two environment variables are available, and two volumes:

```dockerfile
# Exposed volumes:
# `work`: workspace
# `data`: available to mount data volumes independently
VOLUME ["/work", "/data"]

# ...

# Environment variables: Replace PIPELINE_CONFIG for your own pipeline config
ENV PIPELINE_ROOT=/mne-bids-pipeline
ENV PIPELINE_CONFIG=/work/config.py
```

A pipeline can be run from a configuration file on the current folder of the host machine (`./config.py`) as follows:

```
sudo docker run --rm -v .:/work -v /host/data/:/data --name mne-pipeline mne run
```
Details:
 - `sudo`: give docker admin permissions.
 - `docker`: Call the docker program that will run the containerized app.
 - `run`: Take an image (specified at the end) and run a container from it.
 - `--rm`: When the container is finished, remove it.
 - `-v`: Mount a folder from the host machine to the guest container. The directories `/work` and `/data` are made to hold the configuration and data for the pieline.
 - `--name`: give the container a name. This will allow you to control it easily from the docker CLI.
 - `mne`: this is the name of the image built in the last step. It instructs docker to run the container with those instructions.
 - `run`: An container's internal alias to run the pipeline with the configuration file in `/work/config.py`

To run an interactive session, you can add the flags `-it` and don't call the `run` parameter:
```
sudo docker run --rm -it -v /host/work:/work -v /host/data/:/data --name mne-pipeline mne
```

This last command wil land you in a bash root session inside the container. The python binary and requirements are installed.

In your configuration file, your `bids_root` should start with '/data/', so the containerized pipeline might find it in the mounted volume.

You might imagine how you could run many containers on a [docker swarm](https://docs.docker.com/engine/swarm/) to explore different pipelines making use of some distributed computing.