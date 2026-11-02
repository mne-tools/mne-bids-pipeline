#!/bin/bash
# Single-node SLURM for CircleCI (Linux VM executor; also works without systemd).
# Local repro (from the repo root):
#   docker run --rm -v "$PWD/.circleci":/cc:ro ubuntu:26.04 bash -c 'apt-get update -qq \
#     && apt-get install -qq -y sudo procps >/dev/null && useradd -m ci \
#     && echo "ci ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/ci && runuser -u ci /cc/setup_slurm.sh'
set -eo pipefail

CPUS=$(nproc)  # VM executor nproc is the real vCPU count
MEM=$(( $(free -m | awk '/Mem:/{print $2}') - 512 ))
HOST=$(hostname)

sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends slurm-wlm munge
sudo mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /run/munge /etc/munge
# the packaged slurmctld.service runs as User=slurm, matching SlurmUser in slurm.conf
sudo chown slurm:slurm /var/spool/slurmctld /var/log/slurm
# On a systemd VM the munge package generates a key and starts munged itself
if ! pgrep -x munged >/dev/null; then
    sudo dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key status=none
    sudo chown -R munge:munge /etc/munge /run/munge
    sudo chmod 0400 /etc/munge/munge.key
    sudo runuser -u munge -- munged
fi
sed -e "s/@HOST@/$HOST/g" -e "s/@CPUS@/$CPUS/g" -e "s/@MEM@/$MEM/g" \
    "$(dirname "$0")/slurm.conf" | sudo tee /etc/slurm/slurm.conf >/dev/null
sudo cp "$(dirname "$0")/cgroup.conf" /etc/slurm/cgroup.conf
if [ -d /run/systemd/system ]; then
    sudo systemctl restart slurmctld slurmd  # restart picks up our conf if autostarted
else  # containers (local debugging) have no systemd
    sudo slurmctld
    sudo slurmd
fi
for _ in $(seq 30); do
  [[ "$(sinfo -h -o %T 2>/dev/null)" == "idle" ]] && break
  sleep 1
done
sinfo
srun -N1 true  # fail loudly now rather than mid-test
