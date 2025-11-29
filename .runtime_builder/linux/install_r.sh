#!/usr/bin/env bash
set -euo pipefail

R_VERSION="${1:-4.5.0}"

echo "Installing R ${R_VERSION} on Ubuntu 18.04 (this is a stub, adapt as needed)"

# Example: CRAN bionic repo (you’ll want to adjust for real pinning)
apt-get update
apt-get install -y --no-install-recommends dirmngr gnupg

# Add CRAN for R (bionic)
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9
echo "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/" > /etc/apt/sources.list.d/cran-r.list

apt-get update
apt-get install -y --no-install-recommends r-base

R --version
