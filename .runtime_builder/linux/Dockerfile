FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Base deps + software-properties-common so we can add PPAs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        wget \
        curl \
        git \
        ca-certificates \
        lsb-release \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        locales && \
    rm -rf /var/lib/apt/lists/*

# Add toolchain PPA and install g++-9
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends g++-9 && \
    rm -rf /var/lib/apt/lists/*

# Make g++-9 the default
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# Sanity check
RUN ldd --version && g++ --version
