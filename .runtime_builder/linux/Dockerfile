FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic build deps + newer GCC (9)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        g++-9 \
        wget curl git ca-certificates \
        locales \
        libcurl4-openssl-dev libssl-dev libxml2-dev
RUN rm -rf /var/lib/apt/lists/*

# Make g++-9 the default
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# Sanity check
RUN ldd --version && g++ --version
