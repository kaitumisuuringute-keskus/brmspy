FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7

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
        locales \
        # extra libs needed for building Python
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libgdbm-dev \
        libnss3-dev \
        liblzma-dev \
        tk-dev \
        uuid-dev && \
    rm -rf /var/lib/apt/lists/*

# Add toolchain PPA and install g++-9
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends g++-9 && \
    rm -rf /var/lib/apt/lists/*

# Make g++-9 the default
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# Build and install Python 3.12 from source
RUN mkdir -p /tmp/python-build && cd /tmp/python-build && \
    curl -L "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o python.tgz && \
    tar -xzf python.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-lto --prefix=/opt/python-${PYTHON_VERSION} && \
    make -j"$(nproc)" && \
    make install && \
    ln -s /opt/python-${PYTHON_VERSION}/bin/python3.12 /usr/local/bin/python3.12 && \
    ln -s /opt/python-${PYTHON_VERSION}/bin/pip3.12 /usr/local/bin/pip3.12 && \
    cd / && rm -rf /tmp/python-build

# Sanity check
RUN ldd --version && g++ --version && python3.12 --version
