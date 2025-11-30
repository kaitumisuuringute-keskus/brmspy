FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8

# 1. Add Toolchain PPA immediately
# We need software-properties-common for add-apt-repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update

# 2. Install All System Dependencies
# This includes:
# - GCC 9 Toolchain (gcc, g++, gfortran)
# - Python build deps
# - R/rpy2/brms deps (BLAS, LAPACK, V8, GLPK, Graphics)
RUN apt-get install -y --no-install-recommends \
    # --- Toolchain ---
    build-essential \
    gcc-9 \
    g++-9 \
    gfortran-9 \
    # --- Utilities ---
    wget \
    curl \
    git \
    ca-certificates \
    lsb-release \
    locales \
    # --- Python Build Dependencies ---
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
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
    uuid-dev \
    # --- R / rpy2 / brms Dependencies ---
    libpcre2-dev \
    libblas-dev \
    liblapack-dev \
    libv8-dev \
    libglpk-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. Configure Locale (Important for R)
RUN locale-gen en_US.UTF-8

# 4. Configure GCC 9 as the Default Toolchain
# This ensures that 'gcc', 'g++', and 'gfortran' all point to version 9.
# This prevents the "ld: cannot find -lgfortran" error.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 100

# 5. Build and install Python 3.12 from source
# (Now compiled using GCC 9 automatically)
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

# 6. Sanity Check
# Verifies that gcc, g++, and gfortran are all aligned versions
RUN echo "== Checking Toolchain Versions ==" && \
    gcc --version | grep "9\." && \
    g++ --version | grep "9\." && \
    gfortran --version | grep "9\." && \
    python3.12 --version