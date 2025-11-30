#!/usr/bin/env bash
set -euo pipefail

# Configuration
R_VERSION="${1:-4.5.0}"
UBUNTU_CODENAME="bionic"
CRAN_REPO="https://cloud.r-project.org/bin/linux/ubuntu"

# Helper function for logging
log() {
    echo -e "\033[1;34m[$(date +'%H:%M:%S')] $*\033[0m"
}

error() {
    echo -e "\033[1;31m[ERROR] $*\033[0m" >&2
}

log "Targeting R Version: ${R_VERSION} on Ubuntu ${UBUNTU_CODENAME}"

# ==============================================================================
# STEP 1: PREPARE REPOSITORIES
# ==============================================================================
log "Setting up APT repositories..."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends dirmngr gnupg ca-certificates software-properties-common wget

# Add CRAN GPG key
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9 > /dev/null 2>&1

# Add CRAN Repo
echo "deb ${CRAN_REPO} ${UBUNTU_CODENAME}-cran40/" > /etc/apt/sources.list.d/cran-r.list

apt-get update -qq

# ==============================================================================
# STEP 2: ATTEMPT APT INSTALLATION
# ==============================================================================
log "Checking if R ${R_VERSION} is available via APT..."

# We look for the specific version in the cache (e.g., looking for 4.5.0 in 4.5.0-1.1804.0)
# 'madison' lists versions, we grep for the requested version, and grab the first match column 3
APT_CANDIDATE=$(apt-cache madison r-base-core | grep "${R_VERSION}" | awk '{print $3}' | head -n 1 || true)

if [[ -n "$APT_CANDIDATE" ]]; then
    log "Found candidate in repo: ${APT_CANDIDATE}"
    log "Attempting to install via APT..."
    
    # Try to install. If this command fails, the '||' block will trigger the fallback variable
    if apt-get install -y --no-install-recommends \
        r-base-core="${APT_CANDIDATE}" \
        r-base-dev="${APT_CANDIDATE}" \
        r-base="${APT_CANDIDATE}"; then
            log "Successfully installed R ${R_VERSION} via APT."
            R --version
            exit 0
    else
        error "APT install failed despite candidate presence. Moving to fallback..."
    fi
else
    log "Version ${R_VERSION} not found in APT repositories for ${UBUNTU_CODENAME}."
fi

# ==============================================================================
# STEP 3: FALLBACK - COMPILE FROM SOURCE
# ==============================================================================
log "FALLBACK TRIGGERED: Compiling R ${R_VERSION} from source."
log "This will take some time..."

# 3a. Enable Source Repositories (needed for build-dep)
# In many docker images, deb-src is commented out. We uncomment it.
sed -i -- 's/#\s*deb-src/deb-src/g' /etc/apt/sources.list
apt-get update -qq

# 3b. Install Build Dependencies
log "Installing build dependencies..."
# build-dep grabs libraries needed to compile R (pcre, readline, curl, etc)
apt-get build-dep -y r-base

# Install additional useful tools for building
apt-get install -y -qq curl fort77 xorg-dev liblzma-dev libblas-dev gfortran gcc-multilib gobjc++ texlive-fonts-recommended texinfo

# 3c. Download Source
MAJOR_VERSION=$(echo "$R_VERSION" | cut -d. -f1)
SOURCE_URL="https://cran.r-project.org/src/base/R-${MAJOR_VERSION}/R-${R_VERSION}.tar.gz"
WORK_DIR=$(mktemp -d)

log "Downloading source from ${SOURCE_URL}..."
cd "${WORK_DIR}"

if ! curl -f -O "${SOURCE_URL}"; then
    error "Failed to download source. Check if version ${R_VERSION} exists on CRAN."
    exit 1
fi

tar -xf "R-${R_VERSION}.tar.gz"
cd "R-${R_VERSION}"

# 3d. Configure and Compile
log "Configuring build..."
./configure \
    --prefix=/usr/local \
    --enable-R-shlib \
    --with-blas \
    --with-lapack \
    --with-recommended-packages

log "Compiling (Make)..."
# Use -j to use all available cores
JOBS=$(nproc)
make -j"${JOBS}"

log "Installing..."
make install

log "Registering R shared library path..."

R_HOME="$(R RHOME)"
R_LIBDIR="${R_HOME}/lib"

if [ -d "${R_LIBDIR}" ] && [ -f "${R_LIBDIR}/libR.so" ]; then
    echo "${R_LIBDIR}" > /etc/ld.so.conf.d/R.conf
    ldconfig
    log "Registered ${R_LIBDIR} with dynamic linker."
else
    error "libR.so not found in ${R_LIBDIR} — did the build succeed with --enable-R-shlib?"
    exit 1
fi

# 3e. Clean up
cd /
rm -rf "${WORK_DIR}"

# ==============================================================================
# FINAL VERIFICATION
# ==============================================================================
log "Installation complete."
if command -v R >/dev/null 2>&1; then
    INSTALLED_R=$(R --version | head -n 1)
    log "Verification: ${INSTALLED_R}"
else
    error "R executable not found."
    exit 1
fi