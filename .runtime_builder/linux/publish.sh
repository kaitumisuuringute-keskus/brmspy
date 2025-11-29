#!/usr/bin/env sh
set -eu

# Usage:
#   ./publish.sh [TAG] [CONTEXT_DIR] [DOCKERFILE]
#
# Defaults:
#   TAG         = latest
#   CONTEXT_DIR = .runtime_builder
#   DOCKERFILE  = <CONTEXT_DIR>/Dockerfile
#
# Required env vars:
#   - In GitHub Actions:
#       GITHUB_ACTOR
#       GITHUB_TOKEN   (automatically provided)
#       GITHUB_REPOSITORY  (e.g. "kaitumisuuringute-keskus/brmspy")
#   - Locally:
#       GHCR_USER and GHCR_TOKEN (a PAT with "write:packages")

TAG="${1:-latest}"
CONTEXT_DIR="${2:-.runtime_builder/linux}"
DOCKERFILE="${3:-$CONTEXT_DIR/Dockerfile}"

echo "TAG=$TAG"
echo "CONTEXT_DIR=$CONTEXT_DIR"
echo "DOCKERFILE=$DOCKERFILE"

# Determine owner (org/user) for ghcr.io/<OWNER>/<IMAGE_NAME>
if [ -n "${GITHUB_REPOSITORY:-}" ]; then
  # GitHub Actions environment
  OWNER="${GITHUB_REPOSITORY%%/*}"
  USER="${GITHUB_ACTOR}"
  TOKEN="${GITHUB_TOKEN}"
else
  # Local / manual environment
  OWNER="${GHCR_OWNER:-${GHCR_USER:-unknown-owner}}"
  USER="${GHCR_USER:-}"
  TOKEN="${GHCR_TOKEN:-}"
fi

if [ -z "${USER}" ] || [ -z "${TOKEN}" ] || [ "${OWNER}" = "unknown-owner" ]; then
  echo "ERROR: Missing credentials or owner."
  echo "  In GitHub Actions, GITHUB_ACTOR / GITHUB_TOKEN / GITHUB_REPOSITORY must be set."
  echo "  Locally, set GHCR_USER, GHCR_TOKEN and optionally GHCR_OWNER."
  exit 1
fi

IMAGE_NAME="brmspy-runtime-builder"
IMAGE_REF="ghcr.io/${OWNER}/${IMAGE_NAME}:${TAG}"

echo "== Building Docker image =="
echo "  Context:    ${CONTEXT_DIR}"
echo "  Dockerfile: ${DOCKERFILE}"
echo "  Image:      ${IMAGE_REF}"

docker build \
  -t "${IMAGE_REF}" \
  -f "${DOCKERFILE}" \
  "${CONTEXT_DIR}"

echo "== Logging into ghcr.io as ${USER} =="
echo "${TOKEN}" | docker login ghcr.io -u "${USER}" --password-stdin

echo "== Pushing image to ${IMAGE_REF} =="
docker push "${IMAGE_REF}"

echo "== Done =="
