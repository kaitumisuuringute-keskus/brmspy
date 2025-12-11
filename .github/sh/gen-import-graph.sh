#!/bin/bash
if [ "$(basename "$PWD")" = "sh" ]; then
    cd ../../
fi
mkdir -p ./docs/img/
source .venv/bin/activate
pydeps brmspy \
    --max-bacon=7 \
    --only brmspy \
    --show-deps \
    --cluster \
    --noshow \
    --max-cluster-size=1000 \
    --min-cluster-size=2 \
    --rmprefix brmspy.

# Move and rename the generated SVG
if [ -f brmspy.svg ]; then
    mv brmspy.svg ./docs/img/brmspy-import-graph.svg
else
    echo "brmspy.svg not found - pydeps may have failed."
fi