# This script builds a Docker image for the pipeline tester and runs it.


#!/bin/bash

IMAGE_NAME=pipeline-tester

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR/.."

echo " Building Docker image..."
docker build -f "$SCRIPT_DIR/Dockerfile" -t $IMAGE_NAME "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo " Build failed. Exiting."
    exit 1
fi

echo " Running tests in Docker container..."
docker run --rm $IMAGE_NAME
