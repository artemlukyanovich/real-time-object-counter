#!/bin/bash

# Real-time Object Counter - Webcam Script

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Run the application with webcam input
python -m src.main --webcam --config configs/default.yaml
