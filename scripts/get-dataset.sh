#!/bin/bash
# This script downloads the dataset from the given URL and extracts it to the specified directory.
URL="https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
DATA_DIR="data"
DATA_FILE="medical-charges.csv"

#download
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
fi
if [ ! -f "$DATA_DIR/$DATA_FILE" ]; then
    echo "Downloading dataset..."
    curl -o "$DATA_DIR/$DATA_FILE" "$URL"
else
    echo "Dataset already downloaded."
fi

