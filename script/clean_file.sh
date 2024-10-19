#!/bin/bash

echo "Cleaning up files..."

# Remove all files in the figures directory
find dat/train/ -type f -delete
find dat/test/ -type f -delete