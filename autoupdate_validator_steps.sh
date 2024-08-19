#!/bin/bash

# Thank you to Namoray of SN19 for their autoupdate implementation!
# THIS FILE CONTAINS THE STEPS NEEDED TO AUTOMATICALLY UPDATE THE REPO
# THIS FILE ITSELF MAY CHANGE FROM UPDATE TO UPDATE, SO WE CAN DYNAMICALLY FIX ANY ISSUES

echo $CONDA_PREFIX
$CONDA_PREFIX/bin/pip install -e .
$CONDA_PREFIX/bin/python bitmind/download_data.py
echo "Autoupdate steps complete :)"
