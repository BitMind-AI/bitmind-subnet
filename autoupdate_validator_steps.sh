#!/bin/bash

# Thank you to Namoray of SN19 for their autoupdate implementation!
# THIS FILE CONTAINS THE STEPS NEEDED TO AUTOMATICALLY UPDATE THE REPO
# THIS FILE ITSELF MAY CHANGE FROM UPDATE TO UPDATE, SO WE CAN DYNAMICALLY FIX ANY ISSUES

conda activate bitmind
pip install -e .
python bitmind/download_data.sh
echo "Autoupdate steps complete :)"
