#!/bin/bash
source /data1/Anaconda3/etc/profile.d/conda.sh
conda activate base
cd /data2/image_identification/src
python 47_stable_37.py > output/47_log.txt 2>&1
