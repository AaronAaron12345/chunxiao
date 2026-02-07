#!/bin/bash
source /data1/Anaconda3/etc/profile.d/conda.sh
conda activate base
cd /data2/image_identification/src
python 48_batch_datasets_multigpu.py > output/48_log.txt 2>&1
