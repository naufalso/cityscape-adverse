#!/bin/bash

for dataset in fog night rain snow
do
    python tools/dataset_converters/acdc_clean.py data/acdc --split ${dataset} -o ./data/acdc_processed_clean/${dataset}
done