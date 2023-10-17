#!/bin/bash

dataset=$1

bash -c "cd /am-parser-app && mkdir output && bash scripts/predict.sh -f -i /mounted/corpus/${dataset}.txt -T AMR-2020 -o /mounted/amparser-output/${dataset}_intermediary_files && cp /mounted/amparser-output/${dataset}_intermediary_files/parserOut.txt /mounted/amparser-output/${dataset}.txt"
