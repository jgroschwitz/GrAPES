#!/bin/bash

dataset=$1

bash -c "cd /amrbart/fine-tune && conda run -n amrbart bash inference-amr.sh ../AMRBART-large-finetuned-AMR3.0-AMRParsing-v2/ /mounted/corpus/${dataset}_amrbart_input.jsonl && cp /amrbart/fine-tune/outputs/Infer-examples-AMRBART-large-AMRParing-bsz16-lr-1e-5-UnifiedInp/val_outputs/test_generated_predictions_0.txt /mounted/amrbart-output/${dataset}.txt"
