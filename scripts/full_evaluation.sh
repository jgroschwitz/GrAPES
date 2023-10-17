#!/bin/bash

parsers=( "amrbart" "amparser" )
datasets=( "pp_attachment" "long_lists_short" )

for parser in "${parsers[@]}"
do
    for dataset in "${datasets[@]}"
    do
        . single_evaluation.sh $parser $dataset
	done
done
