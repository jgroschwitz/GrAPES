#!/bin/bash

parser=$1
datasets=( "read_by" "see_with" "bought_for" "keep_from" "give_up_in" )

for dataset in "${datasets[@]}"
    do
		. scripts/single_evaluation.sh $parser $dataset
	done
	