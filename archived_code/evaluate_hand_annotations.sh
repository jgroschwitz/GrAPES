#!/bin/bash

parser=$1
datasets=( "word_disambiguation"  "berts_mouth" "unseen_senses_new_sentences" "unseen_roles_new_sentences" "unbounded_dependencies" "winograd" )

for dataset in "${datasets[@]}"
    do
		. scripts/single_evaluation.sh $parser $dataset
	done
	