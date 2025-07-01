#!/bin/bash

# run this script from the base directory

parser=$1
dataset=$2

# preprocess the dataset file for the required parser
if [ "$parser" = "amparser" ]; then
    #PYTHONPATH=./ python3 scripts/preprocessing/amr_file_to_tokenized_text_input.py $dataset
	docker pull jgroschwitz/amparser:amr30
elif [ "$parser" = "amrbart" ]; then
    PYTHONPATH=./ python3 scripts/preprocessing/to_amrbart_input_format.py $dataset
	docker pull jgroschwitz/amrbart:amr30
fi


# run parser in docker image. This will move the output file into the correct location already
# The --rm flag removes the container after the run, so that it stops taking up memory
echo "Running $parser on $dataset"
out_dir="data/raw/parser_outputs/${parser}-output"
mkdir -p $out_dir
if [ "$parser" = "cailam" ]; then
	. scripts/run_cai_lam_single.sh $dataset &> ${out_dir}/${dataset}_log.txt
else
	cd docker-compose
	docker-compose -f $parser.yml run --rm mycontainer bash -c "cd /mounted/docker-compose && bash ${parser}_single.sh ${dataset}" &> ../${out_dir}${dataset}_log.txt
	cd ..
fi

# run evaluation
cd evaluation
PYTHONPATH=../ python3 single_eval.py $dataset $parser
cd ..