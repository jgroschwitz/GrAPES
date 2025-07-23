dataset=$1

conda activate cailam

rm -f ../cai-lam-wrapper/AMR-gs/temp/*
rm -f ../cai-lam-wrapper/AMR-gs/checkpoints/amr2.0.bert.gr/ckpt.pt_*
python3 scripts/file_manipulations/make_cailam_input.py $dataset
yes | mv corpus/${dataset}_cailam.txt ../cai-lam-wrapper/AMR-gs/temp/input.txt

cd ../cai-lam-wrapper/AMR-gs/


java -classpath 'stanford-corenlp-4.5.4/*' -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer 1337 &
sleep 5
sh preprocess_raw.sh temp/input.txt
sh work.sh
sh postprocess_2.0.sh checkpoints/amr2.0.bert.gr/ckpt.pt_test_out.pred
cp checkpoints/amr2.0.bert.gr/ckpt.pt_test_out.pred.post ../../amr-challenge/cailam-output/$dataset.txt
cd ../../amr-challenge/

conda deactivate