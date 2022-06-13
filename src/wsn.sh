#!/bin/bash

source /data/magnolia/venv_set/torch/bin/activate

dataset=fmnist; model=lr; optm=sgd; lr=0.01; gpu=0
topo=complete; sim=0.2;
frac=1.0;
epochs=600;

echo $dataset $model $topo $sim
#for mix in 70 100
#for mix in 100 150 200 250
#for mix in 0
#for mix in 1 2 3 5 7 10 20 30 50 70
#for mix in 1 2 3 5 7 10 12 15 17 20
for mix in 1 2 3 5 7 10 20 30 50 70 100
do
        echo $mix
        CUDA_VISIBLE_DEVICES=0,1 nohup python wsn_pfl_main.py --fake --log_location "../logs/${dataset}/${topo}" \
                                                              --gpu=$gpu --model=$model --dataset=$dataset --iid 0 \
                                                              --lr=$lr --epochs=$epochs --local_ep=1 --local_bs=10 --optimizer=$optm --frac $frac \
                                                              --topo $topo --num_clusters=5 --cluster_similarity $sim --mix_ep=$mix \
                                                              --verbose 0 > wsn_log/$dataset-$sim-$mix 2>&1 &
        sleep 2s
done
#test
#for mix in 1
#do
#        echo $mix
#        CUDA_VISIBLE_DEVICES=0,1 python wsn_pfl_main.py --fake --log_location "../logs/${dataset}/${topo}" \
#                                                        --gpu=$gpu --model=$model --dataset=$dataset --iid 0 \
#                                                        --lr=$lr --epochs=$epochs --local_ep=1 --local_bs=10 --optimizer=$optm --frac $frac \
#                                                        --topo $topo --num_clusters=5 --cluster_similarity $sim --mix_ep=$mix \
#                                                        --verbose 0
#        sleep 2s
#done