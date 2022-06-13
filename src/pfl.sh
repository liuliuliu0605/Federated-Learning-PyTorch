#!/bin/bash

source /data/magnolia/venv_set/torch/bin/activate

## fmnist
dataset=fmnist; model=lr; optm=sgd; lr=0.01; gpu=0
topo=complete; sim=0.8;
frac=1.0;
epochs=200;

echo $dataset $model

#for seed in 10
#do
##for mix in 1 2 3 5 7 10
##for mix in 20 30 50 70 100
#for mix in 1 2 3 5 7 10 20 30 50 70 100
#do
#        echo $mix
#        CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --seed $seed --gpu=$gpu --model=$model --dataset=$dataset --iid 0 \
#                                                        --log_location "../logs/${dataset}/${topo}" \
#                                                        --lr=$lr --epochs=$epochs --local_iter=10 --local_bs=10 --optimizer=$optm --frac $frac \
#                                                        --topo $topo --num_clusters=5 --cluster_similarity $sim --mix_ep=$mix \
#                                                        --verbose 0 > pfl_log/$dataset-$sim-$mix 2>&1 &
#        sleep 5s
#done
#done

for seed in 10
do
for mix in 1
do
        echo $mix
        CUDA_VISIBLE_DEVICES=0,1 python pfl_main.py --seed $seed --gpu=$gpu --model=$model --dataset=$dataset --iid 0 \
                                                        --log_location "../logs/${dataset}/${topo}" \
                                                        --lr=$lr --epochs=$epochs --local_iter=10 --local_bs=10 --optimizer=$optm --frac $frac \
                                                        --topo $topo --num_clusters=5 --cluster_similarity $sim --mix_ep=$mix \
                                                        --verbose 0
done
done






#for mix in 1 2 3 5 7 10
##for mix in 20 30 50 70 100
#do
#        echo $mix
#        CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --gpu=$gpu --model=$model --dataset=$dataset --iid 0 \
#                                                        --lr=$lr --epochs=$epochs --local_ep=1 --local_bs=50 --optimizer=$optm --frac $frac \
#                                                        --topo $topo --num_clusters=5 --cluster_similarity $sim --mix_ep=$mix \
#                                                        --verbose 0 > pfl_log/$dataset-$sim-$mix 2>&1 &
#        sleep 5s
#done

## cifar10
#dataset=cifar; model=cnn; optm=sgd; lr=0.01; gpu=1
#topo=ring; sim=0.0;
#frac=1.0;
#epochs=400;
#
#echo $dataset $model
#for mix in 20 30 50 70 100
##for mix in 1 2 3 5 7 10
#do
#        echo $mix
#        CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --gpu=$gpu --model=$model --dataset=$dataset --iid 0 \
#                                                        --lr=$lr --epochs=$epochs --local_ep=1 --local_bs=10 --optimizer=$optm --frac $frac \
#                                                        --topo $topo --num_clusters=5 --cluster_similarity $sim --mix_ep=$mix \
#                                                        --verbose 0 > pfl_log/$dataset-$sim-$mix 2>&1 &
#        sleep 5s
#done

#dataset=cifar
#model=cnn
#sim=0.0
#optm=sgd
#lr=0.01
#frac=1.0
#gpu=0
#topo=complete
#
#for mix in 70 100
##for mix in 100 150 200 250
##for mix in 0
##for mix in 1 2 3 5 7 10 20 30 50
#do
#        echo $mix
#        CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --gpu=$gpu --model=$model --dataset=$dataset --iid 0 \
#                                                        --lr=$lr --epochs=1000 --local_ep=1 --local_bs=10 --optimizer=$optm --frac $frac \
#                                                        --topo $topo --num_clusters=5 --cluster_similarity $sim --mix_ep=$mix \
#                                                        --verbose 0 > pfl_log/$dataset-$sim-$mix 2>&1 &
#        sleep 5s
#done