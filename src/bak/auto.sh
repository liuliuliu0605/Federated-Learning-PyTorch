#!/bin/bash

source /data/magnolia/venv_set/torch/bin/activate

sim=0.5
optm=sgd
lr=0.01
frac=1.0
epochs=400
gpu=1

CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --optimizer=$optm --lr=$lr --frac $frac --topo complete --cluster_similarity $sim --verbose 0 >complete_0.5.out 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --optimizer=$optm --lr=$lr --frac $frac --topo star --cluster_similarity $sim --verbose 0 >star_0.5.out 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --optimizer=$optm --lr=$lr --frac $frac --topo ring --cluster_similarity $sim --verbose 0 >ring_0.5.out 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 nohup python pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --optimizer=$optm --lr=$lr --frac $frac --topo independent --cluster_similarity $sim --verbose 0 >independent_0.5.out 2>&1 &
