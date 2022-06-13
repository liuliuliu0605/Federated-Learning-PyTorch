#!/bin/bash

source /data/magnolia/venv_set/torch/bin/activate

sim=1
optm=sgd
lr=0.01
frac=1.0
epochs=1000
gpu=1
mix_ep=10


CUDA_VISIBLE_DEVICES=0,1 nohup python wsn_pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --mix_ep=$mix_ep --optimizer=$optm --lr=$lr --frac $frac --topo complete --cluster_similarity $sim --verbose 0 > 1_complete_$mix_ep.out 2>&1 &

#CUDA_VISIBLE_DEVICES=0,1 nohup python wsn_pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --mix_ep=$mix_ep --optimizer=$optm --lr=$lr --frac $frac --topo complete --cluster_similarity $sim --verbose 0 >complete_$sim.out 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1 nohup python wsn_pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --mix_ep=$mix_ep --optimizer=$optm --lr=$lr --frac $frac --topo star --cluster_similarity $sim --verbose 0 >star_$sim.out 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1 nohup python wsn_pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --mix_ep=$mix_ep --optimizer=$optm --lr=$lr --frac $frac --topo ring --cluster_similarity $sim --verbose 0 >ring_$sim.out 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1 nohup python wsn_pfl_main.py --gpu=$gpu --model=cnn --dataset=cifar --gpu=0 --iid 0 --epochs=400 --local_ep=1 --mix_ep=$mix_ep --optimizer=$optm --lr=$lr --frac $frac --topo independent --cluster_similarity $sim --verbose 0 >independent_$sim.out 2>&1 &
