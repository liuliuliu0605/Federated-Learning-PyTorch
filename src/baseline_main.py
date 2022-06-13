#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import os
import torch.distributed as dist
from torch.utils.data import DataLoader

from utils import get_dataset, get_device, init_seed
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# import warnings
# warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = args_parser()

    init_seed(args.seed)

    # To use DDP (Distributed Data Parallel)
    # e.g., torchrun --nproc_per_node 2 baseline_main.py
    # e.g., python -m torch.distributed.launch --nproc_per_node 2 baseline_main.py
    world_size, local_rank = os.environ.get('WORLD_SIZE'), os.environ.get('LOCAL_RANK')
    world_size = 1 if world_size is None else int(world_size)
    local_rank = -1 if local_rank is None else int(local_rank)

    device = get_device(args)

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'lr':
        # Logistic regression
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = LR(dim_in=len_in, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # Prepare for DP/DDP
    if local_rank != -1:
        assert world_size <= torch.cuda.device_count()
        print("Rank %d: Let's use" % local_rank, torch.cuda.device_count(), "GPUs!")
        dist.init_process_group(backend='nccl')
        # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
        torch.cuda.manual_seed_all(args.seed)
        # initialize distributed data parallel (DDP)
        global_model = DDP(global_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    elif args.data_parallel and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # initialize data parallel (DP), only work in single machine with multiple GPUs
        global_model = torch.nn.DataParallel(global_model, device_ids=list(range(torch.cuda.device_count())))

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    sampler = DistributedSampler(train_dataset) if local_rank != -1 else None
    trainloader = DataLoader(train_dataset, sampler=sampler, batch_size=int(64/world_size),
                             shuffle=True if sampler is None else False)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        if local_rank != -1:
            sampler.set_epoch(epoch)

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_idx = batch_idx * world_size + local_rank if local_rank != -1 else batch_idx

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader) / world_size, loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Make sure only one process conducts the following steps.
    if local_rank > 0:
        exit(0)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
