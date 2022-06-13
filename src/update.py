#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import get_device, lr_decay
import copy
import numpy as np


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, train_percent=1.0):  #, last_local_weights=None):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_test(dataset, list(idxs), train_percent=train_percent)
        self.device = get_device(args)
        # Default criterion set to cross entropy loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.criterion = nn.SmoothL1Loss().to(self.device)  # TODO, L = 1/beta
        # self.criterion = nn.MSELoss().to(self.device)
        # self.last_local_weights = last_local_weights

    def train_test(self, dataset, idxs, train_percent=1.0):
        # split indexes for train and test (100, 0) in default
        idxs_train = idxs[:int(train_percent*len(idxs))]
        idxs_test = idxs[int(train_percent*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(int(len(idxs_test)/10), 2), shuffle=False)
        return trainloader, testloader

    def estimate_bak(self, model):
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        initial_weight_array = torch.cat([w.reshape(-1) for w in model.parameters()]).detach().cpu().numpy()
        initial_grad_array_groups = []
        second_weight_array_groups = []
        second_grad_array_groups = []
        loss_groups = []
        counter = 0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # first forward
            model.zero_grad()
            log_probs = model(images)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            loss_groups.append(loss.item())
            initial_grad_array = torch.cat([w.grad.reshape(-1) for w in model.parameters()])
            initial_grad_array_groups.append(initial_grad_array.cpu().numpy())

            # second forward
            model_copy = copy.deepcopy(model)
            optimizer.step()
            model.zero_grad()
            log_probs = model(images)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            second_weight_array = torch.cat([w.reshape(-1) for w in model.parameters()])
            second_grad_array = torch.cat([w.grad.reshape(-1) for w in model.parameters()])
            second_weight_array_groups.append(second_weight_array.detach().cpu().numpy())
            second_grad_array_groups.append(second_grad_array.detach().cpu().numpy())

            # counter += 1
            # if counter >= 10:
            #     break

            # restore initial model
            model = model_copy

        sigma = np.linalg.norm(np.var(initial_grad_array_groups, axis=0))**2 # TODO, training process ?
        G = np.linalg.norm(np.mean(initial_grad_array_groups, axis=0))**2  # TODO, training process ?
        L = np.linalg.norm(np.mean(initial_grad_array_groups, axis=0)-np.mean(second_grad_array_groups, axis=0)) /\
                      np.linalg.norm(initial_weight_array-np.mean(second_weight_array_groups, axis=0))  # ???
        gradient = np.mean(initial_grad_array_groups, axis=0)

        # L = (grad_array_groups[0]-grad_array_groups[1]).norm(2).item() /\
        #               (weight_array_groups[0]-weight_array_groups[1]).norm(2).item()
        params = [L, G, sigma]
        # weight_array_groups = []
        # grad_array_groups = []
        # loss_groups = []
        # counter = 0
        # for batch_idx, (images, labels) in enumerate(self.trainloader):
        #     images, labels = images.to(self.device), labels.to(self.device)
        #     model.zero_grad()
        #     log_probs = model(images)
        #     loss = self.criterion(log_probs, labels)
        #     loss.backward()
        #     loss_groups.append(loss.item())
        #     weight_array = torch.cat([w.reshape(-1) for w in model.parameters()])
        #     grad_array = torch.cat([w.grad.reshape(-1) for w in model.parameters()])
        #     weight_array_groups.append(weight_array)
        #     grad_array_groups.append(grad_array)
        #     optimizer.step()
        #     counter += 1
        #     if counter >= 2:
        #         break
        #
        # # ||\nabla f(w_0) - \nabla f(w_1)|| / ||w_0 - w_1||
        # # params = [L]
        # params.append((grad_array_groups[0]-grad_array_groups[1]).norm(2).item() /\
        #               (weight_array_groups[0]-weight_array_groups[1]).norm(2).item())
        return params, gradient

    def estimate_bak2(self, model, global_round, num_params=1):
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_decay(self.args.lr, global_round),
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_decay(self.args.lr, global_round),
                                         weight_decay=1e-4)

        avg_grads_list = []
        avg_grads_qdrtc_list = []
        weights_list = []
        loss_list = []

        for epoch in range(2):
            loss_groups = []
            avg_grads = {name: torch.zeros_like(param, device=self.device) for name, param in model.named_parameters()}
            avg_grads_qdrtc = {name: torch.tensor(0., device=self.device) for name, param in model.named_parameters()}

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # calculate stochastic gradients
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                loss_groups.append(loss.item())

                for name, param in model.named_parameters():
                    avg_grads[name] += param.grad / len(self.trainloader)  # full gradient
                    avg_grads_qdrtc[name] += param.grad.norm()**2 / len(self.trainloader)

            # save weights and grads
            weights_list.append({name: copy.deepcopy(param) for name, param in model.named_parameters()})
            avg_grads_list.append(avg_grads)
            avg_grads_qdrtc_list.append(avg_grads_qdrtc)
            loss_list.append(np.mean(loss_groups))

            # use full gradients to optimize
            for name, param in model.named_parameters():
                param.grad = avg_grads[name]
            optimizer.step()

        G, grad_delta, weight_delta, x = 0, 0, 0, 0
        for name in avg_grads_list[0]:
            grad_delta += (avg_grads_list[0][name] - avg_grads_list[1][name]).norm()**2
            weight_delta += (weights_list[0][name] - weights_list[1][name]).norm()**2
            G += avg_grads_list[0][name].norm()**2
            x += avg_grads_qdrtc_list[0][name]

        L = torch.sqrt(grad_delta / weight_delta)
        sigma = x - G  # D[X] = E[X^2] - E[X]^2

        # params = {'L': L, 'G': G, 'sigma': sigma}
        # params = [L.item()/num_params, G.item()/num_params, sigma.item()/num_params]
        params = [loss_list[0], L.item(), G.item()/num_params, sigma.item()/num_params]
        init_grads = torch.cat([grad.reshape(-1) for grad in avg_grads_list[0].values()]).cpu().numpy()


        return params, init_grads

    def estimate(self, model):
        avg_loss = 0
        avg_grad = {name: torch.zeros_like(param, device=self.device) for name, param in model.named_parameters()}
        avg_grad_qdrtc = {name: torch.tensor(0., device=self.device) for name, param in model.named_parameters()}
        num_of_batches = len(self.trainloader)

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            model.zero_grad()
            log_probs = model(images)
            loss = self.criterion(log_probs, labels)
            avg_loss += loss.item()
            loss.backward()

            for name, param in model.named_parameters():
                avg_grad[name] += param.grad / num_of_batches
                avg_grad_qdrtc[name] += param.grad.norm() ** 2 / num_of_batches


        avg_loss /= num_of_batches
        EX_2, E_X2 = 0, 0
        for name, param in model.named_parameters():
            EX_2 += avg_grad[name].norm() ** 2
            E_X2 += avg_grad_qdrtc[name]

        sigma = (E_X2 - EX_2).item()

        return avg_loss, sigma, avg_grad


    def update_weights(self, model, global_round, rt_params=None, num_params=1):
        # Set mode to train model
        model.train()
        epoch_loss = []
        lr = lr_decay(self.args.lr, global_round)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        # local updates
        counter = 0
        iterations_of_epoch = len(self.trainloader)
        total_iterations = self.args.local_iter if self.args.local_iter > 0 else iterations_of_epoch * self.args.local_ep

        # estimate parameters
        if isinstance(rt_params, dict) and global_round == 0:
            avg_loss, sigma, avg_grad = self.estimate(model)

            avg_grad_flatten = None
            for name, param in model.named_parameters():
                avg_grad_flatten = avg_grad_flatten[name].reshape(-1) if avg_grad_flatten is None \
                    else torch.cat([avg_grad_flatten, avg_grad[name].reshape(-1)])

            rt_params['estmt_params'] = [avg_loss, sigma / num_params]
            rt_params['init_grad_flatten'] = avg_grad_flatten.cpu().numpy()

        while counter < total_iterations:

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                counter += 1

                if counter >= total_iterations:
                    break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_bak5(self, model, global_round, rt_params=None, num_params=1):
        # Set mode to train model
        model.train()
        epoch_loss = []
        lr = lr_decay(self.args.lr, global_round)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        # local updates
        counter = 0
        iterations_of_epoch = len(self.trainloader)
        total_iterations = self.args.local_iter if self.args.local_iter > 0 else iterations_of_epoch * self.args.local_ep

        # whether to estimate parameters
        is_estimate = isinstance(rt_params, dict)

        if is_estimate:
            weight_delta = 0
            grads_list = []
            avg_grads = {name: torch.zeros_like(param, device=self.device) for name, param in model.named_parameters()}
            avg_grads_qdrtc = {name: torch.tensor(0., device=self.device) for name, param in model.named_parameters()}

        # for iter in range(self.args.local_ep):
        while counter < total_iterations:

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)

                # first iteration
                if is_estimate and counter == 0:
                    init_loss = loss.item()
                    init_weights_grads = {name: (param.clone().detach(), param.grad.clone().detach())
                                          for name, param in model.named_parameters()}

                if iter == 0:  # first epoch
                    for name, param in model.named_parameters():
                        avg_grads[name] += param.grad / len(self.trainloader)
                        avg_grads_qdrtc[name] += param.grad.norm()**2 / len(self.trainloader)

                # grads_list.append(torch.cat([param.grad.clone().detach().reshape(-1) for _, param in model.named_parameters()]))

                optimizer.step()

                if isinstance(rt_params, dict):
                    for name, param in model.named_parameters():
                        weight_delta += (param - init_weights_grads[name][0]).norm() ** 2

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                counter += 1

                if counter >= total_iterations:
                    break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # if isinstance(rt_params, dict) and self.last_local_weights is not None:
        if isinstance(rt_params, dict):

            grad_delta, G = 0, 0
            init_grad_vector = None
            E_X2, EX_2 = 0, 0

            for name, param in model.named_parameters():
                grad_delta += ((init_weights_grads[name][0] - param)/lr - init_weights_grads[name][1]*counter).norm() ** 2
                G += init_weights_grads[name][1].norm() ** 2
                init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
                    else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])

                EX_2 += avg_grads[name].norm() ** 2
                E_X2 += avg_grads_qdrtc[name]


            # L = torch.sqrt(grad_delta / weight_delta / counter).item()
            # G = G.item()

            # sigma = torch.stack(grads_list).var(axis=0, unbiased=False).sum().item()
            sigma = (E_X2 - EX_2).item()
            # rt_params['estmt_params'] = [init_loss, L, G / num_params, sigma / num_params]
            # rt_params['estmt_params'] = [init_loss, L, sigma / num_params]
            rt_params['estmt_params'] = [init_loss, sigma / num_params]
            rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()
            rt_params['weight_delta'] = weight_delta.item()

        else:
            pass
            # init_grad_vector = None
            # for name, param in model.named_parameters():
            #     init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
            #         else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])
            # rt_params['estmt_params'] = [init_loss, 0]
            # rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_bak4(self, model, global_round, rt_params=None, num_params=1):
        # Set mode to train model
        model.train()
        epoch_loss = []
        lr = lr_decay(self.args.lr, global_round)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        # local updates
        counter = 0
        weight_delta = 0
        grads_list = []

        avg_grads = {name: torch.zeros_like(param, device=self.device) for name, param in model.named_parameters()}
        avg_grads_qdrtc = {name: torch.tensor(0., device=self.device) for name, param in model.named_parameters()}

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)

                if counter == 0 and isinstance(rt_params, dict):  # first iteration
                    images_bak, labels_bak = images, labels
                    init_loss = loss.item()
                    init_weights_grads = {name: (param.clone().detach(), param.grad.clone().detach())
                                          for name, param in model.named_parameters()}

                if iter == 0:  # first epoch
                    for name, param in model.named_parameters():
                        avg_grads[name] += param.grad / len(self.trainloader)
                        avg_grads_qdrtc[name] += param.grad.norm()**2 / len(self.trainloader)

                # grads_list.append(torch.cat([param.grad.clone().detach().reshape(-1) for _, param in model.named_parameters()]))

                optimizer.step()

                if isinstance(rt_params, dict):
                    for name, param in model.named_parameters():
                        weight_delta += (param - init_weights_grads[name][0]).norm() ** 2

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                counter += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if isinstance(rt_params, dict):

            model.zero_grad()
            log_probs = model(images_bak)
            loss = self.criterion(log_probs, labels_bak)
            loss.backward()

            grad_delta, G = 0, 0

            init_grad_vector = None
            E_X2, EX_2 = 0, 0

            grad_delta2 = weight_delta2 = 0

            for name, param in model.named_parameters():
                grad_delta += ((init_weights_grads[name][0] - param)/lr - init_weights_grads[name][1]*counter).norm() ** 2

                weight_delta2 += (param - init_weights_grads[name][0]).norm() ** 2
                grad_delta2 += (param.grad - init_weights_grads[name][1]).norm() ** 2

                G += init_weights_grads[name][1].norm() ** 2
                init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
                    else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])

                EX_2 += avg_grads[name].norm() ** 2
                E_X2 += avg_grads_qdrtc[name]


            # L = torch.sqrt(grad_delta / weight_delta / counter).item()
            L = torch.sqrt(grad_delta2 / weight_delta2).item()
            # G = G.item()

            # sigma = torch.stack(grads_list).var(axis=0, unbiased=False).sum().item()
            sigma = (E_X2 - EX_2).item()
            # rt_params['estmt_params'] = [init_loss, L, G / num_params, sigma / num_params]
            # rt_params['estmt_params'] = [init_loss, L, sigma / num_params]
            rt_params['estmt_params'] = [init_loss, sigma / num_params]
            rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()
            rt_params['weight_delta'] = weight_delta.item()

            # rt_params['grad_delta2'] = grad_delta2.item()
            # rt_params['weight_delta2'] = weight_delta2.item()
            # rt_params['L'] = weight_delta2.item() / grad_delta2.item()

        else:
            pass
            # init_grad_vector = None
            # for name, param in model.named_parameters():
            #     init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
            #         else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])
            # rt_params['estmt_params'] = [init_loss, 0]
            # rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_bak3(self, model, global_round, rt_params=None, num_params=1):
        # Set mode to train model
        model.train()
        epoch_loss = []
        lr = lr_decay(self.args.lr, global_round)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        # local updates
        counter = 0
        weight_delta = 0
        grads_list = []

        avg_grads = {name: torch.zeros_like(param, device=self.device) for name, param in model.named_parameters()}
        avg_grads_qdrtc = {name: torch.tensor(0., device=self.device) for name, param in model.named_parameters()}

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)

                if counter == 0 and isinstance(rt_params, dict):  # first iteration
                    init_loss = loss.item()
                    init_weights_grads = {name: (param.clone().detach(), param.grad.clone().detach())
                                          for name, param in model.named_parameters()}

                if iter == 0:  # first epoch
                    for name, param in model.named_parameters():
                        avg_grads[name] += param.grad / len(self.trainloader)
                        avg_grads_qdrtc[name] += param.grad.norm()**2 / len(self.trainloader)

                # grads_list.append(torch.cat([param.grad.clone().detach().reshape(-1) for _, param in model.named_parameters()]))

                optimizer.step()

                if isinstance(rt_params, dict):
                    for name, param in model.named_parameters():
                        weight_delta += (param - init_weights_grads[name][0]).norm() ** 2

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                counter += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # if isinstance(rt_params, dict) and self.last_local_weights is not None:
        if isinstance(rt_params, dict):

            grad_delta, G = 0, 0
            init_grad_vector = None
            E_X2, EX_2 = 0, 0

            for name, param in model.named_parameters():
                grad_delta += ((init_weights_grads[name][0] - param)/lr - init_weights_grads[name][1]*counter).norm() ** 2
                G += init_weights_grads[name][1].norm() ** 2
                init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
                    else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])

                EX_2 += avg_grads[name].norm() ** 2
                E_X2 += avg_grads_qdrtc[name]


            # L = torch.sqrt(grad_delta / weight_delta / counter).item()
            # G = G.item()

            # sigma = torch.stack(grads_list).var(axis=0, unbiased=False).sum().item()
            sigma = (E_X2 - EX_2).item()
            # rt_params['estmt_params'] = [init_loss, L, G / num_params, sigma / num_params]
            # rt_params['estmt_params'] = [init_loss, L, sigma / num_params]
            rt_params['estmt_params'] = [init_loss, sigma / num_params]
            rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()
            rt_params['weight_delta'] = weight_delta.item()

        else:
            pass
            # init_grad_vector = None
            # for name, param in model.named_parameters():
            #     init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
            #         else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])
            # rt_params['estmt_params'] = [init_loss, 0]
            # rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_bak2(self, model, global_round, rt_params=None, num_params=1):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_decay(self.args.lr, global_round),
                                        momentum=0)
                                        # momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_decay(self.args.lr, global_round),
                                         weight_decay=1e-4)

        # local updates
        counter = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)

                if counter == 0 and isinstance(rt_params, dict):
                    images_bak, labels_bak = images, labels
                    init_loss = loss.item()
                    init_weights_grads = {name: (param.clone().detach(), param.grad.clone().detach())
                                          for name, param in model.named_parameters()}
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                counter += 1
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # one more iteration to compute the latest gradients with the same batch
        # if isinstance(rt_params, dict) and self.last_local_weights is not None:
        if isinstance(rt_params, dict):
            # model2 = copy.deepcopy(model)
            # model2.load_state_dict(self.last_local_weights)
            model.zero_grad()
            log_probs = model(images_bak)
            loss = self.criterion(log_probs, labels_bak)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model2.parameters(), max_norm=1, norm_type=2)

            weight_delta, grad_delta, G = 0, 0, 0
            init_grad_vector = None

            for name, param in model.named_parameters():
                weight_delta += (param - init_weights_grads[name][0]).norm() ** 2
                grad_delta += (param.grad - init_weights_grads[name][1]).norm() ** 2
                G += init_weights_grads[name][1].norm() ** 2
                init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
                    else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])

            L = 1 #torch.sqrt(grad_delta / weight_delta).item()
            # print("++++++++++++++", L)
            # G = G.item()
            sigma = 0
            # rt_params['estmt_params'] = [init_loss, L, G / num_params, sigma / num_params]
            rt_params['estmt_params'] = [init_loss, L, sigma / num_params]
            rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()
        else:
            init_grad_vector = None
            for name, param in model.named_parameters():
                init_grad_vector = init_weights_grads[name][1].reshape(-1) if init_grad_vector is None \
                    else torch.cat([init_grad_vector, init_weights_grads[name][1].reshape(-1)])
            rt_params['estmt_params'] = [init_loss, 0, 0]
            rt_params['init_grad_vector'] = init_grad_vector.cpu().numpy()

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_bak(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_decay(self.args.lr, global_round),
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_decay(self.args.lr, global_round),
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct, batches = 0.0, 0.0, 0.0, 0.0

        # for batch_idx, (images, labels) in enumerate(self.testloader):
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            batches += 1

        accuracy = correct/total
        return accuracy, loss/batches


# class LocalUpdate(object):
#     def __init__(self, args, dataset, idxs, logger):
#         self.args = args
#         self.logger = logger
#         self.trainloader, self.validloader, self.testloader = self.train_val_test(
#             dataset, list(idxs))
#         self.device = get_device(args)
#         # Default criterion set to NLL loss function
#         self.criterion = nn.NLLLoss().to(self.device)
#
#     def train_val_test(self, dataset, idxs):
#         """
#         Returns train, validation and test dataloaders for a given dataset
#         and user indexes.
#         """
#         # split indexes for train, validation, and test (80, 10, 10)
#         idxs_train = idxs[:int(0.8*len(idxs))]
#         idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
#         idxs_test = idxs[int(0.9*len(idxs)):]
#
#         trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
#                                  batch_size=self.args.local_bs, shuffle=True)
#         validloader = DataLoader(DatasetSplit(dataset, idxs_val),
#                                  batch_size=int(len(idxs_val)/10), shuffle=False)
#         testloader = DataLoader(DatasetSplit(dataset, idxs_test),
#                                 batch_size=int(len(idxs_test)/10), shuffle=False)
#         return trainloader, validloader, testloader
#
#     def update_weights(self, model, global_round):
#         # Set mode to train model
#         model.train()
#         epoch_loss = []
#
#         # Set optimizer for the local updates
#         if self.args.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
#                                         momentum=0.5)
#         elif self.args.optimizer == 'adam':
#             optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
#                                          weight_decay=1e-4)
#
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.trainloader):
#                 images, labels = images.to(self.device), labels.to(self.device)
#
#                 model.zero_grad()
#                 log_probs = model(images)
#                 loss = self.criterion(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 if self.args.verbose and (batch_idx % 10 == 0):
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         global_round, iter, batch_idx * len(images),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), loss.item()))
#                 self.logger.add_scalar('loss', loss.item())
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
#
#     def inference(self, model):
#         """ Returns the inference accuracy and loss.
#         """
#
#         model.eval()
#         loss, total, correct = 0.0, 0.0, 0.0
#
#         for batch_idx, (images, labels) in enumerate(self.testloader):
#             images, labels = images.to(self.device), labels.to(self.device)
#
#             # Inference
#             outputs = model(images)
#             batch_loss = self.criterion(outputs, labels)
#             loss += batch_loss.item()
#
#             # Prediction
#             _, pred_labels = torch.max(outputs, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()
#             total += len(labels)
#
#         accuracy = correct/total
#         loss = loss/total
#         return accuracy, loss


def test_inference(args, model, test_dataset, idxs=None):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct, batches = 0.0, 0.0, 0.0, 0.0

    device = get_device(args)

    criterion = nn.CrossEntropyLoss().to(device)

    if idxs is None:
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    else:
        testloader = DataLoader(DatasetSplit(test_dataset, idxs), batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        batches += 1

    accuracy = correct/total
    return accuracy, loss/batches
