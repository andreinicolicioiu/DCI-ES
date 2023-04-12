import argparse
from lib2to3.pgen2.token import RSQB # TODO: what is this?

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
from loss_capacity.utils import RSquaredPytorch, MetricsPytorch
from metrics.dci import disentanglement, completeness
from loss_capacity.models import ConvNet, RawData
from copy import deepcopy
import time
import numpy as np
import pdb
import sys
sys.path.insert(1, '../sage/')
import sage


def compute_probe_loss(output, target, factor_sizes, factor_discrete, cross_ent_mult=1.0):
    idx_factor_start = 0
    eps = 0.00001
    all_cross_ent = torch.tensor(0.0, device=target.device)
    all_mse = torch.tensor(0.0, device=target.device)
    for i in range(len(factor_sizes)):
        if factor_discrete[i]:
            target_index = target[:,i] * (factor_sizes[i] - 1) + eps # target is already normalised
            one_hot = output[:, idx_factor_start:idx_factor_start+factor_sizes[i]]
            cross_ent = F.cross_entropy(one_hot, target_index.type(torch.int64))
            all_cross_ent += cross_ent
            idx_factor_start += factor_sizes[i]

        else:
            all_mse += F.mse_loss(output[:,idx_factor_start], target[:,i])
            idx_factor_start += 1
        
    loss = cross_ent_mult * all_cross_ent + all_mse
    return loss, all_cross_ent, all_mse

def train_test_probe(model, dataloader_train, dataloader_val, dataloader_test,
        seed = 0, lr = 0.001, weight_decay=0.0, optim_steps = [0.7, 0.9], epochs = 100, log_interval = 50, save_model = True,
        train_name='model', 
        tb_writer=None, eval_model=False,
        savefolder='./',
        cross_ent_mult=1.0):
    for name, param in model.named_parameters():
        print(f' Learnable [{param.requires_grad}] Layer [{name}] - shape: {param.shape} ')

    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    print(f'Using AdamW with weight_decay: {weight_decay}')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rs = MetricsPytorch(dataloader_val)

    global_step_val = 0
    # EPOCH LOOP
    epoch = 0
    best_score = -99999999
    epochs_ = epochs
    epochs = int(epochs * 295488 // len(dataloader_train.dataset))
    print('-'*120)
    print(f'Train for ***{epochs}*** EPOCHS ( {epochs * len(dataloader_train)} iterations with batch {dataloader_train.batch_size})')
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * optim_steps[0]), int(epochs * optim_steps[1])], gamma=0.1)

    factor_names = dataloader_test.dataset._factor_names
    
    time_loader_end = time.time()

    eval_epochs = np.linspace(0,epochs, epochs_).astype(int)
    print(f'Evaluate after epochs: {eval_epochs}')

    for epoch in range(epochs):
        print(f'starting epoch {epoch}')
        time_loader_start = time.time()
        print(f'Time: loader reset: {time_loader_end - time_loader_start}s')

        tb_writer.add_scalar(f'{train_name}/lr', scheduler.get_last_lr()[-1], global_step_val)
        tb_writer.add_scalar(f'{train_name}/epoch', epoch, global_step_val)
        # TRAINING LOOP
        model.train()
        time_epoch_start = time.time()
        for batch_idx, (data, target) in enumerate(dataloader_train):
            if batch_idx == 0:
                time_epoch_train_start = time.time()
            global_step_val += 1
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss, loss_cross, loss_l2 = compute_probe_loss(output, target, 
                dataloader_train.dataset._factor_sizes,
                dataloader_train.dataset._factor_discrete,
                cross_ent_mult=cross_ent_mult)

            loss.backward()
            optimizer.step()
            if (batch_idx == 0) or ((batch_idx + 1) % (20 * log_interval) == 0):

                print(
                    "[TRAIN] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(dataloader_train.dataset),
                        100.0 * batch_idx / len(dataloader_train),
                        loss.item(),
                    )
                )
                tb_writer.add_scalar(f'{train_name}-Loss/train', loss.item(), global_step_val)
                tb_writer.add_scalar(f'{train_name}-Loss/cross_ent_train', loss_cross.item(), global_step_val)
                tb_writer.add_scalar(f'{train_name}-Loss/l2_train', loss_l2.item(), global_step_val)

            time_epoch_train_end = time.time()
            
        time_epoch_end = time.time()
        training_time = time_epoch_end - time_epoch_start
        just_training_time = time_epoch_train_end - time_epoch_train_start

        print(f'Learning rate: {scheduler.get_last_lr()}')
        scheduler.step()
        # VALIDATION LOOP
        if epoch not in eval_epochs:
            continue
        model.eval()
        test_loss = 0
        correct = 0
        time_test_start = time.time()
        rs.reset()
        # rs_old.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader_val):
                data, target = data.to(device), target.to(device)
                output = model(data)
                rs.acum_stats(output, target)
                loss, _ , _ = compute_probe_loss(output, target, 
                    dataloader_train.dataset._factor_sizes,
                    dataloader_train.dataset._factor_discrete,
                    cross_ent_mult=cross_ent_mult)
                test_loss += (loss * dataloader_val.batch_size)
                
                if (batch_idx == 0) or ((batch_idx + 1) % (10 * log_interval) == 0):
                    print(
                        "[VAL] Evaluating after Epoch: {} \tLoss: {:.6f} Loss2: {:.6f}".format(
                            epoch,
                            loss,
                            test_loss / ( 
                                (batch_idx+1) * dataloader_val.batch_size 
                                *len(dataloader_val.dataset._factor_sizes ))
                        )
                    )

        test_loss = test_loss / (len(dataloader_val.dataset) * len(dataloader_val.dataset._factor_sizes))
        tb_writer.add_scalar(f'{train_name}-Loss/val', test_loss, global_step_val)
        print(
            "\Val set: Average MSE loss: {:.4f})\n".format(
                test_loss
            )
        )
        # get Rsquared and mse
        scores = rs.get_scores()
        val_scores = scores['rsquared']
        all_val_scores = np.array([scores[f'rsquared_{name}'] for name in factor_names ])

        print(f'[Epoch {epoch}] Scores: {scores}')
        tb_writer.add_scalar(f'{train_name}-rsquared_acc/val', scores['rsquared'], global_step_val)

        time_test_end = time.time()
        testing_time = time_test_end - time_test_start

        sample_just = len(dataloader_train.dataset) / just_training_time
        sample_sec = len(dataloader_train.dataset) / training_time
        sample_sec_test = len(dataloader_val.dataset) / testing_time

        print(f'Preparing loading time: {just_training_time} [{sample_just} samples / sec]')
        print(f'Just Training time: {just_training_time} [{sample_just} samples / sec]')
        print(f'Training time: {training_time} [{sample_sec} samples / sec]')
        print(f'Evaluation time: {testing_time} [{sample_sec_test} samples / sec]')

        tb_writer.add_scalar(f'{train_name}/time_samples_per_sec', sample_sec, global_step_val)
        if val_scores > best_score:
            best_score = val_scores
            best_model = deepcopy(model)
        time_loader_end = time.time()

    # TESTING
    model = best_model
    model.eval()
    test_loss = 0
    correct = 0
    time_test_start = time.time()
    rs = MetricsPytorch(dataloader_val)
    rs.reset()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            rs.acum_stats(output, target)
            loss, _, _ = compute_probe_loss(output, target, 
                dataloader_train.dataset._factor_sizes,
                dataloader_train.dataset._factor_discrete,
                cross_ent_mult=cross_ent_mult)
            test_loss += (loss * dataloader_test.batch_size)

            if (batch_idx == 0) or ((batch_idx + 1) % (10 * log_interval) == 0):
                print(
                    "[TEST] Final Testing: {} \tLoss: {:.6f} Loss2: {:.6f}".format(
                        epoch,
                        loss,
                        test_loss / ( 
                            (batch_idx+1) * dataloader_test.batch_size 
                            *len(dataloader_test.dataset._factor_sizes ))
                    )
                )

    test_loss = test_loss / (len(dataloader_test.dataset) * len(dataloader_test.dataset._factor_sizes))
    tb_writer.add_scalar(f'{train_name}-loss/test', test_loss, global_step_val)
    print(
        "\nTest set: Average MSE loss: {:.4f})\n".format(
            test_loss
        )
    )
    # get Rsquared and mse
    scores = rs.get_scores()
    test_final_scores = scores['rsquared']
    all_test_scores = np.array([scores[f'rsquared_{name}'] for name in factor_names ])

    # test_final_scores_old = rs_old.get_scores()['rsquared']

    print(f'[Epoch {epoch}] Scores: {scores}')
    tb_writer.add_scalar(f'{train_name}-rsquared_acc/test', test_final_scores, global_step_val)

    time_test_end = time.time()
    testing_time = time_test_end - time_test_start

    if save_model:
        torch.save(best_model.state_dict(), f"{savefolder}/best_probe_model.pt")

    return best_model, test_loss, all_test_scores, all_val_scores





def evaluate_probe(model, dataloader_test,
        seed = 0,
        log_interval = 50,
        train_name='model', 
        tb_writer=None,
        compute_dci=False,
        group_pixels=False):
    

    for name, param in model.named_parameters():
        print(f' Learnable [{param.requires_grad}] Layer [{name}] - shape: {param.shape} ')

    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    
    # save all testing feats - targets to compute sage-dci 
    inputs, targets = next(iter(dataloader_test))
    len_datasplits = len(dataloader_test.dataset)
    feats = model.encode(inputs.to(device))
    all_feats = np.zeros((len_datasplits, *feats.shape[1:])).astype(np.float32)
    all_targets = np.zeros((len_datasplits, targets.shape[1])).astype(np.float32)
    batch_size = dataloader_test.batch_size

    # TESTING
    model.eval()
    test_loss = 0
    correct = 0
    time_test_start = time.time()
    rs = MetricsPytorch(dataloader_test)
    rs.reset()
    # rs_old.reset()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader_test):
            data, target = data.to(device), target.to(device)
            feats = model.encode(data)
            output= model.get_probe(feats)

            rs.acum_stats(output, target)
            loss, _, _ = compute_probe_loss(output, target, 
                dataloader_test.dataset._factor_sizes,
                dataloader_test.dataset._factor_discrete,
                cross_ent_mult=1.0)
            test_loss += (loss * batch_size)

            if (batch_idx == 0) or ((batch_idx + 1) % (10 * log_interval) == 0):
                print(
                    "[TEST] Final Testing: \tLoss: {:.6f} Loss2: {:.6f}".format(
                        loss,
                        test_loss / ( 
                            (batch_idx+1) * batch_size 
                            *len(dataloader_test.dataset._factor_sizes ))
                    )
                )
            
            if compute_dci:
                all_feats[batch_idx * batch_size: batch_idx * batch_size + data.shape[0]] = feats.detach().cpu().numpy()
                all_targets[batch_idx * batch_size: batch_idx * batch_size + target.shape[0]] = target.detach().cpu().numpy()

    test_loss = test_loss / (len(dataloader_test.dataset) * len(dataloader_test.dataset._factor_sizes))

    print(
        "\nTest set: Average MSE loss: {:.4f})\n".format(
            test_loss
        )
    )
    # get Rsquared and mse
    factor_names = dataloader_test.dataset._factor_names

    scores = rs.get_scores()
    mean_scores = scores['rsquared']
    all_scores = np.array([scores[f'rsquared_{name}'] for name in factor_names ])
    # test_final_scores_old = rs_old.get_scores()['rsquared']

    print(f'Scores: {scores}')

    time_test_end = time.time()
    testing_time = time_test_end - time_test_start

    for i in range(10):
        tb_writer.add_scalar(f'{train_name}-scores/mean_scores', mean_scores, model.count_parameters() + i)

        for ind in range(len(all_scores)):
            tb_writer.add_scalar(f'{train_name}-scores/{factor_names[ind]}',all_scores[ind], model.count_parameters() + i)


    if compute_dci:
        # compute sage-dci scores
        # group_pixels = False
        if group_pixels:
            print('SAGE - use grouped pixels')
            # Group all the colors channels and all 4x4 pixels
            width = 4
            num_superpixels = 64 // width
            groups = []
            for i in range(num_superpixels):
                for j in range(num_superpixels):
                    img = np.zeros((3, 64, 64), dtype=int)
                    img[:,width*i:width*(i+1), width*j:width*(j+1)] = 1
                    img = img.reshape((-1,))
                    groups.append(np.where(img)[0])

        eps = 0.00001
        all_cross_ent = 0
        all_mse = 0
        factor_sizes = dataloader_test.dataset._factor_sizes
        factor_discrete = dataloader_test.dataset._factor_discrete
        if group_pixels:
            #TODO: remove hard codding
            R_sage = np.zeros((256, len(factor_sizes)))
        else:
            R_sage = np.zeros((all_feats.shape[1], len(factor_sizes)))
        # sage seems to give error for input with more than one dim
        all_feats = all_feats.reshape(all_feats.shape[0],-1)

        repr_probe = deepcopy(model)
        repr_probe.model = RawData(latent_dim=all_feats.shape[-1])
        if group_pixels:
            num_marginal_samples = 16
        else:
            num_marginal_samples = 32

        sage_thresh = 0.1 #05


        for i in range(len(factor_sizes)):
            print(f'factor [{i}]: Start calculating SAGE importance')

            if factor_discrete[i]:
                target_factor = all_targets[:,i] * (factor_sizes[i] - 1) + eps # target is already normalised
                # Setup and calculate SAGE
                # get a function that return the i-th factor (probabilities in discrete case)
                model_factor_i = lambda x: repr_probe.forward_factors(x)[i].detach().cpu().numpy()
                if group_pixels:
                    imputer = sage.GroupedMarginalImputerPytorch(model_factor_i, 
                            torch.Tensor(all_feats[:num_marginal_samples]).to(device), 
                            groups)

                else:
                    imputer = sage.MarginalImputerPytorch(model_factor_i, 
                        torch.Tensor(all_feats[:num_marginal_samples]).to(device))

                estimator = sage.PermutationEstimator(imputer, 'cross entropy')
                sage_values = estimator(all_feats, target_factor, batch_size=128, thresh=sage_thresh,
                                verbose=True)
                sage_vals = np.abs(sage_values.values)
                sage_vals = sage_vals / sage_vals.sum()
                R_sage[:,i] = sage_vals
            else:
                target_factor = all_targets[:,i]
                model_factor_i = lambda x: repr_probe.forward_factors(x)[i].detach().cpu().numpy()
                if group_pixels:
                    imputer = sage.GroupedMarginalImputerPytorch(model_factor_i, 
                        torch.Tensor(all_feats[:num_marginal_samples]).to(device), 
                        groups)

                else:
                    imputer = sage.MarginalImputerPytorch(model_factor_i, 
                        torch.Tensor(all_feats[:num_marginal_samples]).to(device))

                estimator = sage.PermutationEstimator(imputer, 'mse')
                sage_values = estimator(all_feats, target_factor, batch_size=128, thresh=sage_thresh,
                                verbose=True)
                sage_vals = np.abs(sage_values.values)
                sage_vals = sage_vals / sage_vals.sum()
                R_sage[:,i] = sage_vals
        
        dci_scores = {}
        dci_scores["informativeness_test"] = mean_scores
        dci_scores["sage_disentanglement"] = disentanglement(R_sage)
        dci_scores["sage_completeness"] = completeness(R_sage)
        dci_scores["sage_dci"] = (dci_scores["informativeness_test"]
                + dci_scores["sage_disentanglement"]
                + dci_scores["sage_completeness"]) / 3.0 

        for name, score in dci_scores.items():
            print(f'DCI: {name}: {score} ')
            for i in range(10):
                tb_writer.add_scalar(f'{train_name}-scores/DCI_sage_{name}', score, model.count_parameters() + i)
    else:
        dci_scores = None

    return model, test_loss, all_scores, dci_scores







def train_test_model(model, dataloader_train, dataloader_val,
        seed = 0, lr = 0.001, weight_decay=0.0, optim_steps = [0.7, 0.9], epochs = 100, log_interval = 50, save_model = True,
        train_name='model', 
        tb_writer=None, eval_model=False,
        savefolder='./'):
    for name, param in model.named_parameters():
        print(f' Learnable [{param.requires_grad}] Layer [{name}] - shape: {param.shape} ')

    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


    rs = RSquaredPytorch(dataloader_val)

    global_step_val = 0
    # EPOCH LOOP 
    # ajust the number of epochs such that we tran on the same number of steps as in mpi3d
    epochs = int(epochs * 295488 // len(dataloader_train.dataset))
    print('-'*120)
    print(f'Train for ***{epochs}*** EPOCHS ( {epochs * len(dataloader_train)} iterations with batch {dataloader_train.batch_size}')
    
    # scheduler = MultiStepLR(optimizer, milestones=[epochs * 7 // 10, epochs * 9 // 10], gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * optim_steps[0]), int(epochs * optim_steps[1])], gamma=0.1)
    print(f'milestones: {[int(epochs * optim_steps[0]), int(epochs * optim_steps[1])]}')
    
    for epoch in range(epochs):
        print(f'starting epoch {epoch}')

        tb_writer.add_scalar(f'{train_name}/lr', scheduler.get_last_lr()[-1], global_step_val)
        tb_writer.add_scalar(f'{train_name}/epoch', epoch, global_step_val)
        # TRAINING LOOP
        model.train()
        time_epoch_start = time.time()
        for batch_idx, (data, target) in enumerate(dataloader_train):
            global_step_val += 1

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = F.mse_loss(output, target)
            # loss = F.mse_loss(output[:,6], target[:,6])


            loss.backward()
            optimizer.step()
            if (batch_idx == 0) or ((batch_idx + 1) % (log_interval) == 0):
            # if batch_idx % 20 == 1:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(dataloader_train.dataset),
                        100.0 * batch_idx / len(dataloader_train),
                        loss.item(),
                    )
                )
                tb_writer.add_scalar(f'{train_name}-Loss/train', loss.item(), global_step_val)
                # if dry_run:
                #     break
        time_epoch_end = time.time()
        training_time = time_epoch_end - time_epoch_start

        print(f'Learning rate: {scheduler.get_last_lr()}')
        scheduler.step()
        # TESTING
        model.eval()
        test_loss = 0
        correct = 0
        time_test_start = time.time()
        rs.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader_val):
                data, target = data.to(device), target.to(device)
                output = model(data)
                rs.acum_stats(output, target)
                test_loss += F.mse_loss(output, target, reduction="sum").item()  # sum up batch loss
                # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()
                
                if (batch_idx == 0) or ((batch_idx + 1) % (10 * log_interval) == 0):
                    print(
                        "Testing after Epoch: {} \tLoss: {:.6f} Loss2: {:.6f}".format(
                            epoch,
                            F.mse_loss(output, target).item(),
                            test_loss / ( 
                                (batch_idx+1) * dataloader_val.batch_size 
                                *len(dataloader_val.dataset._factor_sizes ))
                        )
                    )
                # if dry_run:
                #     break


        test_loss = test_loss / (len(dataloader_val.dataset) * len(dataloader_val.dataset._factor_sizes))
        tb_writer.add_scalar(f'{train_name}-Loss/test', test_loss, global_step_val)
        print(
            "\nTest set: Average MSE loss: {:.4f})\n".format(
                test_loss
            )
        )
        # get Rsquared and mse
        scores = rs.get_scores()
        print(f'[Epoch {epoch}] Scores: {scores}')
        tb_writer.add_scalar(f'{train_name}-rsquared/test', scores['rsquared'], global_step_val)

        time_test_end = time.time()
        testing_time = time_test_end - time_test_start

        sample_sec = len(dataloader_train.dataset) / training_time
        sample_sec_test = len(dataloader_val.dataset) / testing_time

        print(f'Training time: {training_time} [{sample_sec} samples / sec]')
        print(f'Training time: {testing_time} [{sample_sec_test} samples / sec]')

        tb_writer.add_scalar(f'{train_name}/time_samples_per_sec', sample_sec, global_step_val)

    if save_model:
        torch.save(model.state_dict(), f"{savefolder}/best_probe_model.pt")

    return model, test_loss