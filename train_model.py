# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import time
import torch
import copy
import shutil
import json
import psutil
import logging

# local imports
import dataset
import utils
from pytorch_utils import metadata
from pytorch_utils import lr_scheduler
import unet_model

def eval_model(model, pt_dataset, criterion, device, epoch, train_stats, split_name, args):
    logging.info('Evaluating model against {} dataset'.format(split_name))
    start_time = time.time()

    dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
    model.eval()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            images, masks = tensor_dict

            # move data to the device
            images = images.to(device)
            masks = masks.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                batch_loss = criterion(logits, masks)
                train_stats.append_accumulate('{}_loss'.format(split_name), batch_loss.item())
                pred = torch.argmax(logits, dim=1)  # NCHW
                accuracy = torch.mean((pred == masks).type(torch.FloatTensor))
                train_stats.append_accumulate('{}_accuracy'.format(split_name), accuracy.item())

    # close out the accumulating stats with the specified method
    train_stats.close_accumulate(epoch, '{}_loss'.format(split_name), method='avg')
    # this adds the avg loss to the train stats
    train_stats.close_accumulate(epoch, '{}_accuracy'.format(split_name), method='avg')
    train_stats.add(epoch, '{}_wall_time'.format(split_name), time.time() - start_time)


def train_epoch(model, pt_dataset, optimizer, criterion, device, epoch, train_stats, args):

    dataloader = torch.utils.data.DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)

    model.train()
    batch_count = len(dataloader)

    if args.cycle_factor is None or args.cycle_factor == 0:
        cyclic_lr_scheduler = None
    else:
        cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(args.learning_rate / args.cycle_factor), max_lr=(args.learning_rate * args.cycle_factor), step_size_up=int(batch_count / 2), cycle_momentum=False)

    alpha = 1.2 * args.adv_eps
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_time = time.time()
    for batch_idx, tensor_dict in enumerate(dataloader):
        optimizer.zero_grad()

        images, masks = tensor_dict

        # move data to the device
        images = images.to(device)
        masks = masks.to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            # only apply attack to attack_prob of the batches
            if args.adv_prob and np.random.rand() <= args.adv_prob:
                # initialize perturbation randomly
                delta = utils.get_uniform_delta(images.shape, args.adv_eps, requires_grad=True)

                logits = model(images + delta)

                # compute metrics
                batch_train_loss = criterion(logits, masks)
                scaler.scale(batch_train_loss).backward()

                # get gradient for adversarial update
                grad = delta.grad.detach()

                # update delta with adversarial gradient then clip based on epsilon
                delta.data = utils.clamp(delta + alpha * torch.sign(grad), -args.adv_eps, args.adv_eps)

                # add updated delta and get model predictions
                delta = delta.detach()
                logits = model(images + delta)
            else:
                logits = model(images)

            # compute metrics
            batch_loss = criterion(logits, masks)
            # perform the backward pass of the network to compute gradients
            batch_loss.backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            train_stats.append_accumulate('train_loss', batch_loss.item())
            pred = torch.argmax(logits, dim=1)  # NCHW
            accuracy = torch.mean((pred == masks).type(torch.FloatTensor))
            train_stats.append_accumulate('train_accuracy', accuracy.item())

            if cyclic_lr_scheduler is not None:
                cyclic_lr_scheduler.step()

            if batch_idx % 100 == 0:
                # log loss and current GPU utilization
                cpu_mem_percent_used = psutil.virtual_memory().percent
                gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                logging.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

    # close out the accumulating stats with the specified method
    train_stats.close_accumulate(epoch, 'train_loss', method='avg')  # this adds the avg loss to the train stats
    train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')
    train_stats.add(epoch, 'train_wall_time', time.time() - start_time)


def train(args):
    if os.path.exists(args.output_dirpath):
        logging.info("output directory exists, deleting")
        shutil.rmtree(args.output_dirpath)
    os.makedirs(args.output_dirpath)
    # add the file based handler to the logger
    logging.getLogger().addHandler(logging.FileHandler(filename=os.path.join(args.output_dirpath, 'log.txt')))

    try:
        # attempt to get the slurm job id and log it
        logging.info("Slurm JobId: {}".format(os.environ['SLURM_JOB_ID']))
    except KeyError:
        pass

    try:
        # attempt to get the hostname and log it
        import socket
        hn = socket.gethostname()
        logging.info("Job running on host: {}".format(hn))
    except RuntimeError:
        pass

    logging.info("Starting model train with args:")
    logging.info(vars(args))  # log the args

    # write the args configuration to disk
    logging.info("writing args to config.json")
    with open(os.path.join(args.output_dirpath, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, ensure_ascii=True, indent=2)

    # figure out what compute device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_start_time = time.time()

    if args.val_image_dirpath is not None:
        train_dataset = dataset.SemanticSegmentationDataset(image_dirpath=args.train_image_dirpath,
                                                           mask_dirpath=args.train_mask_dirpath,
                                                           img_ext=args.image_extension,
                                                           mask_ext=args.mask_extension,
                                                           transform=dataset.SemanticSegmentationDataset.TRANSFORM_TRAIN,
                                                           tile_size=args.tile_size)
        val_dataset = dataset.SemanticSegmentationDataset(image_dirpath=args.val_image_dirpath,
                                                            mask_dirpath=args.val_mask_dirpath,
                                                            img_ext=args.image_extension,
                                                            mask_ext=args.mask_extension,
                                                            transform=dataset.SemanticSegmentationDataset.TRANSFORM_TEST,
                                                            tile_size=args.tile_size)
    else:
        full_dataset = dataset.SemanticSegmentationDataset(image_dirpath=args.train_image_dirpath,
                                                           mask_dirpath=args.train_mask_dirpath,
                                                           img_ext=args.image_extension,
                                                           mask_ext=args.mask_extension,
                                                           transform=None,
                                                           tile_size=args.tile_size)
        # split the dataset into train/val
        logging.info("No validation dataset provided, splitting train dataset into train/val using a val_fraction={}".format(args.val_fraction))
        train_dataset, val_dataset = full_dataset.train_val_split(val_fraction=args.val_fraction)
        train_dataset.set_transforms(dataset.SemanticSegmentationDataset.TRANSFORM_TRAIN)
        val_dataset.set_transforms(dataset.SemanticSegmentationDataset.TRANSFORM_TEST)

    # this will only do anything if args.test_every_n_steps is not None and > 0
    train_dataset.set_test_every_n_steps(args.test_every_n_steps, args.batch_size)
    logging.info("Training dataset size: {} images".format(len(train_dataset)))
    logging.info("Validation dataset size: {} images".format(len(val_dataset)))

    # create the model
    model = unet_model.UNet(n_channels=train_dataset.get_number_channels(), n_classes=args.num_classes)
    # Move model to device
    model.to(device)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()

    plateau_scheduler = lr_scheduler.EarlyStoppingReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduction_factor, patience=args.patience, threshold=args.loss_eps, max_num_lr_reductions=args.num_lr_reductions, lr_reduction_callback=None)

    # setup the metadata capture object
    train_stats = metadata.TrainingStats()

    best_model = None
    epoch = -1
    # train epochs until loss or accuracy converges
    while not plateau_scheduler.is_done():
        epoch += 1
        logging.info("Epoch: {}".format(epoch))

        train_stats.export(args.output_dirpath)  # update metrics data on disk
        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)

        train_epoch(model, train_dataset, optimizer, criterion, device, epoch, train_stats, args)

        eval_model(model, val_dataset, criterion, device, epoch, train_stats, 'val', args)

        # val_loss = train_stats.get_epoch('val_loss', epoch=epoch)
        val_accuracy = train_stats.get_epoch('val_accuracy', epoch=epoch)
        plateau_scheduler.step(val_accuracy)

        # update global metadata stats
        train_stats.add_global('train_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
        train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
        train_stats.add_global('num_epochs_trained', epoch)

        # handle early stopping when loss converges
        if plateau_scheduler.is_equiv_to_best_epoch:
            logging.info('Updating best model with epoch: {} accuracy: {}'.format(epoch, val_accuracy))
            best_model = copy.deepcopy(model)
            # update the global metrics with the best epoch
            train_stats.update_global(epoch)
            # save a state dict (weights only) version of the model
            torch.save(best_model.state_dict(), os.path.join(args.output_dirpath, 'model-state-dict.pt'))

    # move the model back to the GPU (saving moved the best model back to the cpu)
    if args.test_image_dirpath is not None:
        logging.info("Test data dirpath provided, constructing PyTorch dataset for the provided data.")
        test_dataset = dataset.SemanticSegmentationDataset(image_dirpath=args.test_image_dirpath,
                                                          mask_dirpath=args.test_mask_dirpath,
                                                          img_ext=args.image_extension,
                                                          mask_ext=args.mask_extension,
                                                          transform=dataset.SemanticSegmentationDataset.TRANSFORM_TEST,
                                                          tile_size=args.tile_size)

        eval_model(best_model, test_dataset, criterion, device, train_stats.best_epoch, train_stats, 'test', args)

        # update the global metrics with the best epoch, to include test stats
        train_stats.update_global(train_stats.best_epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logging.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_dirpath)  # update metrics data on disk
    train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    # save a python class embedded version of the model
    torch.save(best_model, os.path.join(args.output_dirpath, 'model.pt'))
    # save a state dict (weights only) version of the model
    torch.save(best_model.state_dict(), os.path.join(args.output_dirpath, 'model-state-dict.pt'))




