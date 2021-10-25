import os
import numpy as np
import time
import torch
import copy
import sklearn.metrics

import fbf_utils
import metadata


def translate_outputs(x):
    # translate between different model output formats
    if isinstance(x, dict):
        try:
            x = x['out']
        except:
            raise RuntimeError('Unexpected model output, please check what values your model returns on a forward pass')
    return x


def compute_metrics(x, y, stats, name, epoch):
    # convert x to numpy
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # convert x from one hot to class label
    x = np.argmax(x, axis=1)  # assumes NCHW tensor order

    assert x.shape == y.shape

    # flatten into a 1d vector
    x = x.flatten()
    y = y.flatten()

    val = sklearn.metrics.accuracy_score(y, x)
    stats.add(epoch, '{}_accuracy'.format(name), val)

    val = sklearn.metrics.f1_score(y, x)
    stats.add(epoch, '{}_f1'.format(name), val)

    val = sklearn.metrics.jaccard_score(y, x)
    stats.add(epoch, '{}_jaccard'.format(name), val)

    val = sklearn.metrics.confusion_matrix(y, x)
    stats.add(epoch, '{}_confusion'.format(name), val)

    return stats


def eval_model(model, dataloader, criterion, device, epoch, stats, name):
    start_time = time.time()

    avg_loss = 0
    batch_count = len(dataloader)
    model.eval()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            images, masks = tensor_dict

            masks = masks.long()  # CrossEntropy expects longs
            # move data to the device
            images = images.to(device)
            masks = masks.to(device)

            with torch.cuda.amp.autocast():
                pred = model(images)
                pred = translate_outputs(pred)
                batch_train_loss = criterion(pred, masks)
                avg_loss += batch_train_loss.item()

    avg_loss /= batch_count
    wall_time = time.time() - start_time

    stats.add(epoch, '{}_wall_time'.format(name), wall_time)
    stats.add(epoch, '{}_loss'.format(name), avg_loss)


def train_epoch(model, dataloader, optimizer, criterion, lr_scheduler, device, epoch, stats, attack_prob, attack_eps):
    avg_train_loss = 0

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    batch_count = len(dataloader)

    alpha = 1.2 * attack_eps

    start_time = time.time()
    for batch_idx, tensor_dict in enumerate(dataloader):
        optimizer.zero_grad()

        images, masks = tensor_dict

        masks = masks.long()  # CrossEntropy expects longs
        # move data to the device
        images = images.to(device)
        masks = masks.to(device)

        with torch.cuda.amp.autocast():
            # only apply attack to attack_prob of the batches
            if attack_prob and np.random.rand() <= attack_prob:
                # initialize perturbation randomly
                delta = fbf_utils.get_uniform_delta(images.shape, attack_eps, requires_grad=True)

                pred = model(images + delta)
                pred = translate_outputs(pred)

                # compute metrics
                batch_train_loss = criterion(pred, masks)
                scaler.scale(batch_train_loss).backward()

                # get gradient for adversarial update
                grad = delta.grad.detach()

                # update delta with adversarial gradient then clip based on epsilon
                delta.data = fbf_utils.clamp(delta + alpha * torch.sign(grad), -attack_eps, attack_eps)

                # add updated delta and get model predictions
                delta = delta.detach()
                pred = model(images + delta)
                pred = translate_outputs(pred)
            else:
                pred = model(images)
                pred = translate_outputs(pred)

            stats = compute_metrics(pred, masks, stats, name='train', epoch=epoch)

            # compute metrics
            batch_train_loss = criterion(pred, masks)
            if batch_idx % 100 == 0:
                print('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}'.format(batch_idx, batch_count, batch_train_loss.item(), lr_scheduler.get_lr()[0]))

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(batch_train_loss).backward()
        avg_train_loss += batch_train_loss.item()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

    avg_train_loss /= batch_count
    wall_time = time.time() - start_time

    stats.add(epoch, 'train_wall_time', wall_time)
    stats.add(epoch, 'train_loss', avg_train_loss)
    return model


def train(train_dataset, val_dataset, test_dataset, model, output_filepath, learning_rate, num_io_workers, early_stopping_epoch_count=5, loss_eps=1e-3, adv_prob=0.0, adv_eps=4.0/255.0, use_CycleLR=True):
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)


    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=torch.utils.data.RandomSampler(train_dataset), num_workers=num_io_workers)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=4, sampler=torch.utils.data.RandomSampler(val_dataset), num_workers=num_io_workers)
    if test_dataset is not None:
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4, sampler=torch.utils.data.RandomSampler(test_dataset), num_workers=num_io_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    lr_scheduler = None
    if use_CycleLR:
        cycle_factor = 5.0
        lr_scheduler_args = {'base_lr': learning_rate / cycle_factor,
                             'max_lr': learning_rate * cycle_factor,
                             'step_size_up': int(len(train_dl) / 2),
                             'cycle_momentum': False}
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **lr_scheduler_args)

    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    epoch = 0
    done = False
    best_model = model
    stats = metadata.TrainingStats()

    start_time = time.time()

    while not done:
        print('Epoch: {}'.format(epoch))

        model = train_epoch(model, train_dl, optimizer, criterion, lr_scheduler, device, epoch, stats, adv_prob, adv_eps)

        eval_model(model, val_dl, criterion, device, epoch, stats, 'val')

        # handle recording the best model stopping
        val_loss = stats.get('{}_loss'.format('val'))
        error_from_best = np.abs(val_loss - np.min(val_loss))
        error_from_best[error_from_best < np.abs(loss_eps)] = 0
        # if this epoch is with convergence tolerance of the global best, save the weights
        if error_from_best[epoch] == 0:
            print('Updating best model with epoch: {} loss: {}, as its less than the best loss plus eps {}.'.format(epoch, val_loss[epoch], loss_eps))
            best_model = copy.deepcopy(model)

            # update the global metrics with the best epoch
            stats.update_global(epoch)

        stats.add_global('training_wall_time', sum(stats.get('train_wall_time')))
        stats.add_global('val_wall_time', sum(stats.get('val_wall_time')))

        # update the number of epochs trained
        stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        stats.export(output_filepath)

        # handle early stopping
        best_val_loss_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
        if epoch >= (best_val_loss_epoch + early_stopping_epoch_count):
            print("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(epoch))
            done = True

        if not done:
            # only advance epoch if we are not done
            epoch += 1

    if test_dataset is not None:
        print('Evaluating model against test dataset')
        eval_model(model, test_dl, criterion, device, epoch, stats, 'test')
        # update the global metrics with the best epoch, to include test stats
        stats.update_global(epoch)

    wall_time = time.time() - start_time
    stats.add_global('wall_time', wall_time)
    print("Total WallTime: ", stats.get_global('wall_time'), 'seconds')

    stats.export(output_filepath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model, os.path.join(output_filepath, 'model.pt'))

