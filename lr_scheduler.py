# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np
import torch.optim

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau

# LR scheduler which does early stopping and exponential decay


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Wrapper around torch.optim.lr_scheduler.ReduceLROnPlateau which exposes a subset of that classes parameters while additionally keeping track of the total number of learning rate reductions which have been applied.
    This class provides two additional callbacks (which must be functions), lr_reduction_callback which gets called when the learning rate is reduced due to the metric having plateaued, and termination_callback which is called when the metric has once again plateaued and the learning rate would have been reduced, but max_num_lr_reductions have already happened. So the scheduler should not reduce the learning rate any more. This is effectively a termination callback, letting the user know when the metric has stopped improving after n lr reductions.

    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        max_num_lr_reductions (int): The maximum number of learning rate reductions which will happend before the termination callback is called.
        lr_reduction_callback (function handle): default = None, function handle which gets called when the learning rate is reduced.
        termination_callback (function handle): default = None, function handle which gets called when the learning rate would have been reduced, but max_num_lr_reductions has been reached.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-3, max_num_lr_reductions=2, lr_reduction_callback=None, termination_callback=None):
        super().__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)

        self.num_lr_reductions = 0
        self.max_num_lr_reductions = max_num_lr_reductions
        self.lr_reduction_callback = lr_reduction_callback
        self.termination_callback = termination_callback
        self.best = np.nan
        self.metric_values = list()
        self.is_equiv_to_best_epoch = False
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def is_done(self):
        return self.num_lr_reductions > self.max_num_lr_reductions

    def step(self, metrics, epoch=None):
        if self.mode not in ['min','max']:
            raise RuntimeError("Invalid mode: {}".format(self.mode))
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if np.isnan(self.best):
            self.best = current

        if np.isnan(self.best):
            self.best = current
        if self.mode == 'min' and current < self.best:
            self.best = current
        if self.mode == 'max' and current > self.best:
            self.best = current

        self.metric_values.append(current)

        error_from_best = np.abs(np.asarray(self.metric_values) - self.best)
        error_from_best[error_from_best < np.abs(self.threshold)] = 0
        if np.all(np.isnan(error_from_best)):
            return
        # unpack numpy array, select first time since that value has happened
        idx = error_from_best == 0
        if np.any(idx):
            best_metric_epoch = np.where(idx)[0][0]
        else:
            best_metric_epoch = 0

        # update the number of "bad" epochs. The (epoch-1) handles 0 based indexing vs natural counting of epochs
        self.num_bad_epochs = (len(self.metric_values) - 1) - best_metric_epoch
        # if this epoch is equivalent in loss to the best
        self.is_equiv_to_best_epoch = error_from_best[-1] == 0

        if self.num_bad_epochs >= self.patience:
            self.num_lr_reductions += 1

            if self.num_lr_reductions > self.max_num_lr_reductions:
                # we have completed the requested number of learning rate reductions, call the provided function handle to let the user respond to this
                if self.termination_callback is not None:
                    self.termination_callback()
            else:
                self._reduce_lr(epoch)
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

                self.metric_values = list()
                if self.lr_reduction_callback is not None:
                    self.lr_reduction_callback()
