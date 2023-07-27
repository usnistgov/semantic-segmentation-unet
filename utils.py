# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np
import subprocess
import torch
import logging
import inspect


# adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py#L270
def configure_optimizer(model, weight_decay, learning_rate, method='adamw'):
    if weight_decay is None:
        weight_decay = 0.0

    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logging.info("num decayed parameter tensors: {}, with {} parameters".format(len(decay_params), num_decay_params))
    logging.info("num non-decayed parameter tensors: {}, with {} parameters".format(len(nodecay_params), num_nodecay_params))

    if method == 'sgd':
        optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.9, nesterov=False)
        logging.info("Using SGD")
    elif method == 'adamw':
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
        logging.info("Using fused AdamW: {}".format(fused_available))
    else:
        raise RuntimeError("Invalid optimizer: {}".format(method))
    return optimizer


def is_ide_debug():
    # check if IDE is in debug mode, and set num parallel worker to 0
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print("Detected IDE debug mode, force enabling debug mode and setting number of workers to 0")
        return True
    return False


def check_for_ide_debug_mode(args):
    if is_ide_debug():
        args.num_workers = 0


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_info = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    memory_total_info = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    memory_used_percent = np.asarray(memory_used_info) / np.asarray(memory_total_info)
    return memory_used_percent, memory_total_info


def get_uniform_delta(shape, eps, requires_grad=True):
    """
    Generates a troch uniform random matrix of shape within +-eps.
    :param shape: the tensor shape to create.
    :param eps: the epsilon bounds 0+-eps for the uniform random tensor.
    :param requires_grad: whether the tensor requires a gradient.
    """
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta


def clamp(X, l, u, cuda=True):
    """
    Clamps a tensor to lower bound l and upper bound u.
    :param X: the tensor to clamp.
    :param l: lower bound for the clamp.
    :param u: upper bound for the clamp.
    :param cuda: whether the tensor should be on the gpu.
    """

    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)