import torch


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

