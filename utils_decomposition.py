import tensorly
from tensorly.decomposition import partial_tucker
import VBMF
import torch
import torch.nn as nn

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data.numpy()
    unfold_0 = tensorly.base.unfold(weights, 0) 
    unfold_1 = tensorly.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    # print(layer, "VBMF Estimated ranks", ranks)
    if 0 in ranks:
        print("zero rank exists... can not be factorized. return the original layer")
        return layer
    
    core, [last, first] = \
        partial_tucker(layer.weight.data.cpu().numpy(), \
            modes=[0, 1], rank=ranks, init='random')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data
    
    first = first.copy()
    first_layer.weight.data = \
        torch.transpose(torch.Tensor(first), 1, 0).unsqueeze(-1).unsqueeze(-1)
    
    last = last.copy()
    last_layer.weight.data = torch.Tensor(last).unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = torch.Tensor(core)

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def count_layers(model):
    return sum(1 for _ in model.modules() if isinstance(_, nn.Conv2d) or isinstance(_, nn.Linear))