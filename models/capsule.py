import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class CapConv2d_old(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, in_groups=1, out_groups=1, bias=True):
        super(CapConv2d_old, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.in_groups = in_groups
        self.out_groups = out_groups

        self.weight = nn.Parameter(torch.Tensor(self.out_groups, self.in_groups,
                                                self.out_channels, self.in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_groups, self.in_groups, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, c, w, h = input.shape
        assert c // self.in_groups == self.in_channels
        input = input.view(b, self.in_groups, self.in_channels, w, h)
        if self.bias is not None:
            return torch.cat([sum([F.conv2d(input[:, j, :, :, :], self.weight[i][j], self.bias[i][j], self.stride,
                            self.padding, self.dilation, 1)
                            for j in range(self.in_groups)]) / self.in_groups
                            for i in range(self.out_groups)], dim=1)
        else:
            return torch.cat([sum([F.conv2d(input[:, j, :, :, :], self.weight[i][j], None, self.stride,
                                            self.padding, self.dilation, 1)
                                   for j in range(self.in_groups)]) / self.in_groups
                              for i in range(self.out_groups)], dim=1)




class CapConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, group=1, bias=True):
        super(CapConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.group = group

        self.weight = nn.Parameter(torch.Tensor(out_channels * self.groups, in_channels // self.groups, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels * self.groups))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, input):
        x = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        b, _, w, h = x.shape
        return x.view(b, self.groups, self.out_channels, w, h).mean(1)

def test():
    torch.random.manual_seed(0)
    input = torch.randint(0, 5, (1, 4, 2, 2))
    b, c, w, h = input.shape

    c1 = CapConv2d_old(2, 3, 1, in_groups=2, out_groups=2, bias=True)
    c1.weight.data = torch.floor(c1.weight.data * 5)
    o1 = c1(input)

    c2 = CapConv2d(4, 6, 1, group=2, bias=True)
    c2.bias.data = c1.bias.transpose(0,1).contiguous().view(12)
    c2.weight.data = c1.weight.transpose(0,1).contiguous().view(12,2, 1,1)
    o2 = c2(input)

    assert o1.data == o2.data
    print()

if __name__ == '__main__':
    test()