import torch
import math
import torch.nn.functional as F
import torch.nn as nn


class RouteFcMaxAct(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=10):
        super(RouteFcMaxAct, self).__init__(in_features, out_features, bias)
        self.topk = topk

    def forward(self, input):
        vote = input[:, None, :] * self.weight
        return vote.topk(10, 2)[0].sum(2)
