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


class RouteFcMeanShrink(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, shrink_rate=8):
        super(RouteFcMeanShrink, self).__init__(in_features, out_features, bias)
        self.shrink_rate = shrink_rate

    def forward(self, input):
        b, c, w, h = input.shape
        o, c = self.weight.shape
        s = c

        s = int(s/self.shrink_rate)
        vote = input.view(b, 1, c, w * h) * self.weight.view(1, o, c, 1)
        distance = torch.abs(vote - vote.mean(2, keepdim=True)).mean(3)
        ind = distance.topk(s, dim=2, largest=False, sorted=True)[1]
        ind = ind.unsqueeze(3).expand(b, o, s, w * h)

        s = int(s/self.shrink_rate)
        vote2 = vote.gather(2, ind)
        distance2 = torch.abs(vote2 - vote2.mean(2, keepdim=True)).mean(3)
        ind = distance2.topk(s, dim=2, largest=False, sorted=True)[1]
        ind = ind.unsqueeze(3).expand(b, o, s, w * h)

        out = vote2.gather(2, ind).view(b, o, s * w * h).mean(2)

        return out


if __name__ == '__main__':
    model = RouteFcMeanShrink(8, 1, shrink_rate=2)
    model.weight.data = torch.tensor([1,1,2,2,3,3,4,4]).float().view(1, 8) / 20.
    input = torch.randint(0,5,(1,8,2,1)).float()
    model(input)