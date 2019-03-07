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
        return vote.topk(self.topk, 2)[0].sum(2)

class RouteFcMeanShift(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, seed=1, iters=3, delta=25):
        super(RouteFcMeanShift, self).__init__(in_features, out_features, bias)
        self.seed = seed
        self.iters = iters
        self.delta = delta#nn.Parameter(torch.Tensor(out_features))
        # self.delta.data.fill_(18)


    def forward(self, input):
        b, c, w, h = input.shape
        o, c = self.weight.shape

        feats = input.view(b,c,w*h)
        feats_norm = feats.norm(dim=2)  # b, c
        feats_normalized = feats / (feats_norm[:, :, None] + 1e-15)  # b, c, w*h
        feat_weight = feats_norm / feats_norm.max(1)[0][:, None] #b, c
        inds = (feat_weight[:,None,:] * self.weight).topk(self.seed, dim=2)[1]# b, o, seed

        X = feats_normalized.gather(1, inds.expand(b,o,w*h)) #b, o, w*h
        for i in range(self.iters):
            K = torch.exp(self.delta * (X.matmul(feats_normalized.transpose(2,1)) - 1))  # b, o, c
            K_weighted = K * (feat_weight[:, None, :] * self.weight[None, :, :]) # b, o, c
            X = K_weighted.matmul(feats_normalized) #b, o, w*h
            X = X / (X.norm(dim=2)[:, :, None] + 1e-15)

        K = torch.exp(self.delta * (X.matmul(feats_normalized.transpose(2,1)) - 1))  # b, o, c
        K_weighted = K * (feat_weight[:, None, :] * self.weight[None, :, :])  # b, o, c

        out = K_weighted.sum(2)

        return out

#
# class RouteFcMeanShrink(nn.Linear):
#
#     def __init__(self, in_features, out_features, bias=True, shrink_rate=4):
#         super(RouteFcMeanShrink, self).__init__(in_features, out_features, bias)
#         self.shrink_rate = shrink_rate
#
#     def forward(self, input):
#         b, c, w, h = input.shape
#         o, c = self.weight.shape
#         s = c
#
#         s = int(s/self.shrink_rate)
#         vote = input.view(b, 1, c, w * h) * self.weight.view(1, o, c, 1)
#         distance = torch.abs(vote - vote.mean(2, keepdim=True)).mean(3)
#         ind = distance.topk(s, dim=2, largest=False, sorted=True)[1]
#         ind = ind.unsqueeze(3).expand(b, o, s, w * h)
#
#         s = int(s/self.shrink_rate)
#         vote2 = vote.gather(2, ind)
#         distance2 = torch.abs(vote2 - vote2.mean(2, keepdim=True)).mean(3)
#         ind = distance2.topk(s, dim=2, largest=False, sorted=True)[1]
#         ind = ind.unsqueeze(3).expand(b, o, s, w * h)
#
#         out = vote2.gather(2, ind).view(b, o, s * w * h).mean(2)
#
#         return out


if __name__ == '__main__':
    # model = RouteFcMeanShrink(8, 1, shrink_rate=2)
    # model.weight.data = torch.tensor([1,1,2,2,3,3,4,4]).float().view(1, 8) / 20.
    # input = torch.randint(0,5,(1,8,2,1)).float()
    # model(input)

    model = RouteFcMeanShift(5, 2)
    model.weight.data = torch.tensor([[1, 2, 3, 4, 5],[5, 4, 3, 2, 1]]).float()
    input = torch.randint(0, 5, (1, 5, 2, 2)).float()
    model(input)