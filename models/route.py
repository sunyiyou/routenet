import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import time

class RouteFcMaxAct(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=25):
        super(RouteFcMaxAct, self).__init__(in_features, out_features, bias)
        self.topk = topk

    def forward(self, input):
        vote = input[:, None, :] * self.weight
        if self.bias is not None:
            out = vote.topk(self.topk, 2)[0].sum(2) + self.bias
        else:
            out = vote.topk(self.topk, 2)[0].sum(2)
        return out


# class RouteConvMaxAct(nn.Linear):
#
#     def __init__(self, in_features, out_features, bias=True, topk=10):
#         super(RouteConvMaxAct, self).__init__(in_features, out_features, bias)
#         self.topk = topk
#
#     def forward(self, input):
#         b, c, w, h = input.shape
#         avg_input = input.view(b, c, w*h).mean(2)
#         vote = avg_input[:, None, :] * self.weight # b, o, c
#         if self.bias is not None:
#             out = vote.topk(self.topk, 2)[0].sum(2) + self.bias
#         else:
#             out = vote.topk(self.topk, 2)[0].sum(2)
#         return out

class RouteFcCondAct(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk_scale=5):
        super(RouteFcCondAct, self).__init__(in_features, out_features, bias)
        self.topk = topk_scale * out_features

    def forward(self, input):
        vote = input[:, None, :] * self.weight
        b, o, c = vote.shape

        inds = vote.view(b, -1).topk(self.topk, 1)[1]
        mask = torch.zeros(b, o * c).cuda().scatter(1, inds, 1.).view(b, o, c)
        if self.bias is not None:
            out = (vote * mask).sum(2) + self.bias
        else:
            out = (vote * mask).sum(2)

        return out

class RouteFcMeanShift(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, seed=1, iters=3, delta=50):
        super(RouteFcMeanShift, self).__init__(in_features, out_features, bias)
        self.seed = seed
        self.iters = iters
        self.delta = delta#nn.Parameter(torch.Tensor(out_features))
        # self.delta.data.fill_(18)


    def forward(self, input):
        b, c, w, h = input.shape
        o, c = self.weight.shape

        fc_weight = self.weight #/ (self.weight.norm(dim=1, keepdim=True) + 1e-15)

        feats = input.view(b,c,w*h)
        feats_norm = feats.norm(dim=2)  # b, c
        feats_normalized = feats / (feats_norm[:, :, None] + 1e-15)  # b, c, w*h
        feat_weight = feats_norm / feats_norm.max(1)[0][:, None] #b, c
        inds = (feat_weight[:,None,:] * fc_weight).topk(self.seed, dim=2)[1]# b, o, seed

        X = feats_normalized.gather(1, inds.expand(b,o,w*h)) #b, o, w*h
        for i in range(self.iters):
            K = torch.exp(self.delta * (X.matmul(feats_normalized.transpose(2,1)) - 1))  # b, o, c
            K_weighted = K * (feat_weight[:, None, :] * fc_weight[None, :, :]) # b, o, c
            X = K_weighted.matmul(feats_normalized) #b, o, w*h
            X = X / (X.norm(dim=2)[:, :, None] + 1e-15)

        K = torch.exp(self.delta * (X.matmul(feats_normalized.transpose(2,1)) - 1))  # b, o, c
        K_weighted = K.data * (feat_weight[:, None, :] * fc_weight[None, :, :])  # b, o, c

        if self.bias is not None:
            out = K_weighted.sum(2) + self.bias
        else:
            out = K_weighted.sum(2)

        return out

class CG(torch.optim.Optimizer):

    def __init__(self, params, lr=0.1, momentum=0, dampening=0, nesterov=False, K=10):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        self.topk = K
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, target=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # if target is not None and len(target) == 1:
                #     if len(p.shape) == 2:
                #         inds = torch.abs(d_p).topk(self.topk, 1)[1]
                #         mask = torch.zeros_like(d_p).cuda().scatter(1, inds, 1.)
                #         # mask_1 = torch.ones_like(d_p)
                #         # mask_1[target] = -1
                #         d_p *= mask
                #         p.data = (1 - group['lr']) * p.data + group['lr'] * d_p
                #     else:
                #         p.data.add_(group['lr'], d_p)
                # else:
                if len(p.shape) == 2:
                    inds = torch.abs(d_p).topk(self.topk, 1)[1]
                    # inds = d_p.topk(self.topk, 1)[1]
                    mask = torch.zeros_like(d_p).cuda().scatter(1, inds, 1.)

                    # d_p *= mask
                    d_p = mask
                    # d_p = -d_p
                    p.data = (1 - group['lr']) * p.data + group['lr'] * d_p
                else:
                    p.data.add_(group['lr'], d_p)

        return loss


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
