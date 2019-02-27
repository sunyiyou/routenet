from util.common import *

mp1 = nn.Sequential(*list(model.children())[:8])
fmap = mp1(input_var)

vmap = torch.matmul(fmap[0].permute(1,2,0), torch.diag(model.fc.weight[895]))
U,S,V = torch.svd(nn.functional.relu(vmap,0).reshape(49,512))
vmap.reshape(49,512).std(1).reshape(7,7)
vmap.reshape(49,512).mean(1).reshape(7,7)

from sklearn.manifold import TSNE
XY = TSNE().fit_transform(vmap.reshape(49,512).cpu().detach().numpy().T)
FXY = TSNE().fit_transform(fmap.reshape(512, 49).cpu().detach().numpy())


