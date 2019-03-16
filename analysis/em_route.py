from util.common import *
from util.image_operations import *
import easydict as edict
import torch.nn.functional as F
from torchvision import transforms
from loader.data_loader import places365_imagenet_loader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


settings = edict.EasyDict({
    "GPU" : True,
    "IMG_SIZE" : 224,
    "CNN_MODEL" : MODEL_DICT['resnet18'],
    "DATASET" : 'places365',
    "DATASET_PATH" : DATASET_PATH['places365'],
    "MODEL_FILE": 'zoo/resnet18_places365.pth.tar',
    "WORKERS" : 16,
    "BATCH_SIZE" : 1,

})

model = settings.CNN_MODEL(num_classes=365)
checkpoint = torch.load(settings.MODEL_FILE, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

n = 1
label = 100
delta = 20

feats_file = 'tmp/resnet18_feats.pth'

if not os.path.exists(feats_file):
    n = 1
    val_loader = places365_imagenet_loader(settings, 'val', 1)

    mp1 = nn.Sequential(*list(model.children())[:8])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    feats = torch.zeros(n, 512, 49)
    for i in range(n):
        with torch.no_grad():
            raw_img_path, target = val_loader.dataset.imgs[i]
            org_img = PIL.Image.open(raw_img_path)
            img_tensor = preprocess(org_img)
            input_var = torch.autograd.Variable(img_tensor.unsqueeze(0))

            fmap = mp1(input_var)
            feats[i, :, :] = fmap.reshape(512, 49).data

    torch.save(feats, feats_file)

else:
    feats = torch.load(feats_file)

fc_weight = model.fc.weight.data
feats_norm = feats[0].norm(dim=1) #512
feats_normalized = feats[0]/(feats_norm[:, None]+1e-15)
topn_ind = feats_norm.topk(n)[1]

# FXY = TSNE().fit_transform((feats[0] * fc_weight[0][:, None]))
def vis_tsne(XY, inds=None, ind_color='r', ind_w=None, w=None):
    if w is not None:
        w = w / w.max()
        for i in range(len(XY)):
            plt.scatter(XY[i, 0], XY[i, 1], c='black', alpha=max(w[i], 0.03))
    else:
        plt.scatter(XY[:,0], XY[:,1], c='black')
    # if inds is not None:
    #     if ind_w is not None:
    #         for i in range(len(inds)):
    #             plt.scatter(XY[inds[i], 0], XY[inds[i], 1], s=3, c=ind_color, alpha=max(ind_w[i]/3, 0))
    #     else:
    #         plt.scatter(XY[inds,0], XY[inds,1], c=ind_color)

    plt.yticks([])
    plt.xticks([])


X = feats_normalized
if not os.path.exists('tmp/tsne.npy'):
    tsne = TSNE(verbose=2).fit_transform(X)
    np.save('tmp/tsne.npy', tsne)
else:
    tsne = np.load('tmp/tsne.npy')
beta_u = 30
beta_a = 2
lembda = 0.001
class_inds = [0, 100, 150]

a = feats_norm
a = a / a.max()  # 512

V = X[None, :, :] * F.relu(fc_weight[class_inds, :, None])  # 3 * 512 * 49
# V = tsne * F.relu(fc_weight[class_inds, :, None])

aj =  torch.zeros(len(class_inds))
mu =  torch.zeros(len(class_inds), V.shape[2])
sigma =  torch.zeros(len(class_inds), V.shape[2])
r_ = torch.ones(len(class_inds), 512) / len(class_inds)

plt.figure()
colors = ['r', 'g', 'b']
for iter in range(3):
    # M steps
    plt.subplot(221)
    vis_tsne(tsne, None, w=feats_norm.numpy())
    r = a * r_
    for j in range(len(class_inds)):
        mu[j, :] = (V[j] * r[j, :, None]).sum(0) / r[j].sum()  # 49
        sigma[j, :] = ((V[j] - mu[j][None, :]) ** 2 * r[j, :, None]).sum(0) / r[j].sum() # 49

        cost = (beta_u + torch.log(sigma[j])) * r[j].sum() # 49
        aj[j] = F.sigmoid(beta_a - cost.sum() * lembda)
        plt.subplot(2,2,j+2)
        pivot_ind = [torch.norm(mu[j] - V[j], dim=1).argmin()]


        # if iter>0:
        #     inds = np.where(r_.argmax(0)==j)[0]
        #     ind_w = r_[j, inds]
        #     plt.scatter(tsne[pivot_ind,0], tsne[pivot_ind,1], c=colors[j])
        # else:
        inds = pivot_ind
        ind_w = None
        vis_tsne(tsne, inds=inds, ind_color=colors[j], ind_w=ind_w, w=(feats_norm * F.relu(fc_weight[class_inds[j], :])).numpy())
    # E steps
    lnp = - (((V - mu[:, None, :]) ** 2) / (2 * (sigma[:, None, :]) + 1e-15)).sum(2) - torch.sqrt(sigma).sum(dim=1)[:, None]  # 3, 512
    r_ = (torch.exp(lnp) * aj[:, None]) / ((torch.exp(lnp) * aj[:, None]).sum(0) + 1e-15)

    plt.tight_layout()
    plt.show()

