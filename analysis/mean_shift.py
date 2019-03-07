from util.common import *
from util.image_operations import *
import easydict as edict
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
def vis_tsne(XY, inds=None, w=None):
    if w is not None:
        w = w / w.max()
        for i in range(len(XY)):
            plt.scatter(XY[i, 0], XY[i, 1], c='b', alpha=max(w[i], 0.05))
    else:
        plt.scatter(XY[:,0], XY[:,1], c='b')
    # if inds is not None:
    #     plt.scatter(XY[inds,0], XY[inds,1], c='r')
    plt.show()
# move original points, move
def test1():
    X = feats_normalized
    XY = TSNE(verbose=2).fit_transform(X)
    vis_tsne(XY, topn_ind, w=feats_norm.numpy())

    w = feats_norm
    w = w / w.max()
    for i in range(3):
        K = torch.exp(5 * X.mm(X.t()))
        K = K * w[None, :]
        K_normed = K / K.max()
        X = K_normed.mm(X)
        X = X / X.norm(dim=1)[:, None]
        XY = TSNE(verbose=2).fit_transform(X)
        vis_tsne(XY, topn_ind, w=w)

# fix original points, move
# def test2():
w = feats_norm
w = w / w.max()

inds = (w * fc_weight[label]).topk(n)[1]
X = feats_normalized[inds]
XY = TSNE(verbose=0).fit_transform(feats_normalized)

plt.scatter(XY[inds, 0], XY[inds, 1], c='r', marker='x')
for i in range(len(XY)):
    if i not in inds:
        plt.scatter(XY[i, 0], XY[i, 1], c='b', alpha=max(w[i], 0.05))
plt.show()

for i in range(3):
    K = torch.exp(delta * X.mm(feats_normalized.t())) / np.exp(delta)
    K_weighted = K * (w[None, :] * fc_weight[label])
    X = K_weighted.mm(feats_normalized)
    X = K_weighted.mm(feats_normalized)
    X = X / X.norm(dim=1)[:, None]
    XY = TSNE(verbose=0).fit_transform(torch.cat([X, feats_normalized]))

    plt.scatter(XY[:n, 0], XY[:n, 1], c='r', marker='x')
    for i in range(n, len(XY)):
        plt.scatter(XY[i, 0], XY[i, 1], c='b', alpha=max(w[i-n], 0.05))
    plt.show()

