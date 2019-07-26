from util.common import *
from util.image_operations import *
import easydict as edict
import torch.nn.functional as F
from torchvision import transforms
from loader.data_loader import places365_imagenet_loader
from util.places365_categories import places365_categories
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


settings = edict.EasyDict({
    "GPU" : True,
    "IMG_SIZE" : 224,
    "CNN_MODEL" : MODEL_DICT['resnet18_fc_ma'],
    "DATASET" : 'places365',
    "DATASET_PATH" : DATASET_PATH['places365'],
    "MODEL_FILE": 'zoo/resnet18_places365_ma_t5.pth',#'zoo/resnet18_places365.pth.tar',
    "WORKERS" : 16,
    "BATCH_SIZE" : 1,

})

val_loader = places365_imagenet_loader(settings, 'val', 1)
model = settings.CNN_MODEL(num_classes=365)
checkpoint = torch.load(settings.MODEL_FILE, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
mp1 = nn.Sequential(*list(model.children())[:8])
preprocess = transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]
             )])


def read_ndcsv(file):
    node_infos = {}
    with open(file, 'r') as f:
        for line in f.readlines()[1:]:
            infos = line.strip().split(',')
            node_infos[int(infos[0])] = (infos[2], float(infos[3]))
    return node_infos

node_infos = read_ndcsv('tmp/tally_t5.csv')


for c in range(365):
    ind_ = np.nonzero(model.rfc.weight[c] > 1e-3)
    wc, indc = model.rfc.weight[c, ind_].sort(0, True)
    inds = ind_[indc].squeeze()

    print("{}\t".format(places365_categories[c]), end='')
    for ind in inds.numpy():
        print("{:.3f} ({} {} {:.2f})\t".format(model.rfc.weight[c, ind].item(), ind, node_infos[ind+1][0], node_infos[ind+1][1]), end='')
    print()

1/0
with torch.no_grad():
    lucky_dog = np.random.choice(len(val_loader.dataset.imgs))
    lucky_dog = 0
    raw_img_path, target = val_loader.dataset.imgs[lucky_dog]
    org_img = PIL.Image.open(raw_img_path)
    img_tensor = preprocess(org_img)
    input_var = torch.autograd.Variable(img_tensor.unsqueeze(0))
    height, width = org_img.size
    org_img = np.array(org_img)
    fmap = mp1(input_var)

    scores = fmap.view(512, 49).mean(1) * model.rfc.weight[target]
    inds = scores.sort(0, True)[1][:5]
    for ind in inds:
        ind = ind.item()
        print("{} {:.4f}:{:.2f}".format(node_infos[ind+1][0], node_infos[ind+1][1], scores[ind]))
    imgs = [imagalize(fmap[0, i].numpy()) for i in inds]
    vis_cams = [(0.3 * cv2.applyColorMap(imresize(img, (width, height)), cv2.COLORMAP_JET)[:, :, ::-1] + org_img * 0.5)
                for img in imgs]
    vis_arr = [PIL.Image.fromarray(org_img)] + [PIL.Image.fromarray(img.astype(np.uint8)) for img in vis_cams if type(img) == np.ndarray]
    imsave('tmp/cam_{}.jpg'.format(lucky_dog), imconcat(vis_arr, margin=2))