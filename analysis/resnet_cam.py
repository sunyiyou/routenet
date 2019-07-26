from util.common import *
from util.image_operations import *
import easydict as edict
from torchvision import transforms
from loader.data_loader import places365_imagenet_loader
from sklearn.manifold import TSNE

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

def test0():
    with torch.no_grad():
        lucky_dog = np.random.choice(len(val_loader.dataset.imgs))
        raw_img_path, target = val_loader.dataset.imgs[28329]
        org_img = PIL.Image.open(raw_img_path)
        img_tensor = preprocess(org_img)
        input_var = torch.autograd.Variable(img_tensor.unsqueeze(0))
        height, width = org_img.size
        org_img = np.array(org_img)
        fmap = mp1(input_var)
        inds = fmap.view(512, 49).sum(1).sort(0, True)[1][:20]
        imgs = [imagalize(fmap[0,i].numpy()) for i in inds]
        vis_cams = [(0.3 * cv2.applyColorMap(imresize(img, (width, height)), cv2.COLORMAP_JET)[:, :, ::-1] + org_img * 0.5) for img in imgs]
        vis_arr = [PIL.Image.fromarray(img.astype(np.uint8)) for img in vis_cams if type(img) == np.ndarray]
        imsave('tmp/cam_{}.jpg'.format(lucky_dog), imconcat(vis_arr, margin=2))

for i in range(5):
    with torch.no_grad():
        raw_img_path, target = val_loader.dataset.imgs[i]
        org_img = PIL.Image.open(raw_img_path)
        img_tensor = preprocess(org_img)
        input_var = torch.autograd.Variable(img_tensor.unsqueeze(0))

        fmap = mp1(input_var)

        cams = CAM(fmap, model.fc.weight, [target])
        height, width = org_img.size

        vis_cams = cv2.applyColorMap(imresize(cams[0], (width, height)), cv2.COLORMAP_JET)[:,:,::-1]
        vis_cams = vis_cams * 0.3 + org_img * 0.5


        fmap_mean = imagalize(fmap.squeeze().mean(0).numpy())
        vis_fmap = cv2.applyColorMap(imresize(fmap_mean, (width, height)), cv2.COLORMAP_JET)[:,:,::-1]
        vis_fmap = vis_fmap * 0.3 + org_img * 0.5

        topk5_ind = fmap[0].mean(1).mean(1).topk(5)[1]

        fmap_top5_mean = imagalize(fmap.squeeze()[topk5_ind].mean(0).numpy())
        vis_fmap_top5 = cv2.applyColorMap(imresize(fmap_top5_mean, (width, height)), cv2.COLORMAP_JET)[:, :, ::-1]
        vis_fmap_top5 = vis_fmap_top5 * 0.3 + org_img * 0.5

        vis_arr = [vis_cams, vis_fmap, vis_fmap_top5]
        vis_arr = [PIL.Image.fromarray(img.astype(np.uint8)) for img in vis_arr if type(img) == np.ndarray]
        imsave('tmp/cam_{}.jpg'.format(i), imconcat(vis_arr, margin=2))





    # vmap = torch.matmul(fmap[0].permute(1,2,0), torch.diag(model.fc.weight[895]))
    # U,S,V = torch.svd(nn.functional.relu(vmap,0).reshape(49,512))
    # vmap.reshape(49,512).std(1).reshape(7,7)
    # vmap.reshape(49,512).mean(1).reshape(7,7)

    # XY = TSNE().fit_transform(vmap.reshape(49,512).cpu().detach().numpy().T)
    # FXY = TSNE().fit_transform(fmap.reshape(512, 49).cpu().detach().numpy())


