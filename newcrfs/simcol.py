from __future__ import absolute_import, division, print_function

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import cv2

from newcrfs.dataloaders.dataloader import ToTensor
from utils import post_process_depth, flip_lr
from networks.NewCRFDepth import NewCRFDepth


def resize_depth(depth_np, resize_to=None):
    depth_resz = cv2.resize(depth_np, resize_to, interpolation=cv2.INTER_NEAREST)
    return depth_resz


def get_model(checkpoint_path):
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=1.)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    return model


def get_data(image_path):
    transform = transforms.Compose([
        ToTensor(mode='test')
    ])
    focal = 518.8579
    image = Image.open(image_path).convert("RGB")
    newsize = (640, 480)
    image = image.resize(newsize)
    image = np.asarray(image, dtype=np.float32) / 255.0

    sample = {'image': image, 'focal': focal}
    sample = transform(sample)
    sample['image'] = sample['image'].unsqueeze(0)
    return sample


def serve_model(model, image_path, post_process=True, final_sz=(475, 475)):
    sample = get_data(image_path)
    with torch.no_grad():
        image = Variable(sample['image'].cuda())
        depth_est = model(image)

        if post_process:
            image_flipped = flip_lr(image)
            depth_est_flipped = model(image_flipped)
            depth_est = post_process_depth(depth_est, depth_est_flipped)

        pred_depth = depth_est.cpu().numpy().squeeze()  # TODO: check normalization

        depth_final = resize_depth(pred_depth, resize_to=final_sz)
    return depth_final


if __name__ == '__main__':
    model_path = '/home/nfrank/Documents/Projects/NeWCRFs/models/newcrfs_nyu/model-320000-best_abs_rel_0.02256.ckpt'
    img_path = 'files/FrameBuffer_0000.png'
    model = get_model(model_path)
    depth = serve_model(model, img_path)
    a = 1



