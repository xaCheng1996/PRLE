import random

import cv2
import os
import torch
import numpy as np
import yaml
import logging
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from heatmap_fusion import show_fig_list


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/yaml/heatmap_finetune.yaml')
    args = parser.parse_args()
    return args


def mask_to_binary(mask):
    mask_ori = np.uint8(255 * mask)
    ret, mask_ori = cv2.threshold(mask_ori, 1, 255, cv2.THRESH_BINARY)
    # mask_ori = cv2.cvtColor(mask_ori, cv2.COLOR_BGR2RGB)
    # mask_ori = 255 - mask_ori
    mask_ori = binary_mask_to_random_pixel(mask_ori)
    # mask_ori = mask_ori/np.max(mask_ori)
    return mask_ori


def binary_mask_to_random_pixel(mask):
    c, h, w = mask.shape
    mask_new = mask
    for t in range(c):
        for i in range(h):
            for j in range(w):
                if mask[t][i][j] == 255:
                    mask_new[t][i][j] = random.randint(1, 255)
    return mask_new


def des_mask(progress, mask):
    mask_dict = {}
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] > 0.00:
                mask_dict[(i, j)] = mask[i][j]
    mask_per = sorted(mask_dict.items(), key=lambda kv: (kv[1], kv[0]))
    mask_per.reverse()
    alpha = progress[0]
    mask_num = int(alpha * len(mask_per))
    mask_point_alpha = mask_per[:mask_num]
    return mask_point_alpha


def finetune_mask(fusion_heatmap, progress):
    mask_finetune = des_mask(progress, fusion_heatmap)
    return mask_finetune

def main():
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    img_list = []
    fusion_heatmap_path = config['fusion_heatmap_path']
    ori_path = config['ori_path']
    save_heatmap_path = config['save_heatmap_path']
    for root, dirs, f_names in os.walk(fusion_heatmap_path):
        for f_name in f_names:
            if f_name.lower().endswith('npy'):
                img_list.append(os.path.join(root, f_name))
    img_list.sort()
    for f_heatmap in img_list:
        show = []
        f_fusion_read = np.load(os.path.join(fusion_heatmap_path, f_heatmap))
        img_name = f_heatmap.replace('.npy', '')
        sample_face = Image.open(os.path.join(ori_path, img_name))
        # f_fusion_read = cv2.applyColorMap(np.uint8(255 * f_fusion_read), cv2.COLORMAP_JET)
        # f_fusion_read = cv2.cvtColor(f_fusion_read, cv2.COLOR_BGR2RGB)
        # f_fusion_read = f_fusion_read/np.max(f_fusion_read)
        # show.append(f_fusion_read)
        # show_fig_list(show)
        mask_finetune = finetune_mask(f_fusion_read, config)
        m = mask_finetune[0]
        # m = np.load(os.path.join(save_m_path, "%s.npy" % img_name))
        sample_resize = sample_face.resize(Image.fromarray(m).size)
        mask_image = add_mask(sample_resize, m)
        mask_image = Image.fromarray(mask_image)
        mask_image.save(os.path.join(save_heatmap_path, img_name))

def add_mask(image, mask):
    # empty = Image.new('RGBA', (image.size), 255)
    # mask = Image.fromarray(np.uint8(mask*255), 'L')
    # new_image = PIL.Image.composite(image, image, mask)
    image = np.array(image)
    mask = np.uint8(mask*255)
    new_image = cv2.bitwise_or(image, image, mask=mask)
    return new_image


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()